#!/usr/bin/env python3
"""Profile standard vs flash attention: latency and peak memory vs sequence length.

Measures forward-only and forward+backward pass latency (ms) plus peak memory
(MB) for both MHA implementations across a configurable sweep of sequence
lengths.  Results are printed as a table and saved as a four-panel figure.

Usage
-----
    # from the repo root
    python src/profile_attention.py [OPTIONS]

    # common overrides
    python src/profile_attention.py --device cpu --seq-lengths 64 128 256 512
    python src/profile_attention.py --flash-variant hybrid --batch-size 8 --iters 20

Options
-------
    --device            cuda | cpu            (default: cuda if available, else cpu)
    --batch-size        int                   (default: 4)
    --emb-dim           int                   (default: 768)
    --num-heads         int                   (default: 12)
    --seq-lengths       int [int ...]         (default: 64 128 256 512 1024 2048)
    --flash-block-size  int                   (default: 64)
    --flash-variant     flash | hybrid        (default: flash)
    --warmup            int                   (default: 5)
    --iters             int                   (default: 20)
    --output            path                  (default: attention_profile.png)
    --no-memory                               skip memory profiling (faster)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from components import flash_multihead_attention        # noqa: E402
from components import hybrid_flash_multihead_attention  # noqa: E402

from components.multihead_attention import (  # noqa: E402
    MultiHeadAttentionConfig,
    MultiHeadAttention as StandardMHA,
)

_FLASH_VARIANTS: dict[str, ModuleType] = {
    "flash": flash_multihead_attention,
    "hybrid": hybrid_flash_multihead_attention,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile standard vs flash MHA latency and memory.")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=768,
                   help="Embedding / model dimension (must be divisible by num-heads)")
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--seq-lengths", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--flash-block-size", type=int, default=64,
                   help="Block size for tiled flash attention")
    p.add_argument("--flash-variant", choices=list(_FLASH_VARIANTS), default="flash",
                   help="Which flash-attention implementation to benchmark")
    p.add_argument("--warmup", type=int, default=5,
                   help="Iterations discarded before measurement begins")
    p.add_argument("--iters", type=int, default=20,
                   help="Measured iterations (results are averaged)")
    p.add_argument("--output", default="attention_profile.png",
                   help="Path to save the figure")
    p.add_argument("--no-memory", action="store_true",
                   help="Skip memory profiling (much faster on CPU)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Module factories
# ---------------------------------------------------------------------------

def _build_standard(emb_dim: int, num_heads: int, seq_len: int,
                    device: torch.device) -> StandardMHA:
    cfg = MultiHeadAttentionConfig(
        d_in=emb_dim,
        d_out=emb_dim,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    )
    return StandardMHA(cfg).to(device).eval()


def _build_flash(flash_module: ModuleType, emb_dim: int, num_heads: int,
                 seq_len: int, block_size: int,
                 device: torch.device) -> torch.nn.Module:
    cfg = flash_module.FlashAttentionConfig(
        d_in=emb_dim,
        d_out=emb_dim,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        block_size=block_size,
        qkv_bias=False,
    )
    return flash_module.MultiHeadAttention(cfg).to(device).eval()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup: int, iters: int,
              device: torch.device) -> float:
    """Return mean latency in ms for ``fn()``.

    Uses CUDA events on CUDA devices for sub-millisecond accuracy; falls back
    to ``time.perf_counter`` on CPU.
    """
    is_cuda = device.type == "cuda"

    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)

    if is_cuda:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        return start_ev.elapsed_time(end_ev) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1_000


def time_forward(module: torch.nn.Module, x: torch.Tensor,
                 warmup: int, iters: int, device: torch.device) -> float:
    """Latency (ms) for one forward pass."""
    def _fwd():
        with torch.no_grad():
            module(x)

    return _time_fn(_fwd, warmup, iters, device)


def time_forward_backward(module: torch.nn.Module, batch: int, seq: int,
                          emb_dim: int, warmup: int, iters: int,
                          device: torch.device) -> float:
    """Latency (ms) for one forward + backward pass.

    A fresh input tensor is created for each measurement so that autograd
    graphs do not accumulate across iterations.
    """
    def _fwd_bwd():
        x = torch.randn(batch, seq, emb_dim,
                        device=device, dtype=torch.float32, requires_grad=True)
        out = module(x)
        out.sum().backward()

    return _time_fn(_fwd_bwd, warmup, iters, device)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current process RSS in MB (Linux /proc only; returns nan elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def _peak_memory_mb(fn, device: torch.device) -> float:
    """Peak memory (MB) consumed by ``fn()`` on ``device``.

    On CUDA: uses ``torch.cuda.max_memory_allocated`` (device allocator).
    On CPU:  uses the delta in process RSS from ``/proc/self/status``
             (Linux only; approximate because the OS may not reclaim freed
             pages immediately, so successive calls can be an under-count).
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        fn()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    else:
        before = _rss_mb()
        fn()
        after = _rss_mb()
        delta = after - before
        # RSS delta can be negative if the GC reclaims pages between reads;
        # return 0 rather than a nonsensical negative value.
        return max(delta, 0.0)


def peak_memory_forward(module: torch.nn.Module, x: torch.Tensor,
                        device: torch.device) -> float:
    def _fwd():
        with torch.no_grad():
            module(x)
    return _peak_memory_mb(_fwd, device)


def peak_memory_forward_backward(module: torch.nn.Module, batch: int, seq: int,
                                 emb_dim: int, device: torch.device) -> float:
    def _fwd_bwd():
        x = torch.randn(batch, seq, emb_dim,
                        device=device, dtype=torch.float32, requires_grad=True)
        module(x).sum().backward()
    return _peak_memory_mb(_fwd_bwd, device)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    seq_len: int
    std_fwd_ms: float
    flash_fwd_ms: float
    std_fwdbwd_ms: float
    flash_fwdbwd_ms: float
    std_fwd_mb: float
    flash_fwd_mb: float
    std_fwdbwd_mb: float
    flash_fwdbwd_mb: float


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    seq_lengths: Sequence[int],
    batch_size: int,
    emb_dim: int,
    num_heads: int,
    flash_block_size: int,
    flash_module: ModuleType,
    warmup: int,
    iters: int,
    device: torch.device,
    measure_memory: bool,
) -> list[_Row]:
    results: list[_Row] = []

    for seq in seq_lengths:
        print(f"  seq_len={seq:5d} …", end="", flush=True)
        x = torch.randn(batch_size, seq, emb_dim,
                        device=device, dtype=torch.float32)

        std = _build_standard(emb_dim, num_heads, seq, device)
        flash = _build_flash(
            flash_module, emb_dim, num_heads, seq, flash_block_size, device)

        # ---- timing ----
        std_fwd = time_forward(std, x, warmup, iters, device)
        flash_fwd = time_forward(flash, x, warmup, iters, device)

        std_fwdbwd = time_forward_backward(
            std, batch_size, seq, emb_dim, warmup, iters, device)
        flash_fwdbwd = time_forward_backward(
            flash, batch_size, seq, emb_dim, warmup, iters, device)

        # ---- memory ----
        nan = float("nan")
        if measure_memory:
            std_fwd_mb = peak_memory_forward(std, x, device)
            flash_fwd_mb = peak_memory_forward(flash, x, device)
            std_fb_mb = peak_memory_forward_backward(
                std, batch_size, seq, emb_dim, device)
            flash_fb_mb = peak_memory_forward_backward(
                flash, batch_size, seq, emb_dim, device)
        else:
            std_fwd_mb = flash_fwd_mb = std_fb_mb = flash_fb_mb = nan

        results.append(_Row(
            seq_len=seq,
            std_fwd_ms=std_fwd,
            flash_fwd_ms=flash_fwd,
            std_fwdbwd_ms=std_fwdbwd,
            flash_fwdbwd_ms=flash_fwdbwd,
            std_fwd_mb=std_fwd_mb,
            flash_fwd_mb=flash_fwd_mb,
            std_fwdbwd_mb=std_fb_mb,
            flash_fwdbwd_mb=flash_fb_mb,
        ))

        speedup_fwd = std_fwd / flash_fwd
        speedup_fb = std_fwdbwd / flash_fwdbwd
        print(
            f" fwd {std_fwd:7.2f} vs {flash_fwd:7.2f} ms"
            f" ({speedup_fwd:.2f}×)"
            f" | fwd+bwd {std_fwdbwd:7.2f} vs {flash_fwdbwd:7.2f} ms"
            f" ({speedup_fb:.2f}×)"
        )

        # Free GPU memory between iterations
        del std, flash, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results: list[_Row], flash_label: str,
                measure_memory: bool) -> None:
    hdr = (
        f"{'seq':>6} │ "
        f"{'fwd std':>10} {f'fwd {flash_label}':>12} {'fwd ×':>7} │ "
        f"{'f+b std':>10} {f'f+b {flash_label}':>12} {'f+b ×':>7}"
    )
    if measure_memory:
        hdr += (
            f" │ {'mem fwd std':>12} {f'mem fwd {flash_label}':>14}"
            f" │ {'mem f+b std':>12} {f'mem f+b {flash_label}':>14}"
        )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        sx = r.std_fwd_ms / r.flash_fwd_ms
        fbx = r.std_fwdbwd_ms / r.flash_fwdbwd_ms
        line = (
            f"{r.seq_len:>6} │ "
            f"{r.std_fwd_ms:>9.2f}ms {r.flash_fwd_ms:>11.2f}ms {sx:>6.2f}× │ "
            f"{r.std_fwdbwd_ms:>9.2f}ms {r.flash_fwdbwd_ms:>11.2f}ms {fbx:>6.2f}×"
        )
        if measure_memory:
            line += (
                f" │ {r.std_fwd_mb:>11.1f}MB {r.flash_fwd_mb:>13.1f}MB"
                f" │ {r.std_fwdbwd_mb:>11.1f}MB {r.flash_fwdbwd_mb:>13.1f}MB"
            )
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = {"standard": "#4C72B0", "flash": "#DD8452"}
_ALPHA_FILL = 0.12


def _speedup_line(ax: plt.Axes, xs, std_vals, flash_vals, color="#555555"):
    """Draw speedup ratio on a twin y-axis."""
    ax2 = ax.twinx()
    speedup = [s / f for s, f in zip(std_vals, flash_vals)]
    ax2.plot(xs, speedup, color=color, linestyle="--",
             linewidth=1.2, marker="^", markersize=5, label="speedup")
    ax2.axhline(1.0, color=color, linewidth=0.7, linestyle=":")
    ax2.set_ylabel("Speedup (std / flash)", fontsize=9, color=color)
    ax2.tick_params(axis="y", labelcolor=color, labelsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
    return ax2


def plot_results(results: list[_Row], output: str, device_str: str,
                 batch_size: int, emb_dim: int, num_heads: int,
                 flash_block_size: int, flash_label: str,
                 measure_memory: bool) -> None:
    xs = [r.seq_len for r in results]
    ncols = 2
    nrows = 3 if measure_memory else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    fig.suptitle(
        f"Standard MHA vs {flash_label} Flash Attention   "
        f"[device={device_str}, batch={batch_size}, emb_dim={emb_dim}, "
        f"heads={num_heads}, block_size={flash_block_size}]",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # ---- helpers ----
    def _latency_ax(ax, std_vals, flash_vals, title):
        ax.plot(xs, std_vals, color=_COLORS["standard"],
                marker="o", linewidth=1.8, markersize=5, label="standard")
        ax.plot(xs, flash_vals, color=_COLORS["flash"],
                marker="s", linewidth=1.8, markersize=5, label=flash_label)
        ax.fill_between(xs, std_vals, flash_vals,
                        alpha=_ALPHA_FILL, color="grey")
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Sequence length (tokens)", fontsize=9)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(xs)
        _speedup_line(ax, xs, std_vals, flash_vals)

    def _memory_ax(ax, std_vals, flash_vals, title):
        has_data = not (
            all(v != v for v in std_vals) and all(v != v for v in flash_vals))
        if not has_data:
            ax.text(0.5, 0.5, "Memory data not available\n(CPU w/o /proc)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(title, fontsize=10, fontweight="semibold")
            return
        ax.plot(xs, std_vals, color=_COLORS["standard"],
                marker="o", linewidth=1.8, markersize=5, label="standard")
        ax.plot(xs, flash_vals, color=_COLORS["flash"],
                marker="s", linewidth=1.8, markersize=5, label=flash_label)
        ax.fill_between(xs, std_vals, flash_vals,
                        alpha=_ALPHA_FILL, color="grey")
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Sequence length (tokens)", fontsize=9)
        ax.set_ylabel("Peak memory (MB)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(xs)
        _speedup_line(ax, xs, std_vals, flash_vals, color="#2e8b57")

    # ---- row 0: latency ----
    _latency_ax(
        axes[0, 0],
        [r.std_fwd_ms for r in results],
        [r.flash_fwd_ms for r in results],
        "Forward pass latency",
    )
    _latency_ax(
        axes[0, 1],
        [r.std_fwdbwd_ms for r in results],
        [r.flash_fwdbwd_ms for r in results],
        "Forward + backward pass latency",
    )

    # ---- row 1: speedup bar chart ----
    fwd_speedups = [r.std_fwd_ms / r.flash_fwd_ms for r in results]
    fb_speedups = [r.std_fwdbwd_ms / r.flash_fwdbwd_ms for r in results]

    bar_colors_fwd = [
        _COLORS["flash"] if s > 1 else _COLORS["standard"] for s in fwd_speedups]
    bar_colors_fb = [
        _COLORS["flash"] if s > 1 else _COLORS["standard"] for s in fb_speedups]

    def _bar_ax(ax, speedups, bar_colors, title):
        bars = ax.bar([str(x) for x in xs], speedups, color=bar_colors,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}×", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Sequence length (tokens)", fontsize=9)
        ax.set_ylabel(
            f"Speedup (standard / {flash_label})\n>1 = {flash_label} is faster",
            fontsize=9)
        ax.tick_params(labelsize=8)
        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color=_COLORS["flash"], label=f"{flash_label} faster"),
            mpatches.Patch(color=_COLORS["standard"], label="standard faster"),
        ], fontsize=8)

    _bar_ax(axes[1, 0], fwd_speedups, bar_colors_fwd, "Speedup — forward pass")
    _bar_ax(axes[1, 1], fb_speedups, bar_colors_fb, "Speedup — forward + backward")

    # ---- row 2 (optional): memory ----
    if measure_memory:
        _memory_ax(
            axes[2, 0],
            [r.std_fwd_mb for r in results],
            [r.flash_fwd_mb for r in results],
            "Peak memory — forward pass",
        )
        _memory_ax(
            axes[2, 1],
            [r.std_fwdbwd_mb for r in results],
            [r.flash_fwdbwd_mb for r in results],
            "Peak memory — forward + backward",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ---- device ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")

    flash_module = _FLASH_VARIANTS[args.flash_variant]
    flash_label = args.flash_variant
    measure_memory = not args.no_memory

    print(
        f"Config: batch={args.batch_size}, emb_dim={args.emb_dim}, "
        f"heads={args.num_heads}, variant={flash_label}, "
        f"block_size={args.flash_block_size}"
    )
    print(
        f"Sweep:  seq_lengths={args.seq_lengths}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    print(f"Memory: {'yes' if measure_memory else 'disabled (--no-memory)'}\n")

    results = run_sweep(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        flash_block_size=args.flash_block_size,
        flash_module=flash_module,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        measure_memory=measure_memory,
    )

    print_table(results, flash_label, measure_memory)

    plot_results(
        results=results,
        output=args.output,
        device_str=str(device),
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        flash_block_size=args.flash_block_size,
        flash_label=flash_label,
        measure_memory=measure_memory,
    )


if __name__ == "__main__":
    main()
