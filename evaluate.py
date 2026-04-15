"""Evaluate a trained GPT model by computing bits-per-byte on each data split.

Bits per byte (BPB) normalises the model's cross-entropy loss by the byte
length of the decoded text, making the metric independent of the tokenizer's
vocabulary size.  This lets you compare models trained with different
tokenizations on equal footing.

Formula
-------
    BPB = Σ_t CE(t) / (B · ln 2)

where Σ_t CE(t) is the sum of per-token cross-entropy losses (in nats) over
all evaluated tokens and B is the total number of UTF-8 bytes obtained by
decoding the target token sequences.

A lower BPB means the model assigns higher probability to the correct next
byte on average, so it is a better compressor of the data.

Usage
-----
    python evaluate.py --checkpoint checkpoints/ckpt_step_0001000.pt

    python evaluate.py \\
        --checkpoint checkpoints/ckpt_step_0001000.pt \\
        --splits val test \\
        --device cpu

    python evaluate.py \\
        --checkpoint checkpoints/ckpt_step_0001000.pt \\
        --train-batches 200

The script reads from the tokenised shard cache produced by training, so no
data is re-downloaded.  The tokenizer and shard files must exist under the
``data_dir`` specified in the training config.

Typical reference values (lower is better)
-------------------------------------------
* Character-level entropy of English text  ≈ 1.3 bpb
* GPT-2 (774 M) on WebText                ≈ 0.93 bpb
* A random model predicting uniformly      = log2(vocab_size) × tokens/byte
"""
from __future__ import annotations

import argparse
import logging
import math
import pathlib
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from tokenization.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>",)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Path to a specific checkpoint file produced by the Trainer.",
    )
    ckpt_group.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Automatically select the checkpoint with the highest step number "
            "from the checkpoint_dir specified in training_config.yaml."
        ),
    )
    p.add_argument(
        "--model-config",
        default="src/configs/gpt2_small.yaml",
        metavar="PATH",
        help=(
            "YAML file following the GPTModelConfig schema. "
            "(default: %(default)s)"
        ),
    )
    p.add_argument(
        "--training-config",
        default="src/configs/training_config.yaml",
        metavar="PATH",
        help=(
            "YAML file following the TrainingConfig schema. "
            "(default: %(default)s)"
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help=(
            "Which splits to evaluate (space-separated). "
            "Val and test are always evaluated in full; "
            "see --train-batches to limit the train split. "
            "(default: train val test)"
        ),
    )
    p.add_argument(
        "--train-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of batches drawn from the training split.  "
            "Evaluating the full training set can be very slow when many shards "
            "are present.  Val and test splits are always evaluated in full."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Evaluation batch size. Defaults to the batch_size in "
            "training_config.yaml."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "PyTorch device string (e.g. cuda, cpu, mps). "
            "Defaults to the device in training_config.yaml."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        autocast: torch.amp.autocast,
        max_batches: int | None = None,
) -> dict[str, float]:
    """Compute cross-entropy loss, perplexity, and bits-per-byte for one split.

    The function accumulates per-token NLL sums (in nats) and the UTF-8 byte
    lengths of the decoded target sequences.  Summing before dividing ensures
    that the final BPB is correctly weighted by sequence length even when the
    last batch is shorter than the others.

    Args:
        model: The GPT model to evaluate.
        loader: DataLoader yielding ``(input_ids, target_ids)`` tensor pairs.
        tokenizer: Trained BPE tokenizer used to decode token ids to text.
        device: Device used for the forward pass.
        autocast: ``torch.amp.autocast`` context manager (may be a no-op on CPU).
        max_batches: Stop after this many batches; ``None`` evaluates the full
            split.

    Returns:
        Dict with keys:

        * ``"loss"``       – average cross-entropy loss in nats per token.
        * ``"perplexity"`` – exp(loss).
        * ``"bpb"``        – bits per byte.
        * ``"n_tokens"``   – total number of tokens evaluated.
        * ``"n_bytes"``    – total number of UTF-8 bytes in the decoded text.
    """
    model.eval()

    total_nats: float = 0.0
    total_tokens: int = 0
    total_bytes: int = 0

    for n_batch, (inputs, targets) in enumerate(loader):
        if max_batches is not None and n_batch >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast:
            logits = model(inputs)
            # Use reduction="sum" so we can accumulate correctly across batches
            # and later divide by the total byte count for BPB.
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )

        total_nats += loss_sum.item()
        total_tokens += targets.numel()

        # Decode each sequence in the batch and count UTF-8 bytes.
        # targets holds the tokens the model predicts, so its byte length is
        # the denominator for BPB (i.e. "how many bits per byte of output?").
        for seq in targets.cpu().numpy():
            text = tokenizer.decode(seq.tolist())
            total_bytes += len(text.encode("utf-8"))

    if total_tokens == 0:
        log.warning("No tokens found — this split may be empty.")
        return {
            "loss": float("nan"),
            "perplexity": float("nan"),
            "bpb": float("nan"),
            "n_tokens": 0,
            "n_bytes": 0,
        }

    avg_loss = total_nats / total_tokens
    bpb = total_nats / (total_bytes * math.log(2))

    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "bpb": bpb,
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """Return the checkpoint file with the highest step number.

    Expects files named ``ckpt_step_XXXXXXX.pt`` as written by the Trainer.

    Args:
        checkpoint_dir: Directory to search for checkpoint files.

    Returns:
        Path to the checkpoint with the highest step number.

    Raises:
        SystemExit: If no checkpoint files are found.
    """
    candidates = sorted(checkpoint_dir.glob("ckpt_step_*.pt"))
    if not candidates:
        log.error(
            "No checkpoint files (ckpt_step_*.pt) found in %s.", checkpoint_dir)
        sys.exit(1)

    # Parse the step number from each filename and pick the highest.
    def _step(p: pathlib.Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    latest = max(candidates, key=_step)
    log.info(
        "Found %d checkpoint(s); selecting latest: %s",
        len(candidates), latest.name,
    )
    return latest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Deferred imports so the module-level import of torch is the only hard
    # dependency for --help to work even without PYTHONPATH set.
    from training.trainer import load_yaml_config, TrainingConfig
    from components.gpt_model import GPTModelConfig, GPTModel
    from components import hybrid_flash_multihead_attention
    from tokenization.tokenizer import Tokenizer
    from training.dataset import TokenizedShardDataset

    # ------------------------------------------------------------------
    # Configs
    # ------------------------------------------------------------------
    model_cfg: GPTModelConfig = load_yaml_config(args.model_config, GPTModelConfig)
    training_cfg: TrainingConfig = load_yaml_config(
        args.training_config, TrainingConfig)

    # Mirror the attention-impl swap performed by the Trainer so the loaded
    # weights map to the correct module architecture.
    if training_cfg.attention_impl == "flash":
        std = model_cfg.transformer_block_config.mha_config
        model_cfg.transformer_block_config.mha_config = (
            hybrid_flash_multihead_attention.FlashAttentionConfig(
                d_in=std.d_in,
                d_out=std.d_out,
                context_length=std.context_length,
                dropout=std.dropout,
                num_heads=std.num_heads,
                block_size=training_cfg.flash_block_size,
                qkv_bias=std.qkv_bias,
            )
        )

    # ------------------------------------------------------------------
    # Device + autocast
    # ------------------------------------------------------------------
    device = _resolve_device(args.device or training_cfg.device)
    log.info("Device: %s", device)

    autocast = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    model = GPTModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")

    if args.latest:
        ckpt_path = _find_latest_checkpoint(
            pathlib.Path(training_cfg.checkpoint_dir))
    else:
        ckpt_path = pathlib.Path(args.checkpoint)
        if not ckpt_path.exists():
            log.error("Checkpoint not found: %s", ckpt_path)
            sys.exit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", -1)
    log.info("Checkpoint loaded (training step %d)", step)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    data_dir = pathlib.Path(training_cfg.data_dir)
    tokenizer_path = data_dir / "tokenizer.pkl"
    if not tokenizer_path.exists():
        log.error(
            "Tokenizer not found at %s.  Run training first to build it.",
            tokenizer_path,
        )
        sys.exit(1)

    tokenizer = Tokenizer.load(tokenizer_path)
    log.info("Tokenizer loaded (vocab_size=%d)", tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Token shard cache
    # ------------------------------------------------------------------
    token_dir = data_dir / "tokens"
    shard_paths = sorted(token_dir.glob("shard_*.npy"))
    if not shard_paths:
        log.error(
            "No tokenised shard files (shard_*.npy) found in %s.  "
            "Run training first to build the cache.",
            token_dir,
        )
        sys.exit(1)
    log.info("Found %d tokenised shard file(s) in %s", len(shard_paths), token_dir)

    context_length = model_cfg.context_length
    batch_size = args.batch_size or training_cfg.batch_size

    # ------------------------------------------------------------------
    # Evaluate each split
    # ------------------------------------------------------------------
    results: dict[str, dict[str, float]] = {}

    for split in args.splits:
        if split == "train" and args.train_batches is not None:
            log.info(
                "Evaluating split '%s' (capped at %d batches) …",
                split, args.train_batches,
            )
        else:
            log.info("Evaluating split '%s' …", split)

        ds = TokenizedShardDataset(
            shard_paths=shard_paths,
            context_length=context_length,
            split=split,
            n_val_tokens=training_cfg.n_val_tokens,
            n_test_tokens=training_cfg.n_test_tokens,
        )
        # All splits use num_workers=0 to avoid duplicate evaluation samples.
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )

        max_batches = args.train_batches if split == "train" else None
        metrics = evaluate_bpb(
            model, loader, tokenizer, device, autocast, max_batches=max_batches)
        results[split] = metrics

        log.info(
            "%-5s | loss %6.4f | ppl %9.2f | bpb %6.4f"
            " | %s tokens | %s bytes",
            split.upper(),
            metrics["loss"],
            metrics["perplexity"],
            metrics["bpb"],
            f"{metrics['n_tokens']:,}",
            f"{metrics['n_bytes']:,}",
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col_w = 12
    header = (
        f"{'Split':<8}"
        f"{'Loss':>{col_w}}"
        f"{'Perplexity':>{col_w}}"
        f"{'BPB':>{col_w}}"
        f"{'Tokens':>{col_w}}"
        f"{'Bytes':>{col_w}}"
    )
    separator = "-" * len(header)

    print()
    print(f"Checkpoint: {ckpt_path}  (step {step})")
    print(separator)
    print(header)
    print(separator)
    for split, m in results.items():
        print(
            f"{split.upper():<8}"
            f"{m['loss']:>{col_w}.4f}"
            f"{m['perplexity']:>{col_w}.2f}"
            f"{m['bpb']:>{col_w}.4f}"
            f"{m['n_tokens']:>{col_w},}"
            f"{m['n_bytes']:>{col_w},}"
        )
    print(separator)
    print()


if __name__ == "__main__":
    main()
