"""GPT trainer using the Muon + AdamW optimizer pair on climbmix-400b-shuffle.

Typical usage
-------------
.. code-block:: python

    from training.trainer import Trainer

    trainer = Trainer(
        model_config_path="src/configs/gpt2_small.yaml",
        training_config_path="src/configs/training_config.yaml",
    )
    trainer.train()

Config files
------------
Both config files are YAML files that are deserialised into dataclasses via
``dacite``.  The model config follows the :class:`~components.gpt_model.GPTModelConfig`
schema (see ``src/configs/gpt2_small.yaml``).  The training config follows the
:class:`TrainingConfig` schema (see ``src/configs/training_config.yaml``).

Startup sequence
----------------
1. Parse configs.
2. Build the GPT model (with ``vocab_size`` taken from the model config).
3. Estimate the number of parquet shards required for Chinchilla-optimal
   pre-training and download them from ``karpathy/climbmix-400b-shuffle``.
4. Train a BPE tokenizer on a representative sample of the downloaded text
   (or load an existing tokenizer from ``<data_dir>/tokenizer.pkl``).  The
   tokenizer is trained to exactly ``vocab_size`` tokens as specified in the
   model config file.
5. Tokenise every downloaded shard and cache the token arrays as ``.npy``
   files in ``<data_dir>/tokens/``.
6. Build DataLoaders backed by :class:`~training.dataset.TokenizedShardDataset`.

Shard count estimation
-----------------------
The Chinchilla optimal token budget is ``20 × N`` (model parameters).  We
divide by :data:`~training.dataset.TOKENS_PER_SHARD_EST` (50 M tokens/shard,
conservative) to determine the number of shards to download before training
begins.  Actual coverage is verified after tokenisation via a logged summary.

Optimizer strategy
------------------
``torch.optim.Muon`` (available since PyTorch 2.10,
https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) is
applied to all 2-D weight matrices inside ``nn.Linear`` layers.  It handles
weight decay internally via a decoupled step applied *before* the
orthogonalized gradient update, and scales the effective learning rate by
``sqrt(max(1, rows/cols))`` so that the RMS of the update is consistent across
matrices of different aspect ratios.

Embedding weights and 1-D parameters (biases, layer-norm scale/shift) receive
AdamW updates.  Both optimizers share a cosine learning-rate schedule with
linear warmup.

Metrics reported
----------------
* **Training** – cross-entropy loss and perplexity every ``log_interval`` steps,
  plus throughput in tokens / second.
* **Validation** – cross-entropy loss and perplexity every ``eval_interval``
  steps, averaged over ``eval_iters`` batches.
* **Test** – cross-entropy loss and perplexity reported once at the end of
  training, averaged over ``eval_iters`` batches.
* **Final** – a three-row summary (train / val / test) after the last step.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
import time
from typing import Any

import dacite
import torch
import torch.nn.functional as F
import yaml

from components import hybrid_flash_multihead_attention
from components import gpt_model
from tokenization.tokenizer import Tokenizer
from training import dataset as dataset_module

log = logging.getLogger(__name__)

_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>",)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """Hyperparameters for training a GPT model on climbmix-400b-shuffle.

    All fields map directly to YAML keys in ``training_config.yaml``.
    No defaults are provided — every field must be explicitly set in the
    config file so that runs are fully reproducible from their YAML alone.

    Optimizer
    ~~~~~~~~~
    muon_lr:
        Peak learning rate for ``torch.optim.Muon`` (applied to 2-D weights).
    adamw_lr:
        Peak learning rate for AdamW (applied to embeddings, biases, norms).
    muon_momentum:
        Nesterov momentum coefficient for Muon.
    muon_weight_decay:
        Decoupled weight decay applied inside ``torch.optim.Muon`` before the
        orthogonalized gradient step.
    adamw_betas:
        (beta1, beta2) for AdamW.
    adamw_weight_decay:
        L2 regularisation coefficient for AdamW.
    gradient_clip:
        Maximum global gradient norm; 0 to disable.

    Schedule
    ~~~~~~~~
    warmup_steps:
        Number of steps over which the LR is linearly increased from 0.

    Data
    ~~~~
    data_dir:
        Root directory for downloaded shards and tokenised caches.
        Subdirectories ``raw/`` (parquet files) and ``tokens/`` (``.npy``
        arrays) are created automatically.
    tokenizer_train_fraction:
        Fraction of documents sampled from each downloaded shard to train
        the BPE tokenizer.  A small fraction (e.g. 0.01) is sufficient for
        BPE learning and keeps tokenizer training fast.
    batch_size:
        Number of sequences per micro-batch.
    gradient_accumulation_steps:
        Number of forward/backward passes before one optimizer step.  The
        effective batch size is ``batch_size * gradient_accumulation_steps``.
    n_val_tokens:
        Number of tokens (from the start of the first shard) reserved for
        the validation split.
    n_test_tokens:
        Number of tokens immediately following the validation window reserved
        for the test split.
    num_workers:
        DataLoader worker processes for the training split.

    Architecture overrides
    ~~~~~~~~~~~~~~~~~~~~~~
    attention_impl:
        Which multi-head attention implementation to use.  One of:

        * ``"standard"`` – ``torch.nn.functional.scaled_dot_product_attention``
          (fused CUDA kernel when available; materialises O(T²) attention
          scores in VRAM).
        * ``"flash"`` – block-tiled manual Flash Attention (O(T) VRAM).

    flash_block_size:
        Tile size for the manual Flash Attention inner loop.  Only used when
        ``attention_impl`` is ``"flash"``.

    Logging & checkpoints
    ~~~~~~~~~~~~~~~~~~~~~
    log_interval:
        Steps between training-loss log lines.
    eval_interval:
        Steps between validation evaluations.
    eval_iters:
        Number of batches averaged during each evaluation pass.
    checkpoint_dir:
        Directory to write ``ckpt_step_XXXXXXX.pt`` files.
    device:
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"mps"``).  Falls back
        to CPU when the requested device is unavailable.
    """

    # Optimizer
    muon_lr: float
    adamw_lr: float
    muon_momentum: float
    muon_weight_decay: float
    adamw_betas: list[float]
    adamw_weight_decay: float
    gradient_clip: float

    # Schedule
    warmup_steps: int

    # Data
    data_dir: str
    tokenizer_train_fraction: float
    batch_size: int
    gradient_accumulation_steps: int
    n_val_tokens: int
    n_test_tokens: int
    num_workers: int

    # Architecture overrides
    attention_impl: str
    flash_block_size: int

    # Logging & checkpoints
    log_interval: int
    eval_interval: int
    eval_iters: int
    checkpoint_dir: str
    device: str

    def __post_init__(self) -> None:
        allowed = {"standard", "flash"}
        if self.attention_impl not in allowed:
            raise ValueError(
                f"attention_impl must be one of {allowed!r}; "
                f"got {self.attention_impl!r}."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | pathlib.Path, cls: type) -> Any:
    """Deserialise a YAML file into a dataclass using ``dacite``.

    Args:
        path: Path to the YAML config file.
        cls:  Target dataclass type.

    Returns:
        An instance of ``cls`` populated from the YAML data.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return dacite.from_dict(data_class=cls, data=raw)


def _lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    """Cosine decay with linear warmup, decaying to 10 % of the peak LR."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trains a GPT model on climbmix-400b-shuffle using Muon + AdamW.

    On first run the trainer:

    1. Downloads the Chinchilla-optimal number of parquet shards.
    2. Trains a BPE tokenizer (vocab_size from the model config) on a
       representative sample of the downloaded text and saves it to
       ``<data_dir>/tokenizer.pkl``.
    3. Tokenises every shard and caches the ``uint32`` token arrays under
       ``<data_dir>/tokens/``.

    On subsequent runs, existing caches are detected and skipped so training
    can resume quickly.

    Args:
        model_config_path: Path to a YAML file following the
            :class:`~components.gpt_model.GPTModelConfig` schema.
        training_config_path: Path to a YAML file following the
            :class:`TrainingConfig` schema.
    """

    def __init__(
            self,
            model_config_path: str | pathlib.Path,
            training_config_path: str | pathlib.Path) -> None:
        self._cfg = load_yaml_config(training_config_path, TrainingConfig)
        model_cfg: gpt_model.GPTModelConfig = load_yaml_config(
            model_config_path, gpt_model.GPTModelConfig)

        if self._cfg.attention_impl == "flash":
            std = model_cfg.transformer_block_config.mha_config
            model_cfg.transformer_block_config.mha_config = (
                hybrid_flash_multihead_attention.FlashAttentionConfig(
                    d_in=std.d_in,
                    d_out=std.d_out,
                    context_length=std.context_length,
                    dropout=std.dropout,
                    num_heads=std.num_heads,
                    block_size=self._cfg.flash_block_size,
                    qkv_bias=std.qkv_bias,
                )
            )
            log.info(
                "Attention implementation: flash (block_size=%d)",
                self._cfg.flash_block_size,
            )
        else:
            log.info("Attention implementation: standard")

        self._device = self._resolve_device(self._cfg.device)
        log.info("Device: %s", self._device)

        # bf16 autocast: forward + backward run in bfloat16, master weights
        # stay fp32.  No GradScaler is needed — bf16 has sufficient exponent
        # range that gradient underflow is not a concern (unlike fp16).
        self._autocast = torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=(self._device.type == "cuda"),
        )
        log.info(
            "Mixed precision: %s",
            "bf16 autocast" if self._device.type == "cuda" else "disabled (CPU)",
        )

        # Build model.  vocab_size is taken directly from the model config and
        # must match the tokenizer that will be trained in the next step.
        self._model = gpt_model.GPTModel(model_cfg).to(self._device)
        param_count = sum(p.numel() for p in self._model.parameters())
        log.info("Model parameters: %s", f"{param_count:,}")

        self._context_length = model_cfg.context_length

        # ------------------------------------------------------------------
        # Data preparation
        # ------------------------------------------------------------------
        cfg = self._cfg
        data_dir = pathlib.Path(cfg.data_dir)
        raw_dir = data_dir / "raw"
        token_dir = data_dir / "tokens"

        # Estimate shards needed for Chinchilla-optimal training.
        n_shards = dataset_module.estimate_shards_needed(param_count)
        log.info(
            "Chinchilla-optimal budget: %s tokens → %d shards needed"
            " (%s tok/shard estimate)",
            f"{20 * param_count:,.0f}",
            n_shards,
            f"{dataset_module.TOKENS_PER_SHARD_EST:,}",
        )

        # Download shards (cached runs are skipped automatically).
        raw_shard_paths = dataset_module.download_shards(n_shards, raw_dir)

        # ------------------------------------------------------------------
        # Tokenizer — train or load from cache
        # ------------------------------------------------------------------
        tokenizer_path = data_dir / "tokenizer.pkl"
        target_vocab_size = model_cfg.vocab_size

        tokenizer: Tokenizer | None = None
        if tokenizer_path.exists():
            tokenizer = Tokenizer.load(tokenizer_path)
            if tokenizer.vocab_size != target_vocab_size:
                log.warning(
                    "Cached tokenizer has vocab_size=%d but model config "
                    "expects %d — re-training.",
                    tokenizer.vocab_size, target_vocab_size,
                )
                tokenizer = None

        if tokenizer is None:
            log.info(
                "Training BPE tokenizer (vocab_size=%d, fraction=%.3f) …",
                target_vocab_size, cfg.tokenizer_train_fraction,
            )
            training_text = dataset_module.collect_training_text(
                raw_shard_paths, fraction=cfg.tokenizer_train_fraction)
            tokenizer = Tokenizer()
            tokenizer.train(
                training_text,
                vocab_size=target_vocab_size,
                special_token_texts=_SPECIAL_TOKENS,
            )
            tokenizer.save(tokenizer_path)
            log.info("Tokenizer saved → %s", tokenizer_path)

        self._tokenizer = tokenizer

        # ------------------------------------------------------------------
        # Tokenise shards and cache to disk
        # ------------------------------------------------------------------
        self._token_shard_paths = dataset_module.tokenize_and_cache_shards(
            raw_shard_paths,
            tokenizer_path=tokenizer_path,
            token_dir=token_dir,
            special_tokens=_SPECIAL_TOKENS,
            num_workers=cfg.num_workers,
        )

        # ------------------------------------------------------------------
        # Training schedule
        # ------------------------------------------------------------------
        tokens_per_step = (
            cfg.batch_size * cfg.gradient_accumulation_steps * self._context_length)
        self._max_steps = math.ceil(20 * param_count / tokens_per_step)
        log.info(
            "Chinchilla-optimal steps: %d  (20 × %s params / %d tok/step)",
            self._max_steps, f"{param_count:,}", tokens_per_step,
        )

        self._optimizers, self._schedulers = self._build_optimizers()
        self._loaders = self._build_dataloaders()

        self._ckpt_dir = pathlib.Path(self._cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        if requested == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _build_optimizers(
            self,
    ) -> tuple[list[torch.optim.Optimizer],
               list[torch.optim.lr_scheduler.LRScheduler]]:
        """Partition parameters and create Muon + AdamW optimizer pair."""
        cfg = self._cfg

        # Collect embedding parameter ids so we can route them to AdamW.
        embedding_ids: set[int] = set()
        for module in self._model.modules():
            if isinstance(module, torch.nn.Embedding):
                for param in module.parameters(recurse=False):
                    embedding_ids.add(id(param))

        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        for param in self._model.parameters():
            if param.requires_grad:
                if param.ndim == 2 and id(param) not in embedding_ids:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        log.info(
            "Optimizer split — Muon params: %d groups, AdamW params: %d groups",
            len(muon_params), len(adamw_params))

        # torch.optim.Muon is available since PyTorch 2.10. It handles
        # weight decay internally (decoupled, applied before the orthogonalized
        # update) and adjusts the effective LR by sqrt(max(1, rows/cols)) so
        # that the update RMS is consistent across matrices of all shapes.
        muon_opt = torch.optim.Muon(
            muon_params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.muon_weight_decay,
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=cfg.adamw_lr,
            betas=tuple(cfg.adamw_betas),
            weight_decay=cfg.adamw_weight_decay,
        )

        lr_lambda = lambda step: _lr_lambda(  # noqa: E731
            step, cfg.warmup_steps, self._max_steps)
        muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
        adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)

        return [muon_opt, adamw_opt], [muon_sched, adamw_sched]

    def _build_dataloaders(
            self,
    ) -> dict[str, torch.utils.data.DataLoader]:
        cfg = self._cfg

        def _make(split: str) -> torch.utils.data.DataLoader:
            ds = dataset_module.TokenizedShardDataset(
                shard_paths=self._token_shard_paths,
                context_length=self._context_length,
                split=split,
                n_val_tokens=cfg.n_val_tokens,
                n_test_tokens=cfg.n_test_tokens,
            )
            # Val / test splits must use num_workers=0 to avoid duplicating
            # evaluation batches across workers.
            workers = cfg.num_workers if split == "train" else 0
            # pin_memory speeds up host→device DMA, but only benefits async
            # prefetch workers.  With num_workers=0 the main thread would pin
            # each batch synchronously (pure overhead), so skip it.
            return torch.utils.data.DataLoader(
                ds,
                batch_size=cfg.batch_size,
                num_workers=workers,
                pin_memory=(self._device.type == "cuda" and workers > 0),
                persistent_workers=(workers > 0),
                prefetch_factor=(4 if workers > 0 else None),
            )

        return {"train": _make("train"), "val": _make("val"), "test": _make("test")}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
            self,
            split: str,
            n_iters: int,
    ) -> dict[str, float]:
        """Compute average cross-entropy loss and perplexity for ``split``.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            n_iters: Maximum number of batches to evaluate.

        Returns:
            Dict with keys ``"loss"`` and ``"perplexity"``.
        """
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        for inputs, targets in self._loaders[split]:
            if n_batches >= n_iters:
                break
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            with self._autocast:
                logits = self._model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n_batches += 1
        self._model.train()
        avg_loss = total_loss / max(1, n_batches)
        return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop and report metrics throughout."""
        cfg = self._cfg
        self._model.train()
        for opt in self._optimizers:
            opt.zero_grad()

        step = 0
        accum_loss = torch.tensor(0.0, device=self._device)
        accum_steps = 0
        t0 = time.perf_counter()
        train_iter = iter(self._loaders["train"])

        log.info(
            "Starting training | max_steps=%d | effective_batch=%d tokens",
            self._max_steps,
            cfg.batch_size * cfg.gradient_accumulation_steps * self._context_length,
        )

        while step < self._max_steps:
            # ---- Gradient accumulation --------------------------------
            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    # Restart the training iterator when all shards are
                    # exhausted (unlikely with Chinchilla-sized data).
                    train_iter = iter(self._loaders["train"])
                    inputs, targets = next(train_iter)

                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                with self._autocast:
                    logits = self._model(inputs)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1))
                (loss / cfg.gradient_accumulation_steps).backward()
                accum_loss += loss.detach()
                accum_steps += 1

            # ---- Gradient clipping ------------------------------------
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), cfg.gradient_clip)

            # ---- Optimizer + scheduler step ---------------------------
            for opt in self._optimizers:
                opt.step()
            for sched in self._schedulers:
                sched.step()
            for opt in self._optimizers:
                opt.zero_grad()

            step += 1

            # ---- Training log -----------------------------------------
            if step % cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = (accum_loss / accum_steps).item()
                tokens_per_sec = (
                    cfg.log_interval
                    * cfg.gradient_accumulation_steps
                    * cfg.batch_size
                    * self._context_length
                    / elapsed
                )
                muon_lr = self._schedulers[0].get_last_lr()[0]
                adamw_lr = self._schedulers[1].get_last_lr()[0]
                log.info(
                    "step %7d | loss %6.4f | ppl %8.2f | %8.0f tok/s"
                    " | muon_lr %.2e | adamw_lr %.2e",
                    step, avg_loss, math.exp(avg_loss),
                    tokens_per_sec, muon_lr, adamw_lr,
                )
                accum_loss = torch.tensor(0.0, device=self._device)
                accum_steps = 0
                t0 = time.perf_counter()

            # ---- Validation + checkpoint ------------------------------
            if step % cfg.eval_interval == 0:
                val = self._evaluate("val", cfg.eval_iters)
                log.info(
                    "step %7d | VAL  loss %6.4f | ppl %8.2f",
                    step, val["loss"], val["perplexity"],
                )
                self._save_checkpoint(step)

        # ---- Final evaluation on all three splits ---------------------
        log.info("Training complete.  Running final evaluation…")
        for split_name in ("train", "val", "test"):
            metrics = self._evaluate(split_name, cfg.eval_iters)
            log.info(
                "Final %-5s | loss %6.4f | perplexity %8.2f",
                split_name.upper(), metrics["loss"], metrics["perplexity"],
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        path = self._ckpt_dir / f"ckpt_step_{step:07d}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self._model.state_dict(),
                "muon_state_dict": self._optimizers[0].state_dict(),
                "adamw_state_dict": self._optimizers[1].state_dict(),
                "muon_sched_state_dict": self._schedulers[0].state_dict(),
                "adamw_sched_state_dict": self._schedulers[1].state_dict(),
            },
            path,
        )
        log.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str | pathlib.Path) -> int:
        """Restore model + optimizer state from a checkpoint file.

        Args:
            path: Path to a checkpoint written by :meth:`_save_checkpoint`.

        Returns:
            The training step at which the checkpoint was saved.
        """
        ckpt = torch.load(path, map_location=self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._optimizers[0].load_state_dict(ckpt["muon_state_dict"])
        self._optimizers[1].load_state_dict(ckpt["adamw_state_dict"])
        self._schedulers[0].load_state_dict(ckpt["muon_sched_state_dict"])
        self._schedulers[1].load_state_dict(ckpt["adamw_sched_state_dict"])
        log.info("Checkpoint loaded from %s (step %d)", path, ckpt["step"])
        return ckpt["step"]
