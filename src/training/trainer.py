"""GPT trainer using the Muon + AdamW optimizer pair on the ClimbMix corpus.

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

from components import gpt_model
from training import dataset as dataset_module

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """Hyperparameters for training a GPT model on ClimbMix.

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
    batch_size:
        Number of sequences per micro-batch.
    gradient_accumulation_steps:
        Number of forward/backward passes before one optimizer step.  The
        effective batch size is ``batch_size * gradient_accumulation_steps``.
    n_val_examples:
        Number of ClimbMix rows reserved as the validation split.
    n_test_examples:
        Number of ClimbMix rows reserved as the test split.
    num_workers:
        DataLoader worker processes per split.

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
    batch_size: int
    gradient_accumulation_steps: int
    n_val_examples: int
    n_test_examples: int
    num_workers: int

    # Logging & checkpoints
    log_interval: int
    eval_interval: int
    eval_iters: int
    checkpoint_dir: str
    device: str


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
    """Trains a GPT model on ClimbMix using Muon + AdamW.

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

        self._device = self._resolve_device(self._cfg.device)
        log.info("Device: %s", self._device)

        self._model = gpt_model.GPTModel(model_cfg).to(self._device)
        param_count = sum(p.numel() for p in self._model.parameters())
        log.info("Model parameters: %s", f"{param_count:,}")

        self._context_length = model_cfg.context_length

        # Chinchilla scaling law: train on ~20 tokens per parameter.
        # max_steps = ceil(20N / tokens_per_step)
        cfg = self._cfg
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
            ds = dataset_module.ClimbMixDataset(
                context_length=self._context_length,
                split=split,
                n_val_examples=cfg.n_val_examples,
                n_test_examples=cfg.n_test_examples,
            )
            return torch.utils.data.DataLoader(
                ds,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=(self._device.type == "cuda"),
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
        accum_loss = 0.0
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
                    # Restart the training iterator when the stream is
                    # exhausted (unlikely with 400 B tokens, but safe).
                    train_iter = iter(self._loaders["train"])
                    inputs, targets = next(train_iter)

                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                logits = self._model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
                (loss / cfg.gradient_accumulation_steps).backward()
                accum_loss += loss.item()
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
                avg_loss = accum_loss / accum_steps
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
                accum_loss = 0.0
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
