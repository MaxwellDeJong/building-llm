"""Streaming dataset wrapper for nvidia/Nemotron-ClimbMix.

ClimbMix is a 400-billion-token pre-training corpus distributed by NVIDIA on
Hugging Face (https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix).
Each row contains a pre-tokenized document (GPT-2 vocabulary) stored in the
``tokens`` column.

This module exposes :class:`ClimbMixDataset`, a
:class:`torch.utils.data.IterableDataset` that:

1. Streams rows from HuggingFace without downloading the full dataset.
2. Concatenates token sequences from consecutive rows into a continuous buffer.
3. Yields ``(input_ids, target_ids)`` pairs of length ``context_length`` by
   sliding a non-overlapping window over the buffer.

Split strategy
--------------
The HuggingFace dataset only exposes a single ``"train"`` split.  We create
virtual splits by reserving the first ``n_val_examples`` rows for validation
and the next ``n_test_examples`` rows for test.  All remaining rows are used
for training.  Using ``.take()`` and ``.skip()`` on a streaming dataset is
O(n) but only iterates metadata-level shards, so it remains practical for
small ``n_val_examples`` / ``n_test_examples`` values (≤ 10 000).

Multi-worker DataLoader support
--------------------------------
When the DataLoader uses ``num_workers > 0``, each worker receives a replica
of this dataset.  We call ``.shard()`` on the underlying HuggingFace
IterableDataset so that worker *i* sees only 1/num_workers of the data,
preventing duplicate samples across workers.
"""
from __future__ import annotations

from typing import Iterator

import datasets
import torch
from torch.utils.data import IterableDataset

_DATASET_NAME = "nvidia/Nemotron-ClimbMix"

# ClimbMix ships pre-tokenized token IDs under the GPT-2 vocabulary
# (vocab_size=50257).  No tokenizer call is needed during training — the
# dataset yields integer token IDs directly.  The Tokenizer class in
# src/tokenization/tokenizer.py is used at inference time to encode prompts
# and decode generated outputs.


class ClimbMixDataset(IterableDataset):
    """Streaming (input_ids, target_ids) pairs from the ClimbMix corpus.

    Args:
        context_length: Number of tokens per training example.  Each yielded
            pair contains ``context_length`` tokens.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        n_val_examples: Number of dataset rows reserved for validation.
        n_test_examples: Number of dataset rows reserved for test.
    """

    def __init__(
            self,
            context_length: int,
            split: str = "train",
            n_val_examples: int = 5_000,
            n_test_examples: int = 5_000) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")
        self._context_length = context_length
        self._split = split
        self._n_val = n_val_examples
        self._n_test = n_test_examples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_stream(self) -> datasets.IterableDataset:
        """Return the raw HuggingFace streaming dataset for the requested split."""
        ds: datasets.IterableDataset = datasets.load_dataset(
            _DATASET_NAME,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        if self._split == "val":
            return ds.take(self._n_val)
        if self._split == "test":
            return ds.skip(self._n_val).take(self._n_test)
        # train: skip the reserved val + test rows
        return ds.skip(self._n_val + self._n_test)

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(input_ids, target_ids)`` pairs of length ``context_length``.

        Tokens from consecutive dataset rows are concatenated into a ring
        buffer.  A non-overlapping window of size ``context_length + 1`` is
        consumed from the front of the buffer; the first ``context_length``
        tokens form ``input_ids`` and the last ``context_length`` tokens
        (shifted by one) form ``target_ids``.
        """
        stream = self._base_stream()

        # Shard across DataLoader workers to avoid duplicate samples.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            stream = stream.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        chunk_size = self._context_length + 1
        buffer: list[int] = []

        for example in stream:
            buffer.extend(example["tokens"])
            while len(buffer) >= chunk_size:
                chunk = torch.tensor(buffer[:chunk_size], dtype=torch.long)
                buffer = buffer[chunk_size:]
                yield chunk[:-1], chunk[1:]
