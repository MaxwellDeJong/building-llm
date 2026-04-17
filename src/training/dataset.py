"""Dataset utilities for karpathy/climbmix-400b-shuffle."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import pathlib
from typing import Iterator

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from tokenization.tokenizer import Tokenizer


log = logging.getLogger(__name__)

DATASET_REPO = "karpathy/climbmix-400b-shuffle"
TEXT_COLUMN = "text"

# Conservative tokens-per-shard estimate used only for pre-download planning.
TOKENS_PER_SHARD_EST = 50_000_000


def estimate_shards_needed(
        n_params: int, chinchilla_mult: float = 20.0) -> int:
    """Return the number of shards for Chinchilla-optimal training."""
    chinchilla_tokens = chinchilla_mult * n_params
    return math.ceil(chinchilla_tokens / TOKENS_PER_SHARD_EST)

def download_shards(
        n_shards: int, raw_dir: pathlib.Path) -> list[pathlib.Path]:
    """Download missing parquet shards from DATASET_REPO."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []

    for i in range(n_shards):
        filename = f"shard_{i:05d}.parquet"
        local_path = os.path.join(raw_dir / filename

        if local_path.exists():
            log.info(
                "Shard %d/%d already cached: %s", i + 1, n_shards, filename)
        else:
            log.info(
                "Downloading shard %d/%d: %s …", i + 1, n_shards, filename)
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(raw_dir))
            local_path = pathlib.Path(downloaded)
        paths.append(local_path)
    return paths

def collect_training_text(
        raw_paths: list[pathlib.Path],
        shard_fraction: float = 0.01,
        max_chars: int = 200_000) -> str:
    """Sample the first shard to form a representative training corpus."""
    table = pq.read_table(raw_paths[0], columns=[TEXT_COLUMN])
    column = table[TEXT_COLUMN].to_pylist()
    n_sample = max(1, int(len(column) * shard_fraction))

    parts: list[str] = []
    total_chars = 0
    for text in column[:n_sample]:
        if not text:
            continue
        s = str(text)
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        parts.append(s[:remaining])
        total_chars += min(len(s), remaining)
    result = "\n".join(parts)
    log.info(
        "Tokenizer training corpus: %d chars from %d documents (shard 0).",
        len(result), len(parts))
    return result

def _tokenize_shard_worker(
        raw_path: pathlib.Path,
        tokenizer_path: pathlib.Path,
        token_dir: pathlib.Path,
        special_tokens: tuple[str, ...]) -> tuple[pathlib.Path, int, int]:
    """Tokenise one parquet shard and save the result as a `.npy` file."""
    tokenizer = Tokenizer.load(tokenizer_path)
    eos_ids = tokenizer.encode("<|endoftext|>", special_tokens)
    eos_token: int = eos_ids[0] if eos_ids else 0

    table = pq.read_table(raw_path, columns=[TEXT_COLUMN])
    texts = table[TEXT_COLUMN].to_pylist()

    all_tokens: list[int] = []
    for text in texts:
        if text:
            all_tokens.extend(tokenizer.encode(str(text), special_tokens))
            all_tokens.append(eos_token)

    cache_path = token_dir / f"{raw_path.stem}.npy"
    arr = np.array(all_tokens, dtype=np.uint32)
    np.save(cache_path, arr)
    return cache_path, len(arr), len(texts)

def tokenize_and_cache_shards(
        raw_paths: list[pathlib.Path],
        tokenizer_path: pathlib.Path,
        token_dir: pathlib.Path,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        num_workers: int | None = None) -> list[pathlib.Path]:
    """Tokenize missing parquet shards in parallel."""
    token_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output paths upfront to preserve ordering regardless of
    # completion order.
    cache_paths: list[pathlib.Path] = [
        token_dir / f"{p.stem}.npy" for p in raw_paths]

    # Separate already-cached shards from those that need work.
    pending: list[tuple[int, pathlib.Path]] = []
    for i, (raw_path, cache_path) in enumerate(zip(raw_paths, cache_paths)):
        if cache_path.exists():
            log.info("Token cache already exists: %s", cache_path.name)
        else:
            pending.append((i, raw_path))

    if not pending:
        return cache_paths

    workers = min(num_workers or os.cpu_count() // 2 or 1, len(pending))
    log.info(
        "Tokenising %d shard(s) across %d worker process(es) …",
        len(pending), workers)

    # Map each future back to its original index so we can log in completion
    # order while still returning results in shard order.
    futures: dict[concurrent.futures.Future, int] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i, raw_path in pending:
            fut = pool.submit(
                _tokenize_shard_worker,
                raw_path,
                tokenizer_path,
                token_dir,
                special_tokens)
            futures[fut] = i
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            cache_path, n_tokens, n_docs = fut.result()
            log.info(
                "[%d/%d] Cached %d tokens from %d docs → %s",
                completed, len(pending), n_tokens, n_docs, cache_path.name)
    return cache_paths


class TokenizedShardDataset(IterableDataset):
    """Yields (input_ids, target_ids) pairs from pre-tokenized files."""

    def __init__(
            self,
            shard_paths: list[pathlib.Path],
            context_length: int,
            split: str = "train",
            n_val_tokens: int = 5_120_000,
            n_test_tokens: int = 5_120_000) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test'; got {split!r}")
        self._shard_paths = shard_paths
        self._context_length = context_length
        self._split = split
        self._n_val = n_val_tokens
        self._n_test = n_test_tokens

    @staticmethod
    def _iter_array(arr: np.ndarray, chunk_size: int) -> (
            Iterator[tuple[torch.Tensor, torch.Tensor]]):
        """Yield non-overlapping (input_ids, target_ids) chunks from array."""
        n_chunks = len(arr) // chunk_size
        for i in range(n_chunks):
            # .astype() produces a contiguous int64 copy; torch.from_numpy()
            # then wraps it with zero additional allocation.
            chunk = torch.from_numpy(
                arr[i * chunk_size:(i + 1) * chunk_size].astype(np.int64))
            yield chunk[:-1], chunk[1:]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input_ids, target_ids) pairs respecting context length."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id: int = 0 if worker_info is None else worker_info.id
        num_workers: int = 1 if worker_info is None else worker_info.num_workers

        chunk_size = self._context_length + 1

        if self._split == "val":
            arr = np.load(self._shard_paths[0], mmap_mode="r")
            yield from self._iter_array(arr[:self._n_val], chunk_size)
            return
        if self._split == "test":
            arr = np.load(self._shard_paths[0], mmap_mode="r")
            yield from self._iter_array(
                arr[self._n_val:self._n_val + self._n_test], chunk_size)
            return
        # Training: each worker owns every num_workers-th shard.
        train_skip = self._n_val + self._n_test
        for shard_idx, path in enumerate(self._shard_paths):
            if shard_idx % num_workers != worker_id:
                continue
            arr = np.load(path, mmap_mode="r")
            start = train_skip if shard_idx == 0 else 0
            yield from self._iter_array(arr[start:], chunk_size)

