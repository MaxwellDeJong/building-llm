"""Dataset utilities for karpathy/climbmix-400b-shuffle with custom BPE tokenization.

Workflow
--------
1. :func:`estimate_shards_needed` determines how many parquet shards are required
   for Chinchilla-optimal pre-training given the model's parameter count.
2. :func:`download_shards` fetches those shards from the Hugging Face Hub and
   caches them locally (already-downloaded shards are skipped).
3. :func:`collect_training_text` samples a representative fraction of the raw
   text for BPE tokenizer training.
4. :func:`tokenize_and_cache_shards` runs the trained :class:`~tokenization.tokenizer.Tokenizer`
   over every document in each shard and stores the resulting token ids as a
   ``uint32`` ``.npy`` file (re-tokenization is skipped when the cache exists).
5. :class:`TokenizedShardDataset` reads the cached ``.npy`` files (memory-mapped)
   and yields ``(input_ids, target_ids)`` pairs of length ``context_length``.

Split strategy
--------------
The first ``n_val_tokens`` tokens in the concatenated stream (drawn from the
beginning of shard 0) are reserved for validation.  The next ``n_test_tokens``
tokens are reserved for test.  Everything beyond that offset is used for
training.

Multi-worker DataLoader support
--------------------------------
For the training split, shard files are distributed across DataLoader workers
using round-robin assignment (shard index modulo number of workers), so no
document is seen by more than one worker.  Validation and test splits should be
loaded with ``num_workers=0`` to avoid duplicating evaluation samples.

Shard count estimation
----------------------
Each compressed parquet shard is ~92 MB.  The full dataset contains ~400 B
tokens across ~6 500 shards (600 GB total), giving roughly 61 M tokens per
shard with the dataset's native tokenizer.  We use a conservative estimate of
``TOKENS_PER_SHARD_EST = 50_000_000`` so that slightly fewer tokens per shard
(due to a custom vocabulary) still yields sufficient training data.
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import pathlib
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from tokenization.tokenizer import Tokenizer

log = logging.getLogger(__name__)

DATASET_REPO = "karpathy/climbmix-400b-shuffle"
TEXT_COLUMN = "text"

# Conservative tokens-per-shard estimate used only for pre-download planning.
# Actual token counts are determined after tokenization and cached to disk.
TOKENS_PER_SHARD_EST = 50_000_000


# ---------------------------------------------------------------------------
# Shard count estimation
# ---------------------------------------------------------------------------

def estimate_shards_needed(
        n_params: int,
        chinchilla_mult: float = 20.0,
) -> int:
    """Return the minimum number of shards for Chinchilla-optimal training.

    Uses the Chinchilla scaling law ``tokens = chinchilla_mult × N`` and a
    conservative estimate of ``TOKENS_PER_SHARD_EST`` tokens per shard.

    Args:
        n_params: Total trainable parameter count of the model.
        chinchilla_mult: Tokens-per-parameter multiplier (default 20).

    Returns:
        Number of shards to download.
    """
    chinchilla_tokens = chinchilla_mult * n_params
    return math.ceil(chinchilla_tokens / TOKENS_PER_SHARD_EST)


# ---------------------------------------------------------------------------
# Shard download
# ---------------------------------------------------------------------------

def download_shards(
        n_shards: int,
        raw_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Download parquet shards from ``karpathy/climbmix-400b-shuffle``.

    Shards that already exist in *raw_dir* are not re-downloaded.

    Args:
        n_shards: Number of shards to download (``shard_00000`` … ``shard_{n-1}``).
        raw_dir: Directory where parquet files are cached.

    Returns:
        Ordered list of paths to local parquet files.
    """
    from huggingface_hub import hf_hub_download  # deferred import

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []

    for i in range(n_shards):
        filename = f"shard_{i:05d}.parquet"
        local_path = raw_dir / filename

        if local_path.exists():
            log.info("Shard %d/%d already cached: %s", i + 1, n_shards, filename)
        else:
            log.info("Downloading shard %d/%d: %s …", i + 1, n_shards, filename)
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            local_path = pathlib.Path(downloaded)

        paths.append(local_path)

    return paths


# ---------------------------------------------------------------------------
# Text collection for tokenizer training
# ---------------------------------------------------------------------------

def collect_training_text(
        raw_paths: list[pathlib.Path],
        fraction: float = 0.01,
        max_chars: int = 200_000,
) -> str:
    """Sample a representative text corpus for BPE tokenizer training.

    The custom BPE implementation runs one O(n) pass per merge step, so its
    total cost is O(vocab_size × n_tokens).  For vocab_size=50 257 this means
    ~50 000 passes; keeping *n_tokens* small (≲ 100 K) is essential to finish
    in a few minutes.  ``max_chars`` (default 500 KB) enforces this ceiling
    regardless of *fraction* or the number of shards passed.

    Only the **first shard** is used for tokenizer training regardless of how
    many shards are in *raw_paths*.  BPE quality saturates quickly — a few
    hundred thousand characters of diverse text is sufficient to learn good
    subword merges for a general-purpose corpus.

    Args:
        raw_paths: Raw parquet shard files.  Only ``raw_paths[0]`` is read.
        fraction: Fraction of documents sampled from the first shard.
        max_chars: Hard upper bound on the total character count collected.

    Returns:
        Newline-joined concatenation of sampled documents, truncated to
        ``max_chars`` characters.
    """
    import pyarrow.parquet as pq  # deferred import

    # Only sample from the first shard to keep the corpus small.
    table = pq.read_table(raw_paths[0], columns=[TEXT_COLUMN])
    column = table[TEXT_COLUMN].to_pylist()
    n_sample = max(1, int(len(column) * fraction))

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
        len(result), len(parts),
    )
    return result


# ---------------------------------------------------------------------------
# Tokenisation + caching
# ---------------------------------------------------------------------------

def _tokenize_shard_worker(
        raw_path: pathlib.Path,
        tokenizer_path: pathlib.Path,
        token_dir: pathlib.Path,
        special_tokens: tuple[str, ...],
) -> tuple[pathlib.Path, int, int]:
    """Tokenise one parquet shard and save the result as a ``.npy`` file.

    This is a module-level function so that ``ProcessPoolExecutor`` can
    pickle it by reference.  Each worker process loads the tokenizer from
    disk independently, avoiding the cost of pickling the full vocabulary
    across the process boundary for every submitted task.

    Args:
        raw_path: Parquet file to tokenise.
        tokenizer_path: Path to the saved tokenizer pickle.
        token_dir: Directory to write the ``.npy`` file.
        special_tokens: Special token strings kept intact during encoding.

    Returns:
        ``(cache_path, n_tokens, n_docs)`` tuple for logging.
    """
    import pyarrow.parquet as pq  # deferred — only imported in worker processes
    from tokenization.tokenizer import Tokenizer  # noqa: PLC0415

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
        num_workers: int | None = None,
) -> list[pathlib.Path]:
    """Tokenise parquet shards in parallel and cache each as a ``.npy`` file.

    Shards whose cache file already exists are skipped immediately.  The
    remaining shards are distributed across ``num_workers`` subprocesses
    (default: ``os.cpu_count()``).  Each worker loads the tokenizer from
    *tokenizer_path* independently so the full vocabulary is not pickled
    across the process boundary on every task submission.

    Args:
        raw_paths: Parquet files to tokenise.
        tokenizer_path: Path to the saved tokenizer pickle (produced by
            :meth:`~tokenization.tokenizer.Tokenizer.save`).
        token_dir: Directory to write ``shard_XXXXX.npy`` files.
        special_tokens: Special token strings kept intact during encoding.
        num_workers: Subprocess count.  Defaults to ``os.cpu_count()``.

    Returns:
        Ordered list of paths to ``.npy`` token cache files (``uint32``),
        in the same order as *raw_paths*.
    """
    token_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output paths upfront to preserve ordering regardless of
    # completion order.
    cache_paths: list[pathlib.Path] = [
        token_dir / f"{p.stem}.npy" for p in raw_paths
    ]

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
        len(pending), workers,
    )

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
                special_tokens,
            )
            futures[fut] = i

        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            cache_path, n_tokens, n_docs = fut.result()
            log.info(
                "[%d/%d] Cached %d tokens from %d docs → %s",
                completed, len(pending), n_tokens, n_docs, cache_path.name,
            )

    return cache_paths


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

class TokenizedShardDataset(IterableDataset):
    """Yields ``(input_ids, target_ids)`` pairs from pre-tokenised ``.npy`` files.

    The first ``n_val_tokens`` tokens in the concatenated stream (from the
    beginning of shard 0) are reserved for validation, and the next
    ``n_test_tokens`` are reserved for test.  The remainder is used for
    training.

    For the training split, ``.npy`` shard files are distributed across
    DataLoader workers by round-robin on the shard index, ensuring each
    document is processed by exactly one worker.  Val / test splits should
    use ``num_workers=0``.

    Args:
        shard_paths: Ordered list of ``.npy`` files (``uint32`` token arrays).
        context_length: Sequence length for each training example.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        n_val_tokens: Tokens reserved for the validation split.
        n_test_tokens: Tokens reserved for the test split.
    """

    def __init__(
            self,
            shard_paths: list[pathlib.Path],
            context_length: int,
            split: str = "train",
            n_val_tokens: int = 5_120_000,
            n_test_tokens: int = 5_120_000,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test'; got {split!r}")
        self._shard_paths = shard_paths
        self._context_length = context_length
        self._split = split
        self._n_val = n_val_tokens
        self._n_test = n_test_tokens

    @staticmethod
    def _iter_array(
            arr: np.ndarray,
            chunk_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield non-overlapping ``(input_ids, target_ids)`` chunks from *arr*.

        Slices *arr* directly (zero-copy view on mmap arrays) and converts each
        ``uint32`` chunk to ``int64`` only when constructing the output tensors,
        keeping both CPU RAM and GIL pressure minimal inside DataLoader workers.
        Incomplete trailing tokens (fewer than ``chunk_size``) are silently
        dropped, matching the behaviour of the previous buffer implementation.
        """
        n_chunks = len(arr) // chunk_size
        for i in range(n_chunks):
            # .astype() produces a contiguous int64 copy; torch.from_numpy()
            # then wraps it with zero additional allocation.
            chunk = torch.from_numpy(
                arr[i * chunk_size:(i + 1) * chunk_size].astype(np.int64))
            yield chunk[:-1], chunk[1:]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(input_ids, target_ids)`` pairs of length ``context_length``."""
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
