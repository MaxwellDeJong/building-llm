"""Matrix multiplication utilities with strict broadcasting semantics."""
import torch


def matmul(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Matmul forbidding any broadcasting in batch dimensions."""
    if t1.shape[:-2] != t2.shape[:-2]:
        raise RuntimeError(
            f'Strict matmul disallows broadcasting: '
            f't1 batch dims = {t1.shape[:-2]}, '
            f't2 batch dims = {t2.shape[:-2]}')
    return torch.matmul(t1, t2)
