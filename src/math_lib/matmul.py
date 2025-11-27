import torch


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul forbidding any broadcasting in batch dimensions."""
    if a.shape[:-2] != b.shape[:-2]:
        raise RuntimeError(
            f'Strict matmul disallows broadcasting: '
            f'A batch dims = {a.shape[:-2]}, B batch dims = {b.shape[:-2]}')
    return torch.matmul(a, b)

