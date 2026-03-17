"""Layer normalisation module."""
import dataclasses

from jaxtyping import Float
import torch


@dataclasses.dataclass
class LayerNormConfig:
    """Configuration for layer normalisation."""
    emb_dim: int
    unbiased: bool = False


class LayerNorm(torch.nn.Module):
    """Layer normalisation over the last (embedding) dimension."""

    def __init__(self, ln_config: LayerNormConfig) -> None:
        super().__init__()
        self._emb_dim = ln_config.emb_dim
        self._eps = 1e-5
        self._unbiased = ln_config.unbiased
        self._scale = torch.nn.Parameter(torch.ones(ln_config.emb_dim))
        self._shift = torch.nn.Parameter(torch.zeros(ln_config.emb_dim))

    def forward(
            self,
            x: Float[torch.Tensor, '*batch emb']) -> (
                Float[torch.Tensor, '*batch emb']):
        """Normalise x and apply learned scale and shift."""
        if x.shape[-1] != self._emb_dim:
            raise TypeError(
                f'Expected last dim == {self._emb_dim}, got {x.shape[-1]}.')
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=self._unbiased)
        norm_x = (x - mean) / torch.sqrt(var + self._eps)
        return self._scale * norm_x + self._shift
