"""Feed-forward network module used inside transformer blocks."""
import dataclasses

import torch
from jaxtyping import Float

from components import gelu


@dataclasses.dataclass
class FeedForwardNetworkConfig:
    """Configuration for a two-layer feed-forward network."""
    expansion_factor: int = 4
    emb_dim: int | None = None


class FeedForwardNetwork(torch.nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, ffn_config: FeedForwardNetworkConfig) -> None:
        super().__init__()
        self._emb_dim = ffn_config.emb_dim
        self._layers = torch.nn.Sequential(
            torch.nn.Linear(self._emb_dim, ffn_config.expansion_factor * self._emb_dim),
            gelu.GELU(),
            torch.nn.Linear(ffn_config.expansion_factor * self._emb_dim, self._emb_dim))

    def forward(self, x: Float[torch.Tensor, "*batch emb"]) -> (
            Float[torch.Tensor, "*batch emb"]):
        """Apply the feed-forward network to x."""
        if x.shape[-1] != self._emb_dim:
            raise TypeError(
                f'Expected last dim {self._emb_dim}, got {x.shape[-1]}.')
        return self._layers(x)
