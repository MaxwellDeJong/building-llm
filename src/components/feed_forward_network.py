from typing import Any, Dict

import torch
from jaxtyping import Float

from components import gelu


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any], expansion_factor: int = 4) -> None:
        super().__init__()
        self._emb_dim = cfg['emb_dim']

        self._layers = torch.nn.Sequential(
            torch.nn.Linear(self._emb_dim, expansion_factor * self._emb_dim),
            gelu.GELU(),
            torch.nn.Linear(expansion_factor * self._emb_dim, self._emb_dim))

    def forward(self, x: Float[torch.Tensor, "*batch emb"]) -> (
            Float[torch.Tensor, "*batch emb"]):
        if x.shape[-1] != self._emb_dim:
            raise TypeError(
                f'Expected last dim {self._emb_dim}, got {x.shape[-1]}.')
        return self._layers(x)
