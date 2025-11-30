from typing import Any, Dict

from jaxtyping import Float
import torch

from components import feed_forward_network
from components import layer_norm
from components import multihead_attention


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self._emb_dim: int = cfg['emb_dim']
        self._mha = multihead_attention.MultiHeadAttention(
            d_in=self._emb_dim,
            d_out=self._emb_dim,
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_out'],
            qkv_bias=cfg['qkv_bias'])
        self._ffn = feed_forward_network.FeedForwardNetwork(cfg)
        self._mha_norm = layer_norm.LayerNorm(self._emb_dim)
        self._ffn_norm = layer_norm.LayerNorm(self._emb_dim)
        self._drop_out = torch.nn.Dropout(cfg['drop_out'])

    def forward(self, x: Float[torch.Tensor, 'batch seq emb']) -> (
            Float[torch.Tensor, 'batch seq emb']):
        if x.shape[-1] != self._emb_dim:
            raise TypeError(
                f'Expected emb_dim={self._emb_dim}, got {x.shape[-1]}.')
        res_input = x
        # Attention block.
        x = self._mha_norm(x)
        x = self._mha(x)
        x = self._drop_out(x)
        x += res_input
        # FFN block.
        res_input = x
        x = self._ffn_norm(x)
        x = self._ffn(x)
        x = self._drop_out(x)
        x += res_input
        return x

