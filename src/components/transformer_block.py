"""Transformer block combining MHA with MLP."""
import dataclasses

from jaxtyping import Float
import torch

from components import feed_forward_network
from components import flash_multihead_attention
from components import layer_norm
from components import multihead_attention

AttentionConfig = (
    multihead_attention.MultiHeadAttentionConfig
    | flash_multihead_attention.FlashAttentionConfig
)


@dataclasses.dataclass
class TransformerBlockConfig:
    """Configuration for a single transformer block."""
    mha_config: AttentionConfig
    ffn_config: feed_forward_network.FeedForwardNetworkConfig

    def __post_init__(self):
        if (self.ffn_config.emb_dim is not None
                and self.mha_config.d_out != self.ffn_config.emb_dim):
            raise ValueError('MHA and FFN must have the same output dimension.')

    @property
    def emb_dim(self) -> int | None:
        """Get embedding dimension."""
        return self.ffn_config.emb_dim


class TransformerBlock(torch.nn.Module):
    """Single transformer block combining attention with FFN with residuals."""

    def __init__(self, transformer_config: TransformerBlockConfig) -> None:
        super().__init__()
        self._emb_dim = transformer_config.emb_dim
        if isinstance(transformer_config.mha_config,
                      flash_multihead_attention.FlashAttentionConfig):
            self._mha = flash_multihead_attention.MultiHeadAttention(
                transformer_config.mha_config)
        else:
            self._mha = multihead_attention.MultiHeadAttention(
                transformer_config.mha_config)
        self._ffn = feed_forward_network.FeedForwardNetwork(
            transformer_config.ffn_config)
        self._mha_norm = layer_norm.LayerNorm(
            layer_norm.LayerNormConfig(emb_dim=self._emb_dim))
        self._ffn_norm = layer_norm.LayerNorm(
            layer_norm.LayerNormConfig(emb_dim=self._emb_dim))
        self._drop_out = torch.nn.Dropout(
            transformer_config.mha_config.dropout)

    def forward(self, x: Float[torch.Tensor, 'batch seq emb']) -> (
            Float[torch.Tensor, 'batch seq emb']):
        """Apply one transformer block to the input tensor."""
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
