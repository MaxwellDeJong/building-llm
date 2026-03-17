"""Multi-head causal self-attention module."""
import dataclasses

from einops import rearrange
from jaxtyping import Bool, Float
import torch

from math_lib import matmul


@dataclasses.dataclass
class MultiHeadAttentionConfig:
    """Configuration for multi-head causal self-attention."""
    d_in: int
    d_out: int
    context_length: int
    dropout: float
    num_heads: int
    qkv_bias: bool = False

    def __post_init__(self):
        if self.d_out % self.num_heads != 0:
            raise ValueError('d_out must be divisible by num_heads.')


class MultiHeadAttention(torch.nn.Module):
    """Multi-head causal self-attention with optional KV caching."""

    def __init__(self, mha_config: MultiHeadAttentionConfig) -> None:
        super().__init__()
        self._d_in = mha_config.d_in
        self._d_out = mha_config.d_out
        self._num_heads = mha_config.num_heads
        # Reduce the projection dim to match desired output dim.
        self._head_dim = mha_config.d_out // mha_config.num_heads

        self._W_query = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        self._W_key = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        self._W_value = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        # Linear layer to combine head outputs.
        self._out_proj = torch.nn.Linear(mha_config.d_out, mha_config.d_out)
        self._dropout = torch.nn.Dropout(mha_config.dropout)
        self.register_buffer(
            '_mask',
            torch.triu(
              torch.ones(mha_config.context_length, mha_config.context_length),
              diagonal=1),
            persistent=False)

        # Initialize KV cache variables.
        self.register_buffer('_cache_k', None, persistent=False)
        self.register_buffer('_cache_v', None, persistent=False)
        self._token_position_ptr = 0

    def forward(self, x: Float[torch.Tensor, 'b t din']) -> (
            Float[torch.Tensor, 'b t dout']):
        """Compute multi-head causal self-attention over x."""
        num_tokens = x.shape[1]

        keys: Float[torch.Tensor,'b t dout'] = self._W_key(x)
        queries: Float[torch.Tensor, 'b t dout'] = self._W_query(x)
        values: Float[torch.Tensor, 'b t dout'] = self._W_value(x)

        keysT = rearrange(
            keys,
            'b t (nh hd) -> b nh hd t',
            nh=self._num_heads,
            hd=self._head_dim)
        values = rearrange(
            values,
            'b t (nh hd) -> b nh t hd',
            nh=self._num_heads,
            hd=self._head_dim)
        queries = rearrange(
            queries,
            'b t (nh hd) -> b nh t hd',
            nh=self._num_heads,
            hd=self._head_dim)

        # Compute scaled dot-product attention (aka self-attention) with a
        # causal mask.
        attn_scores: Float[torch.Tensor, 'b nh t t'] = matmul.matmul(
            queries, keysT)

        # Original mask truncated to the number of tokens and converted to
        # boolean.
        mask_bool: Bool[torch.Tensor, 't t'] = self._mask.bool()[
            :num_tokens, :num_tokens]

        # Use the mask to fill attention scores.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self._head_dim**0.5, dim=-1)
        attn_weights = self._dropout(attn_weights)

        context_vec: Float[torch.Tensor, 'b nh t hd'] = matmul.matmul(
            attn_weights, values)
        context_vec = rearrange(context_vec, 'b nh t hd -> b t (nh hd)')
        context_vec = self._out_proj(context_vec) # optional projection

        return context_vec
