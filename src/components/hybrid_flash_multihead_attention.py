"""Multi-head causal self-attention module using flash attention.

Vectorised variant: the inner key-block loop from flash_multihead_attention.py
is replaced by a single einsum that scores every causally-visible key at once.
Only the outer query-block loop remains, preserving the tiled structure while
eliminating all Python-level iteration over key blocks.
"""
import dataclasses
import math

from einops import rearrange
from jaxtyping import Float, Int
import torch


@dataclasses.dataclass
class FlashAttentionConfig:
    """Configuration for flash attention."""
    d_in: int
    d_out: int
    context_length: int
    dropout: float
    num_heads: int
    block_size: int = 64
    qkv_bias: bool = False

    def __post_init__(self):
        if self.d_out % self.num_heads != 0:
            raise ValueError('d_out must be divisible by num_heads.')


class MultiHeadAttention(torch.nn.Module):
    """Multi-head causal self-attention with fast attention."""

    def __init__(self, mha_config: FlashAttentionConfig) -> None:
        super().__init__()
        self._d_in = mha_config.d_in
        self._d_out = mha_config.d_out
        self._num_heads = mha_config.num_heads
        # Reduce the projection dim to match desired output dim.
        self._head_dim = mha_config.d_out // mha_config.num_heads
        self._block_size = mha_config.block_size

        self._W_query = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        self._W_key = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        self._W_value = torch.nn.Linear(
            mha_config.d_in, mha_config.d_out, bias=mha_config.qkv_bias)
        # Linear layer to combine head outputs.
        self._out_proj = torch.nn.Linear(mha_config.d_out, mha_config.d_out)
        self._dropout = torch.nn.Dropout(mha_config.dropout)

    def forward(self, x: Float[torch.Tensor, 'b t din']) -> (
            Float[torch.Tensor, 'b t dout']):
        """Compute multi-head causal self-attention over x."""
        device = x.device
        num_tokens = x.shape[1]
        Bs = self._block_size
        scale = self._head_dim ** -0.5

        keys: Float[torch.Tensor, 'b t dout'] = self._W_key(x)
        queries: Float[torch.Tensor, 'b t dout'] = self._W_query(x)
        values: Float[torch.Tensor, 'b t dout'] = self._W_value(x)

        keys = rearrange(
            keys,
            'b t (nh hd) -> (b nh) t hd',
            nh=self._num_heads,
            hd=self._head_dim)
        queries = rearrange(
            queries,
            'b t (nh hd) -> (b nh) t hd',
            nh=self._num_heads,
            hd=self._head_dim)
        values = rearrange(
            values,
            'b t (nh hd) -> (b nh) t hd',
            nh=self._num_heads,
            hd=self._head_dim)

        output: Float[torch.Tensor, 'b_nh t hd'] = torch.zeros_like(queries)
        num_blocks = math.ceil(num_tokens / self._block_size)
        for i_block in range(num_blocks):
            q_start = i_block * self._block_size
            q_end = min(q_start + self._block_size, num_tokens)
            block_size_q = q_end - q_start

            Q_i: Float[torch.Tensor, 'b_nh blk_q hd'] = (
                scale * queries[:, q_start:q_end, :])

            # Slice all causally-visible keys/values.
            K_causal: Float[torch.Tensor, 'b_nh t_k hd'] = keys[:, :q_end, :]
            V_causal: Float[torch.Tensor, 'b_nh t_k hd'] = values[:, :q_end, :]

            # Score every query against every causally-visible key at once by
            # contracting over the head-dimension d.
            S_ij: Float[torch.Tensor, 'b_nh blk_q t_k'] = torch.einsum(
                'bqd,bkd->bqk', Q_i, K_causal)

            # Build causal mask.
            q_pos: Int[torch.Tensor, '1 blk_q 1'] = torch.arange(
                q_start, q_end, device=device).view(1, block_size_q, 1)
            k_pos: Int[torch.Tensor, '1 1 t_k'] = torch.arange(
                q_end, device=device).view(1, 1, q_end)
            S_ij = S_ij.masked_fill(k_pos > q_pos, -torch.inf)

            m_i: Float[torch.Tensor, 'b_nh blk_q 1'] = S_ij.amax(
                dim=-1, keepdim=True)
            P_ij: Float[torch.Tensor, 'b_nh blk_q t_k'] = torch.exp(
                S_ij - m_i)
            P_ij = self._dropout(P_ij)
            l_i: Float[torch.Tensor, 'b_nh blk_q 1'] = P_ij.sum(
                dim=-1, keepdim=True)

            O_i: Float[torch.Tensor, 'b_nh blk_q hd'] = (
                torch.einsum('bqk,bkd->bqd', P_ij, V_causal) / l_i)
            output[:, q_start:q_end, :] = O_i

        # Restore shape for projection layer.
        output = rearrange(
            output, '(b nh) t hd -> b t (nh hd)', nh=self._num_heads)
        return self._out_proj(output)
