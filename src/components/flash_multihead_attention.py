"""Multi-head causal self-attention module using flash attention."""
import dataclasses

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
    block_size: int
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

    def _apply_causal_mask(
            self,
            x: Float[torch.Tensor, 'b_nh blk blk'],
            q_indices: Int[torch.Tensor, 'blk 1'],
            k_idx: int,
            block_size_k: int) -> Float[torch.Tensor, 'b_nh blk blk']:
        k_indices: Int[torch.Tensor, '1 blk'] = torch.arange(
            k_idx, k_idx+block_size_k, device=x.device).unsqueeze(0)
        causal_mask: Int[torch.Tensor, 'blk blk'] = (k_indices > q_indices)
        return x.masked_fill(causal_mask, -torch.inf)

    def forward(self, x: Float[torch.Tensor, 'b t din']) -> (
            Float[torch.Tensor, 'b t dout']):
        """Compute multi-head causal self-attention over x."""
        dtype = x.dtype
        batch_size = x.shape[0]
        num_tokens = x.shape[1]
        bh = batch_size * self._num_heads
        scale = self._head_dim ** -0.5

        keys: Float[torch.Tensor,'b t dout'] = self._W_key(x)
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
        for i in range(0, num_tokens, self._block_size):
            Q_i: Float[torch.Tensor, 'b_nh blk hd'] = (
                scale * queries[:, i:i+self._block_size, :])
            # The final block may be smaller than self._block_size, so we need
            # to record the actual block size used.
            block_size = Q_i.shape[1]
            q_indices = torch.arange(
                i, i + block_size, device=x.device, dtype=torch.int32).unsqueeze(-1)
            m_i = torch.full(
                (bh, block_size), -torch.inf, device=x.device, dtype=dtype)
            l_i = torch.zeros((bh, block_size), device=x.device, dtype=dtype)
            O_i = torch.zeros(
                (bh, block_size, self._head_dim), device=x.device, dtype=dtype)
            for j in range(0, i + block_size, self._block_size):
                K_jT = rearrange(
                    keys[:, j:j+self._block_size, :],
                    'b_nh blk hd -> b_nh hd blk')
                block_size_K = K_jT.shape[2]
                V_j = values[:, j:j+self._block_size, :]
                S_ij: Float[torch.Tensor, 'b_nh blk blk'] = torch.matmul(
                    Q_i, K_jT)
                S_ij = self._apply_causal_mask(
                    S_ij, q_indices, j, block_size_K)

                m_ij = torch.max(S_ij, dim=-1).values
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                P_ij = self._dropout(P_ij)
                l_ij = torch.sum(P_ij, dim=-1)
                m_new = torch.maximum(m_i, m_ij)

                alpha = torch.exp(m_i - m_new)
                beta = torch.exp(m_ij - m_new)
                l_i = alpha * l_i + beta * l_ij

                O_i = (alpha.unsqueeze(-1) * O_i +
                    beta.unsqueeze(-1) * torch.matmul(P_ij, V_j))
                m_i = m_new
            output[:, i:i+self._block_size] = O_i / l_i.unsqueeze(-1)

        # ---- Restore shape ----
        output = rearrange(
            output, '(b nh) t dout -> b t (nh dout)', nh=self._num_heads)
        return self._out_proj(output)
