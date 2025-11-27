from einops import rearrange
from jaxtyping import Bool, Float
from jaxtyping import install_import_hook
install_import_hook("mha", "typeguard")
import torch

from math_lib import matmul


class MultiHeadAttention(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            context_length: int,
            dropout: float,
            num_heads: int,
            qkv_bias: bool = False) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads."

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce the projection dim to match desired output dim.
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        # Linear layer to combine head outputs.
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Float[torch.Tensor, 'b t din']) -> (
            Float[torch.Tensor, 'b t dout']):
        num_tokens = x.shape[1]

        keys: Float[torch.Tensor,'b t dout'] = self.W_key(x)
        queries: Float[torch.Tensor, 'b t dout'] = self.W_query(x)
        values: Float[torch.Tensor, 'b t dout'] = self.W_value(x)

        keysT = rearrange(
            keys,
            'b t (nh hd) -> b nh hd t',
            nh=self.num_heads,
            hd=self.head_dim)
        values = rearrange(
            values,
            'b t (nh hd) -> b nh t hd',
            nh=self.num_heads,
            hd=self.head_dim)
        queries = rearrange(
            queries,
            'b t (nh hd) -> b nh t hd',
            nh=self.num_heads,
            hd=self.head_dim)

        # Compute scaled dot-product attention (aka self-attention) with a
        # causal mask.
        attn_scores: Float[Tensor, 'b nh t t'] = matmul.matmul(queries, keysT)
        
        # Original mask truncated to the number of tokens and converted to
        # boolean.
        mask_bool: Bool[Tensor, 't, t'] = self.mask.bool()[
            :num_tokens, :num_tokens]

        # Use the mask to fill attention scores.
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec: Float[Tensor, 'b nh t hd'] = matmul.matmul(
            attn_weights, values)
        context_vec = rearrange(context_vec, 'b nh t hd -> b t (nh hd)')
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

