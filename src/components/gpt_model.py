from typing import Any, Dict

from jaxtyping import Float, Int
import torch

from components import layer_norm
from components import transformer_block


class GPTModel(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self._emb_dim: int = cfg['emb_dim']
        self._vocab_size: int = cfg['vocab_size']
        self._token_embedding = torch.nn.Embedding(
            self._vocab_size, self._emb_dim)
        self._positional_embedding = torch.nn.Embedding(
            cfg['context_length'], self._emb_dim)
        self._embedding_drop_out = torch.nn.Dropout(cfg['drop_out'])
        self._transformer_blocks = torch.nn.Sequential(
            *[transformer_block.TransformerBlock(cfg)
              for _ in range(cfg['n_layers'])])
        self._output_norm = layer_norm.LayerNorm(self._emb_dim)
        self._out_head = torch.nn.Linear(
            self._emb_dim, self._vocab_size, bias=False)

    def forward(self, tokens: Int[torch.Tensor, 'batch seq']) -> (
            Float[torch.Tensor, 'batch seq vocab']):
        seq_len = tokens.shape[1]
        token_embeds: Float[torch.Tensor, 'batch seq emb'] = (
            self._token_embedding(tokens))
        pos_embeds: Float[torch.Tensor, 'seq emb'] = (
            self._positional_embedding(
                torch.arange(seq_len, device=tokens.device)))
        x: Float[torch.Tensor, 'batch seq emb'] = token_embeds + pos_embeds
        x = self._embedding_drop_out(x)
        x = self._transformer_blocks(x)
        x = self._output_norm(x)
        logits: Float[torch.Tensor, 'batch seq vocab'] = self._out_head(x)
        return logits

