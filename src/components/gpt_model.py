"""GPT language model assembling components."""
import dataclasses

from jaxtyping import Float, Int
import torch

from components import layer_norm
from components import transformer_block


@dataclasses.dataclass
class GPTModelConfig:
    """Configuration for the GPT model."""
    vocab_size: int
    n_layers: int
    drop_out: float
    transformer_block_config: transformer_block.TransformerBlockConfig

    @property
    def emb_dim(self) -> int:
        """Network embedding dimension."""
        return self.transformer_block_config.emb_dim

    @property
    def context_length(self) -> int:
        """Model context length."""
        return self.transformer_block_config.mha_config.context_length


class GPTModel(torch.nn.Module):
    """GPT-2 style autoregressive language model."""

    def __init__(self, cfg: GPTModelConfig) -> None:
        super().__init__()
        self._emb_dim: int = cfg.emb_dim
        self._vocab_size: int = cfg.vocab_size
        self._token_embedding = torch.nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self._positional_embedding = torch.nn.Embedding(
            cfg.context_length, cfg.emb_dim)
        self._embedding_drop_out = torch.nn.Dropout(cfg.drop_out)
        self._transformer_blocks = torch.nn.Sequential(
            *[transformer_block.TransformerBlock(cfg.transformer_block_config)
              for _ in range(cfg.n_layers)])
        self._output_norm = layer_norm.LayerNorm(
            layer_norm.LayerNormConfig(emb_dim=cfg.emb_dim))
        self._out_head = torch.nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: Int[torch.Tensor, 'batch seq']) -> (
            Float[torch.Tensor, 'batch seq vocab']):
        """Return raw logits for each position in the sequence."""
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
