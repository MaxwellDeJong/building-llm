from jaxtyping import Float
import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int, unbiased: bool = False) -> None:
        super().__init__()
        self._emb_dim = emb_dim
        self._eps = 1e-5
        self._unbiased = unbiased
        self._scale = torch.nn.Parameter(torch.ones(emb_dim))
        self._shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(
            self,
            x: Float[torch.Tensor, '*batch emb']) -> (
                Float[torch.Tensor, '*batch emb']):
        if x.shape[-1] != self._emb_dim:
            raise TypeError(
                f'Expected last dim == {self.emb_dim}, got {x.shape[-1]}.')
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=self._unbiased)
        norm_x = (x - mean) / torch.sqrt(var + self._eps)
        return self._scale * norm_x + self._shift

