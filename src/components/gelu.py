from jaxtyping import Float
import torch


class GELU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_scale_coeff",
            torch.sqrt(torch.tensor(2.0 / torch.pi)),
            persistent=False)
        self._cubic_coeff = 0.044715

    def forward(self, x: Float[torch.Tensor, '*dim']) -> (
            Float[torch.Tensor, '*dim']):
        scale_coeff = self._scale_coeff.to(device=x.device, dtype=x.dtype)
        return 0.5 * x * (1 + torch.tanh(scale_coeff * (
            x + self._cubic_coeff * torch.pow(x, 3))))

