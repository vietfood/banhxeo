import torch
from jaxtyping import Float
from torch import nn

from banhxeo.model.llm.config import GPT2Config


class LayerNorm(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg

        if self.cfg.ln.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(cfg.d_model))
            self.bias = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self,
        residual: Float[torch.Tensor, "batch seq_len d_model"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:  # noqa: F722
        # Reference: https://nn.labml.ai/normalization/layer_norm/index.html

        # Calculate mean
        residual_mean = residual.mean(dim=-1, keepdim=True)

        # Calculate variance = std^2
        residual_var = residual.var(
            dim=-1, keepdim=True, correction=1
        )  # divide N-1 instead of N to unbias

        # Normalize input
        # x_norm = (x - mean) / sqrt(variance + epsilon)
        residual = (residual - residual_mean) / (residual_var + self.cfg.ln.eps).sqrt()

        if self.cfg.ln.elementwise_affine:
            # Layer norm with learnable parameters
            # output = x_norm * gamma + beta
            return residual * self.gain + (self.bias if self.cfg.ln.bias else 0)
        else:
            return residual
