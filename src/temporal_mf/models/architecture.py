"""1D Fourier neural operator with optional fidelity FiLM.

The widths are the preregistered placeholder from the study spec
(4 blocks, 12 modes, width 32) unless a config overrides them before the
test split is opened. This is an adapted backbone, not a Setinek checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        spec = torch.fft.rfft(x, dim=-1)
        out = torch.zeros(
            x.shape[0], self.weight.shape[1], spec.shape[-1], device=x.device, dtype=torch.cfloat
        )
        m = min(self.modes, spec.shape[-1])
        out[:, :, :m] = torch.einsum("bix,iox->box", spec[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=n, dim=-1)


class FidelityFiLM(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, 2 * width))

    def forward(self, h: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        scale, shift = self.net(xi).chunk(2, dim=-1)
        return h * (1.0 + scale[:, :, None]) + shift[:, :, None]


class FNO1d(nn.Module):
    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_blocks: int = 4,
        conditioned: bool = False,
    ):
        super().__init__()
        self.conditioned = conditioned
        self.lift = nn.Conv1d(1, width, 1)
        self.specs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_blocks)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_blocks)])
        self.films = nn.ModuleList([FidelityFiLM(width) for _ in range(n_blocks)]) if conditioned else None
        self.proj = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, u: torch.Tensor, xi: torch.Tensor | None = None) -> torch.Tensor:
        """u is (batch, nx). xi is (batch, 1) in [0, 1] when conditioned."""
        h = self.lift(u[:, None, :])
        for i, (spec, local) in enumerate(zip(self.specs, self.ws)):
            h = torch.nn.functional.gelu(spec(h) + local(h))
            if self.films is not None:
                if xi is None:
                    raise ValueError("conditioned model requires a fidelity token")
                h = self.films[i](h, xi)
        return self.proj(h)[:, 0, :]
