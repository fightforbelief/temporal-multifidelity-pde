"""Periodic Fourier interpolation onto a common grid."""

from __future__ import annotations

import torch


def fourier_resample(u: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample the last dimension. Identity when ``n_out`` equals the input size."""
    n_in = u.shape[-1]
    if n_out == n_in:
        return u.clone()
    spec = torch.fft.rfft(u, dim=-1)
    n_freq_out = n_out // 2 + 1
    out_spec = torch.zeros(*u.shape[:-1], n_freq_out, dtype=spec.dtype, device=u.device)
    n_copy = min(spec.shape[-1], n_freq_out)
    out_spec[..., :n_copy] = spec[..., :n_copy]
    # Nyquist handling when shrinking even grids: keep a real Nyquist.
    if n_out < n_in and n_out % 2 == 0:
        out_spec[..., -1] = out_spec[..., -1].real.to(out_spec.dtype)
    scaled = out_spec * (n_out / n_in)
    return torch.fft.irfft(scaled, n=n_out, dim=-1)
