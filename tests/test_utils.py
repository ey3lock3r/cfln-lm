import torch
from cfln.utils import (
    complex_rope_multiplicative,
    complex_layer_norm,
    init_stiefel,
    verify_stiefel,
    entmax15_fast,
)


def test_crope_magnitude_preserved():
    """CRoPE multiplies by exp(iθ), |exp(iθ)|=1 so magnitudes must be unchanged."""
    x = torch.randn(4, 16, dtype=torch.cfloat)
    d_c = x.shape[-1]
    out = complex_rope_multiplicative(x, t=7, d_c=d_c, rope_base=10000.0)
    torch.testing.assert_close(out.abs(), x.abs(), rtol=1e-5, atol=0.0)


def test_complex_layer_norm_phase_preserved():
    """RMS norm must preserve phase exactly and normalise mean squared magnitude to 1."""
    x = torch.randn(8, 32, dtype=torch.cfloat)
    out = complex_layer_norm(x, dims=[-1])
    # Phase preservation
    torch.testing.assert_close(torch.angle(out), torch.angle(x), atol=1e-5, rtol=0.0)
    # Mean squared magnitude ≈ 1 across feature dim
    mean_mag_sq = (out.real.pow(2) + out.imag.pow(2)).mean(dim=-1)
    torch.testing.assert_close(mean_mag_sq, torch.ones_like(mean_mag_sq), atol=1e-4, rtol=0.0)


def test_verify_stiefel_after_init():
    """init_stiefel must produce a matrix satisfying W @ W†= I."""
    W = init_stiefel(32, 64)
    assert verify_stiefel(W), "W from init_stiefel must satisfy the Stiefel constraint"


def test_entmax15_sums_to_one():
    """entmax15_fast output must be non-negative and sum to 1."""
    z = torch.randn(6, 20)
    p = entmax15_fast(z, dim=-1)
    # Non-negative
    assert (p >= 0).all(), "entmax15 output must be non-negative"
    # Sums to 1 along last dim
    sums = p.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=0.0)
