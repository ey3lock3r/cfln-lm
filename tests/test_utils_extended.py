"""Unit tests for cfln.utils math primitives. Each test quotes its spec section."""
import math
import torch
import pytest
from cfln.utils import (
    rq_routing, compute_energies, entmax15_fast, entmax15_with_floor,
    complex_rope_multiplicative, complex_layer_norm,
    init_stiefel, verify_stiefel, apply_psd_to_weight_matrix,
    batched_cayley_retraction, batched_cayley_with_per_unit_lr,
)


# ── RQ Routing (§1.3) ───────────────────────────────────────────────────────

def test_rq_kernel_closed_form():
    """§1.3: rq_routing must match (1 + E/ℓ²)^(-α) exactly."""
    E = torch.tensor([[0.0, 0.5, 1.0, 2.0]])
    log_alpha = torch.full((4,), math.log(1.5))
    log_ell   = torch.zeros(4)
    expected  = (1.0 + E) ** -1.5
    result    = rq_routing(E, log_alpha, log_ell)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


def test_rq_kernel_zero_energy_is_one():
    """§1.3: At E=0 (centroid), routing weight = 1 for all α, ℓ."""
    E = torch.zeros(1, 8)
    log_alpha = torch.randn(8)
    log_ell   = torch.randn(8)
    result    = rq_routing(E, log_alpha, log_ell)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0.0, atol=1e-6)


def test_rq_kernel_gradient_wrt_e_is_negative():
    """§1.3: ∂p/∂E < 0 — higher energy reduces routing weight."""
    E = torch.tensor([[0.5, 1.0, 2.0]], requires_grad=True)
    log_alpha = torch.zeros(3)
    log_ell   = torch.zeros(3)
    p = rq_routing(E, log_alpha, log_ell)
    p.sum().backward()
    assert E.grad is not None
    assert (E.grad < 0).all(), "∂p/∂E must be negative everywhere"


def test_rq_kernel_alpha_clamp():
    """§1.3: alpha clamped to [0.1, 10.0] — extreme log_alpha doesn't crash."""
    E = torch.ones(1, 4)
    log_alpha = torch.tensor([-100.0, -1.0, 1.0, 100.0])  # exp spans huge range
    log_ell   = torch.zeros(4)
    result    = rq_routing(E, log_alpha, log_ell)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_rq_kernel_ell_clamp():
    """§1.3: ell_sq clamped to [1e-4] — log_ell=-inf doesn't divide by zero."""
    E = torch.ones(1, 4)
    log_alpha = torch.zeros(4)
    log_ell   = torch.tensor([-100.0, -10.0, 0.0, 10.0])
    result    = rq_routing(E, log_alpha, log_ell)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


# ── Compute Energies (§1.1) ─────────────────────────────────────────────────

def test_compute_energies_zero_at_centroid():
    """§1.1: E[i,i]=0 — energy is zero when query equals its own centroid."""
    d_c, d_e, N = 8, 4, 6
    mu  = torch.randn(N, d_c, dtype=torch.cfloat)
    W   = init_stiefel(d_e, d_c).unsqueeze(0).expand(N, -1, -1)
    E   = compute_energies(mu, W, mu)           # (N, N); diagonal = self-energy
    torch.testing.assert_close(E.diagonal(), torch.zeros(N), rtol=0.0, atol=1e-5)


def test_compute_energies_nonnegative():
    """§1.1: Energies must be ≥ 0 (they are squared norms)."""
    d_c, d_e, N, B = 8, 4, 6, 3
    x_c = torch.randn(B, d_c, dtype=torch.cfloat)
    mu  = torch.randn(N, d_c, dtype=torch.cfloat)
    W   = torch.stack([init_stiefel(d_e, d_c) for _ in range(N)])
    E   = compute_energies(x_c, W, mu)
    assert (E >= -1e-6).all(), "Energies must be non-negative"


def test_compute_energies_gradient_exists():
    """§1.1: Gradient must flow back through energy computation."""
    d_c, d_e, N = 8, 4, 4
    x_c = torch.randn(2, d_c, dtype=torch.cfloat, requires_grad=True)
    mu  = torch.randn(N, d_c, dtype=torch.cfloat)
    W   = torch.stack([init_stiefel(d_e, d_c) for _ in range(N)])
    E   = compute_energies(x_c, W, mu)
    E.real.sum().backward()
    assert x_c.grad is not None
    assert not torch.isnan(x_c.grad).any()


# ── Entmax-1.5 (§1.4) ───────────────────────────────────────────────────────

def test_entmax15_sums_to_one_extended():
    """§1.4: entmax-1.5 output sums to 1 on arbitrary inputs."""
    z = torch.randn(4, 16)
    p = entmax15_fast(z, dim=-1)
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(4), rtol=0.0, atol=1e-5)


def test_entmax15_nonnegative():
    """§1.4: entmax-1.5 output is non-negative (sparse, not soft)."""
    z = torch.randn(4, 16)
    p = entmax15_fast(z, dim=-1)
    assert (p >= 0).all()


def test_entmax15_sparsity():
    """§1.4: entmax-1.5 produces exact zeros for dominated entries."""
    z = torch.tensor([[10.0, 0.0, 0.0, -10.0]])  # extreme spread
    p = entmax15_fast(z, dim=-1)
    assert (p[0, 1:] == 0.0).all(), "Dominated entries must be exactly zero"
    assert p[0, 0] > 0.99, "Dominant entry must receive near-full weight"


def test_entmax15_gradient_flows():
    """§1.4: Gradient must flow back through entmax."""
    z = torch.randn(2, 8, requires_grad=True)
    p = entmax15_fast(z, dim=-1)
    p.sum().backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any()


def test_entmax15_with_floor_clamps_minimum():
    """§1.4: entmax_with_floor ensures no weight is exactly zero (floor=eps)."""
    z = torch.tensor([[10.0, 0.0, 0.0, -10.0]])
    eps = 1e-4
    p = entmax15_with_floor(z, eps, dim=-1)
    assert (p >= eps).all(), f"All weights must be ≥ {eps}"


# ── Complex Layer Norm (A8) ──────────────────────────────────────────────────

def test_complex_layer_norm_rms_equals_one():
    """A8: After RMS norm, mean(|z_k|²) = 1 across feature dim."""
    x = torch.randn(4, 16, dtype=torch.cfloat)
    y = complex_layer_norm(x, dims=[-1])
    mag_sq = (y.real.pow(2) + y.imag.pow(2)).mean(dim=-1)
    torch.testing.assert_close(mag_sq, torch.ones(4), rtol=0.0, atol=5e-5)


def test_complex_layer_norm_phase_preserved():
    """A8: Phase (angle) must be exactly preserved — RMS norm scales magnitude only."""
    x = torch.randn(4, 16, dtype=torch.cfloat)
    y = complex_layer_norm(x, dims=[-1])
    torch.testing.assert_close(torch.angle(y), torch.angle(x), rtol=0.0, atol=1e-5)


def test_complex_layer_norm_gradient_exists():
    """A8: Gradient must flow through complex layer norm."""
    x = torch.randn(2, 8, dtype=torch.cfloat, requires_grad=True)
    y = complex_layer_norm(x, dims=[-1])
    y.abs().sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.cfloat


# ── CRoPE (A6/§1.3) ─────────────────────────────────────────────────────────

def test_crope_magnitude_exactly_preserved():
    """A6/§1.3: |exp(iθ)| = 1 → CRoPE preserves magnitude exactly."""
    x = torch.randn(4, 16, dtype=torch.cfloat)
    y = complex_rope_multiplicative(x, t=42, d_c=16, rope_base=10000.0)
    torch.testing.assert_close(y.abs(), x.abs(), rtol=0.0, atol=1e-5)


def test_crope_different_positions_differ():
    """A6: Different positions must produce different encodings."""
    x = torch.ones(1, 16, dtype=torch.cfloat)
    y1 = complex_rope_multiplicative(x, t=0,  d_c=16)
    y2 = complex_rope_multiplicative(x, t=1,  d_c=16)
    assert not torch.allclose(y1, y2), "Different positions must give different encodings"


# ── Stiefel / Cayley (A10/§1.10) ────────────────────────────────────────────

def test_init_stiefel_satisfies_constraint():
    """A10: init_stiefel must produce W where W@W†=I."""
    W = init_stiefel(d_e=4, d_c=8)
    assert verify_stiefel(W, tol=1e-5), "init_stiefel must satisfy W@W†=I"


def test_cayley_retraction_preserves_stiefel():
    """A10/§1.10: batched_cayley_retraction must maintain W@W†=I after update."""
    n, d_e, d_c = 6, 4, 8
    W = torch.stack([init_stiefel(d_e, d_c) for _ in range(n)])  # (n, d_e, d_c)
    G = torch.randn_like(W) * 0.01  # small gradient
    lr = torch.full((n,), 1e-3)
    W_new = batched_cayley_with_per_unit_lr(W, G, lr)
    for i in range(n):
        assert verify_stiefel(W_new[i], tol=1e-4), \
            f"Cayley retraction must preserve Stiefel constraint (unit {i})"


def test_cayley_retraction_gradient_exists():
    """A10: Gradient must flow through Cayley retraction."""
    n, d_e, d_c = 4, 4, 8
    W = torch.stack([init_stiefel(d_e, d_c) for _ in range(n)]).requires_grad_(True)
    G = torch.randn(n, d_e, d_c, dtype=torch.cfloat)
    lr = torch.full((n,), 1e-3)
    W_new = batched_cayley_with_per_unit_lr(W, G, lr)
    W_new.abs().sum().backward()
    assert W.grad is not None
    assert not torch.isnan(W.grad).any()


# ── PSD Projection (§1.11) ───────────────────────────────────────────────────

def test_apply_psd_all_eigenvalues_nonneg():
    """§1.11: After PSD projection, all eigenvalues must be ≥ eps."""
    W = torch.randn(8, 8)
    W_psd = apply_psd_to_weight_matrix(W)
    ev = torch.linalg.eigvalsh(W_psd.float())
    assert (ev >= -1e-6).all(), "PSD projection must eliminate negative eigenvalues"


def test_apply_psd_symmetric():
    """§1.11: PSD projection output must be symmetric (W = Wᵀ)."""
    W = torch.randn(8, 8)
    W_psd = apply_psd_to_weight_matrix(W)
    torch.testing.assert_close(W_psd, W_psd.T, rtol=0.0, atol=1e-5)


def test_apply_psd_already_psd_unchanged():
    """§1.11: Already-PSD matrix should be nearly unchanged."""
    v = torch.randn(8, 4)
    W_psd_in = (v @ v.T).float()
    W_psd_out = apply_psd_to_weight_matrix(W_psd_in)
    torch.testing.assert_close(W_psd_in, W_psd_out, rtol=0.0, atol=1e-4)
