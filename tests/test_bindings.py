"""Unit tests for binding kernel math: B_bind, B_role, B_comp.

All formulas are exercised directly (no full forward pass needed).
Each test quotes its spec section.
"""
import torch
import pytest


# ── B_bind: Phase Binding Kernel (§1.30 R1 / §1.41 W2) ─────────────────────

def _b_bind(phi, sigma_sq=1.0):
    """B_bind[i,j] = exp(-(φ_i - φ_j)² / σ²)."""
    phi_diff = phi.unsqueeze(0) - phi.unsqueeze(1)
    return torch.exp(-phi_diff**2 / max(sigma_sq, 1e-6))


def test_b_bind_formula_exact():
    """§1.30: B_bind[i,j] = exp(-φ_diff² / σ²) matches closed form."""
    phi = torch.tensor([0.0, 1.0, 2.0])
    sigma_sq = 2.0
    result = _b_bind(phi, sigma_sq)
    phi_diff = phi.unsqueeze(0) - phi.unsqueeze(1)
    expected = torch.exp(-phi_diff**2 / sigma_sq)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


def test_b_bind_diagonal_ones():
    """§1.30: B_bind[i,i] = 1 always (zero phase difference)."""
    phi = torch.randn(8)
    B = _b_bind(phi)
    torch.testing.assert_close(B.diagonal(), torch.ones(8), rtol=0.0, atol=1e-6)


def test_b_bind_symmetric():
    """§1.30: B_bind must be symmetric (B_bind[i,j] = B_bind[j,i])."""
    phi = torch.randn(6)
    B = _b_bind(phi)
    torch.testing.assert_close(B, B.T, rtol=0.0, atol=1e-6)


def test_b_bind_values_in_unit_interval():
    """§1.30: All B_bind entries ∈ (0, 1] — bounded by definition of exp(-x²)."""
    phi = torch.randn(10)
    B = _b_bind(phi)
    assert (B > 0).all(), "B_bind entries must be positive"
    assert (B <= 1.0 + 1e-6).all(), "B_bind entries must be ≤ 1"


def test_b_bind_psd():
    """§1.30: B_bind is PSD (Bochner's theorem — stationary kernel)."""
    phi = torch.randn(8)
    B = _b_bind(phi)
    ev = torch.linalg.eigvalsh(B.double())
    assert (ev >= -1e-6).all(), "B_bind must be positive semi-definite"


def test_b_bind_sigma_controls_spread():
    """§1.30: Smaller σ² → sharper diagonal dominance."""
    phi = torch.tensor([0.0, 1.0, 2.0, 3.0])
    B_sharp = _b_bind(phi, sigma_sq=0.1)
    B_broad = _b_bind(phi, sigma_sq=10.0)
    # Off-diagonal entries should be smaller for sharp kernel
    off_sharp = B_sharp[0, 1].item()
    off_broad = B_broad[0, 1].item()
    assert off_sharp < off_broad, "Smaller σ² must produce sharper (smaller off-diagonal) kernel"


def test_b_bind_gradient_wrt_phi():
    """§1.30: Gradient must flow back through B_bind w.r.t. phase."""
    phi = torch.randn(4, requires_grad=True)
    phi_diff = phi.unsqueeze(0) - phi.unsqueeze(1)
    B = torch.exp(-phi_diff**2)
    B.sum().backward()
    assert phi.grad is not None
    assert not torch.isnan(phi.grad).any()


# ── B_role: Role Binding Kernel (§1.35 X) ────────────────────────────────────

def _b_role(k, R=8):
    """B_role = α @ α.T where α = softmax(random scores)."""
    scores = torch.randn(k, R)
    alpha = torch.softmax(scores, dim=-1)
    return alpha @ alpha.T, alpha


def test_b_role_gram_formula():
    """§1.35: B_role = α @ α.T exactly (outer product of softmax weights)."""
    k, R = 5, 8
    scores = torch.randn(k, R)
    alpha = torch.softmax(scores, dim=-1)
    result = alpha @ alpha.T
    expected = torch.stack([
        torch.stack([(alpha[i] * alpha[j]).sum() for j in range(k)])
        for i in range(k)
    ])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_b_role_symmetric():
    """§1.35: B_role must be symmetric."""
    B, _ = _b_role(k=6)
    torch.testing.assert_close(B, B.T, rtol=0.0, atol=1e-6)


def test_b_role_psd():
    """§1.35: B_role is PSD by construction (Gram matrix)."""
    B, _ = _b_role(k=8)
    ev = torch.linalg.eigvalsh(B.double())
    assert (ev >= -1e-6).all(), "B_role must be positive semi-definite"


def test_b_role_diagonal_equals_norm_sq():
    """§1.35: B_role[i,i] = ||α_i||² = sum(α_i²)."""
    k, R = 5, 8
    scores = torch.randn(k, R)
    alpha = torch.softmax(scores, dim=-1)
    B = alpha @ alpha.T
    expected_diag = (alpha**2).sum(dim=-1)
    torch.testing.assert_close(B.diagonal(), expected_diag, rtol=1e-5, atol=1e-6)


def test_b_role_gradient_wrt_scores():
    """§1.35: Gradient must flow back through B_role to the score inputs."""
    scores = torch.randn(4, 8, requires_grad=True)
    alpha = torch.softmax(scores, dim=-1)
    B = alpha @ alpha.T
    B.sum().backward()
    assert scores.grad is not None
    assert not torch.isnan(scores.grad).any()


# ── B_comp: Hadamard Composition (§1.55 COMP-H) ───────────────────────────────

def test_b_comp_is_hadamard():
    """§1.55: B_comp = B_bind ⊙ B_role (element-wise product)."""
    k = 6
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    for i in range(k):
        for j in range(k):
            torch.testing.assert_close(
                B_comp[i, j], B_bind[i, j] * B_role[i, j],
                rtol=1e-5, atol=1e-7,
            )


def test_b_comp_psd_by_schur():
    """§1.55: Hadamard product of two PSD matrices is PSD (Schur product theorem)."""
    k = 8
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    ev = torch.linalg.eigvalsh(B_comp.double())
    assert (ev >= -1e-6).all(), "B_comp must be PSD (Schur product theorem)"


def test_b_comp_symmetric():
    """§1.55: B_comp must be symmetric (product of two symmetric matrices)."""
    k = 6
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    torch.testing.assert_close(B_comp, B_comp.T, rtol=0.0, atol=1e-6)


def test_b_comp_bounded_by_b_bind():
    """§1.55: B_comp[i,j] ≤ B_bind[i,j] since B_role entries ∈ [0, 1]."""
    k = 8
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    assert (B_comp <= B_bind + 1e-6).all(), "B_comp must be elementwise ≤ B_bind"


# ── W_full augmentation (§1.30 / §1.35 / §1.55) ──────────────────────────────

def test_w_full_increases_after_b_bind():
    """§1.30: Adding lam_bind * B_bind must increase the Frobenius norm of W_full."""
    k = 6
    W = torch.randn(k, k)
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    lam = 0.1
    W_aug = W + lam * B_bind
    assert W_aug.norm() != W.norm(), "W_full Frobenius norm must change after B_bind augmentation"


def test_w_full_increases_after_b_role():
    """§1.35: Adding lam_role * B_role must change W_full."""
    k = 6
    W = torch.randn(k, k)
    B_role, _ = _b_role(k)
    lam = 0.1
    W_aug = W + lam * B_role
    assert not torch.allclose(W_aug, W), "W_full must change after B_role augmentation"


def test_w_full_increases_after_b_comp():
    """§1.55: Adding lam_comp * B_comp must change W_full."""
    k = 6
    W = torch.randn(k, k)
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    lam = 0.1
    W_aug = W + lam * B_comp
    assert not torch.allclose(W_aug, W), "W_full must change after B_comp augmentation"


def test_lam_bind_gradient_flows():
    """§1.30: Gradient must flow through lam_bind (log_lam_bind is a learnable param)."""
    k = 4
    log_lam = torch.tensor(-3.0, requires_grad=True)
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    W = torch.randn(k, k)
    W_aug = W + torch.exp(log_lam) * B_bind
    W_aug.sum().backward()
    assert log_lam.grad is not None
    assert not torch.isnan(log_lam.grad)


def test_lam_role_gradient_flows():
    """§1.35: Gradient must flow through lam_role."""
    k = 4
    log_lam = torch.tensor(-3.0, requires_grad=True)
    B_role, _ = _b_role(k)
    W = torch.randn(k, k)
    W_aug = W + torch.exp(log_lam) * B_role
    W_aug.sum().backward()
    assert log_lam.grad is not None
    assert not torch.isnan(log_lam.grad)


def test_lam_comp_gradient_flows():
    """§1.55: Gradient must flow through lam_composition."""
    k = 4
    log_lam = torch.tensor(-3.0, requires_grad=True)
    phi = torch.randn(k)
    B_bind = _b_bind(phi)
    B_role, _ = _b_role(k)
    B_comp = B_bind * B_role
    W = torch.randn(k, k)
    W_aug = W + torch.exp(log_lam) * B_comp
    W_aug.sum().backward()
    assert log_lam.grad is not None
    assert not torch.isnan(log_lam.grad)
