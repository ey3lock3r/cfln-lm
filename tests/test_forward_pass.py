"""E2E tests for CFLNModel.forward() — shapes, dtypes, invariants, gradient flow.

All tests use the tiny_model / tiny_cfg fixtures from conftest.py.
"""
import torch
import pytest


# ── Output shape and dtype ────────────────────────────────────────────────────

def test_forward_returns_three_tuple(tiny_model, tiny_cfg):
    """§2.23: forward() must return (logits, U_fin, aux)."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    result = tiny_model(ids, training=False)
    assert isinstance(result, tuple) and len(result) == 3


def test_logits_shape(tiny_model, tiny_cfg):
    """§2.23: logits shape must be (B, T, vocab_size)."""
    B, T, V = 1, tiny_cfg['T'], tiny_cfg['vocab_size']
    ids = torch.randint(0, V, (B, T))
    logits, _, _ = tiny_model(ids, training=False)
    assert logits.shape == (B, T, V), f"Expected ({B},{T},{V}), got {logits.shape}"


def test_logits_dtype_is_float32(tiny_model, tiny_cfg):
    """Architecture invariant: logits must be float32, never cfloat."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    logits, _, _ = tiny_model(ids, training=False)
    assert logits.dtype == torch.float32, f"logits must be float32, got {logits.dtype}"


def test_u_fin_shape(tiny_model, tiny_cfg):
    """§2.15: U_fin shape must be (B, T)."""
    B, T = 1, tiny_cfg['T']
    ids = torch.randint(0, tiny_cfg['vocab_size'], (B, T))
    _, U_fin, _ = tiny_model(ids, training=False)
    assert U_fin.shape == (B, T), f"Expected ({B},{T}), got {U_fin.shape}"


def test_u_fin_in_unit_interval(tiny_model, tiny_cfg):
    """§2.15: U_fin values must lie in [0, 1]."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    _, U_fin, _ = tiny_model(ids, training=False)
    assert (U_fin >= 0.0).all() and (U_fin <= 1.0).all(), "U_fin must be in [0, 1]"


# ── No NaN / Inf ──────────────────────────────────────────────────────────────

def test_no_nan_in_logits(tiny_model, tiny_cfg):
    """Smoke: logits must contain no NaN or Inf after forward."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    logits, _, _ = tiny_model(ids, training=False)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"


def test_no_nan_in_u_fin(tiny_model, tiny_cfg):
    """Smoke: U_fin must contain no NaN after forward."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    _, U_fin, _ = tiny_model(ids, training=False)
    assert not torch.isnan(U_fin).any(), "NaN in U_fin"


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_loss_backward_no_crash(tiny_model, tiny_cfg):
    """§3.5: loss.backward() must run without error."""
    T, V = tiny_cfg['T'], tiny_cfg['vocab_size']
    ids = torch.randint(0, V, (1, T))
    logits, _, _ = tiny_model(ids, training=True)
    targets = torch.randint(0, V, (1, T - 1))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :T - 1, :].reshape(-1, V),
        targets.reshape(-1),
    )
    loss.backward()


def test_trainable_params_have_grad(tiny_model, tiny_cfg):
    """§3.5: At least one trainable parameter must have a non-zero gradient after backward."""
    T, V = tiny_cfg['T'], tiny_cfg['vocab_size']
    ids = torch.randint(0, V, (1, T))
    logits, _, _ = tiny_model(ids, training=True)
    targets = torch.randint(0, V, (1, T - 1))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :T - 1, :].reshape(-1, V), targets.reshape(-1),
    )
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tiny_model.parameters()
        if p.requires_grad
    )
    assert has_grad, "No trainable parameter has a non-zero gradient after backward"


def test_no_nan_in_grads(tiny_model, tiny_cfg):
    """§3.5: No NaN in gradients after backward."""
    T, V = tiny_cfg['T'], tiny_cfg['vocab_size']
    ids = torch.randint(0, V, (1, T))
    logits, _, _ = tiny_model(ids, training=True)
    targets = torch.randint(0, V, (1, T - 1))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :T - 1, :].reshape(-1, V), targets.reshape(-1),
    )
    loss.backward()
    for name, p in tiny_model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ── Training vs inference mode ────────────────────────────────────────────────

def test_titans_m_updates_during_forward(tiny_model, tiny_cfg):
    """§2.3: Titans M is an online associative memory — updates every forward pass."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    M_before = tiny_model.encoder.titans.M.clone()
    tiny_model(ids, training=True)
    changed = not torch.allclose(tiny_model.encoder.titans.M, M_before)
    assert changed, "Titans M must update on each forward (Wirtinger rank-1 update)"


def test_titans_m_frozen_in_thinking_mode(tiny_model, tiny_cfg):
    """§2.3 / CTP: _in_thinking_mode suppresses Titans M update."""
    tiny_model.encoder.titans._in_thinking_mode = True
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    M_before = tiny_model.encoder.titans.M.clone()
    tiny_model(ids, training=True)
    tiny_model.encoder.titans._in_thinking_mode = False
    torch.testing.assert_close(tiny_model.encoder.titans.M, M_before, rtol=0.0, atol=1e-5)


# ── VQ write active (OI-8) ────────────────────────────────────────────────────

def test_vq_l1_ptr_advances_after_forward(tiny_model, tiny_cfg):
    """§1.59 OI-8: bank._L1_ptr must advance after a forward pass (VQ writes active)."""
    ptr_before = tiny_model.bank._L1_ptr
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    tiny_model(ids, training=True)
    assert tiny_model.bank._L1_ptr > ptr_before, \
        "bank._L1_ptr must advance — VQ writes were silently suppressed (OI-8)"


# ── reset_for_inference ───────────────────────────────────────────────────────

def test_reset_for_inference_clears_state(tiny_model, tiny_cfg):
    """§2.23: reset_for_inference() must zero g_c, clear goal_stack, zero r_lista."""
    ids = torch.randint(0, tiny_cfg['vocab_size'], (1, tiny_cfg['T']))
    tiny_model(ids, training=True)
    tiny_model.reset_for_inference()
    cun = tiny_model.diff_aux.cun
    assert cun._goal_stack == [], "goal_stack must be empty after reset"
    assert cun.r_lista.abs().sum().item() == 0.0, "r_lista must be zero after reset"
