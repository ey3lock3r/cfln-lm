"""E2E tests for train_step_v605 — ordering invariants, param updates, post-step state.

All tests use tiny_model / tiny_cfg from conftest.py.
"""
import torch
import pytest
from cfln.training.train_step import train_step_v605
from cfln.training.optimizers import build_optimizers_v605
from cfln.modules.si import SynapticIntelligence
from cfln.utils import verify_stiefel


def _make_batch(tiny_cfg, device='cpu'):
    T, B = tiny_cfg['T'], tiny_cfg['B']
    return {'input_ids': torch.randint(0, tiny_cfg['vocab_size'], (B, T))}


def _build_si(tiny_cfg):
    return SynapticIntelligence(
        c_SI=tiny_cfg['c_SI'],
        rho_SI=tiny_cfg['rho_SI'],
        beta_SI=tiny_cfg['beta_SI'],
    )


# ── Full step does not crash ──────────────────────────────────────────────────

def test_train_step_v605_runs_without_error(tiny_model, tiny_cfg):
    """§3.5: train_step_v605 must complete one full step without error."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    result = train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                              total_steps=100, cfg=tiny_cfg)
    assert isinstance(result, dict)


def test_train_step_returns_l_task(tiny_model, tiny_cfg):
    """§3.5: train_step must return L_task as a finite float."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    result = train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                              total_steps=100, cfg=tiny_cfg)
    assert 'L_task' in result
    assert isinstance(result['L_task'], float)
    assert not (result['L_task'] != result['L_task'])  # not NaN


# ── Stiefel constraint holds after step ───────────────────────────────────────

def test_w_l_satisfies_stiefel_after_step(tiny_model, tiny_cfg):
    """§1.10 / A10: W_l units must satisfy W@W†=I after Cayley retraction."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    n = tiny_model.bank.n_l
    for i in range(min(n, 4)):
        assert verify_stiefel(tiny_model.bank.W_l.data[i], tol=1e-4), \
            f"W_l[{i}] violates Stiefel constraint after train step"


def test_w_p_satisfies_stiefel_after_step(tiny_model, tiny_cfg):
    """§1.10 / A10: W_p must satisfy W@W†=I after Cayley retraction."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    n_p = tiny_model.bank.n_p
    for i in range(min(n_p, 4)):
        assert verify_stiefel(tiny_model.bank.W_p.data[i], tol=1e-4), \
            f"W_p[{i}] violates Stiefel constraint after train step"


# ── _L_compress_accum cleared (§1.59) ────────────────────────────────────────

def test_l_compress_accum_cleared_after_step(tiny_model, tiny_cfg):
    """§1.59: _L_compress_accum must be None after train_step_v605 uses it."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    assert tiny_model._L_compress_accum is None, \
        "_L_compress_accum must be cleared after train step (stale accumulation bug)"


# ── W_ll cache cleared after opt_p.step() (H14) ──────────────────────────────

def test_w_ll_cache_cleared_after_step(tiny_model, tiny_cfg):
    """§3.5 H14: _W_ll_cache must be empty after opt_p.step() — stale cache → routing misalign."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    for layer in tiny_model.cfl_layers:
        assert len(layer._W_ll_cache) == 0, \
            f"_W_ll_cache must be cleared after opt_p.step(); found {len(layer._W_ll_cache)} entries"


# ── SI omega accumulates (displacement only) ──────────────────────────────────

def test_si_omega_accumulates_after_step(tiny_model, tiny_cfg):
    """§2.19 SI: omega must be nonzero for at least one param after update_omega."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    total_omega = sum(v.sum().item() for v in si.omega.values())
    assert total_omega > 0.0, "SI omega must be nonzero after at least one step"


# ── Trainable params changed after step ───────────────────────────────────────

def test_params_change_after_step(tiny_model, tiny_cfg):
    """§3.5: At least one trainable parameter must change after a train step."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    # Snapshot params before
    snap = {n: p.data.clone() for n, p in tiny_model.named_parameters()
            if p.requires_grad and p.data.is_floating_point()}
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    changed = any(
        not torch.allclose(snap[n], p.data, rtol=1e-6, atol=1e-8)
        for n, p in tiny_model.named_parameters()
        if n in snap
    )
    assert changed, "No trainable parameter changed after a full train step"


# ── Fisher-magnitude freeze called at step 0 (step % 100 == 0) ───────────────

def test_fisher_magnitude_freeze_called_at_step_0(tiny_model, tiny_cfg):
    """§1.63 C1: update_fisher_magnitude_freeze must run at step 0 (step % 100 == 0)."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    # _fisher_diag must be populated (update_fisher_magnitude_freeze reads it)
    assert hasattr(tiny_model, '_fisher_diag'), "_fisher_diag must exist after step 0"
    assert len(tiny_model._fisher_diag) > 0, "_fisher_diag must be populated after step 0"
