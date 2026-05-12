"""E2E tests for SI continual learning — omega accumulation, alpha_freeze, exemplar reactivation.

Tests use tiny_model / tiny_cfg from conftest.py.
"""
import torch
import pytest
from cfln.training.train_step import train_step_v605
from cfln.training.optimizers import build_optimizers_v605
from cfln.modules.si import SynapticIntelligence


def _make_batch(tiny_cfg):
    T, B = tiny_cfg['T'], tiny_cfg['B']
    return {'input_ids': torch.randint(0, tiny_cfg['vocab_size'], (B, T))}


def _build_si(tiny_cfg):
    return SynapticIntelligence(
        c_SI=tiny_cfg['c_SI'],
        rho_SI=tiny_cfg['rho_SI'],
        beta_SI=tiny_cfg['beta_SI'],
    )


# ── SI omega — displacement-only accumulation ─────────────────────────────────

def test_si_omega_uses_displacement_not_velocity(tiny_model, tiny_cfg):
    """§2.19 SI: update_omega uses (p - prev_p)² (displacement), not gradient²."""
    si = _build_si(tiny_cfg)
    si_params = si._get_named_params(tiny_model)
    # Prime omega entries to zero
    for k in si_params:
        si.omega[k] = torch.zeros_like(si_params[k].data, dtype=torch.float32)
    prev_params = {k: p.data.clone().detach() for k, p in si_params.items()}
    # Perturb params so displacement is nonzero
    with torch.no_grad():
        for p in si_params.values():
            p.data += torch.randn_like(p.data) * 0.1
    si_params2 = si._get_named_params(tiny_model)
    si.update_omega(si_params2, prev_params)
    # At least one omega entry must now be nonzero
    total = sum(v.sum().item() for v in si.omega.values())
    assert total > 0.0, "SI omega must be nonzero after displacement update"


def test_si_omega_nonneg(tiny_model, tiny_cfg):
    """§2.19 SI: omega values must be non-negative (squared displacement)."""
    si = _build_si(tiny_cfg)
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    for name, om in si.omega.items():
        assert (om >= 0).all(), f"SI omega[{name}] has negative values"


def test_si_loss_zero_before_snapshot(tiny_model, tiny_cfg):
    """§2.19 SI: SI loss must be 0 before first snapshot (si.active=False)."""
    si = _build_si(tiny_cfg)
    si_params = si._get_named_params(tiny_model)
    loss = si.compute_loss(si_params)
    assert float(loss) == 0.0, "SI loss must be 0 before first task snapshot"


def test_si_loss_nonzero_after_snapshot_and_update(tiny_model, tiny_cfg):
    """§2.19 SI: SI loss must be > 0 after snapshot + param update."""
    si = _build_si(tiny_cfg)
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si_params = si._get_named_params(tiny_model)
    # Manually set omega to nonzero so loss fires
    for k in si_params:
        si.omega[k] = torch.ones_like(si_params[k].data, dtype=torch.float32)
    si.save_task_snapshot(si_params)
    # Perturb params so displacement is nonzero
    for p in tiny_model.parameters():
        if p.requires_grad:
            p.data += torch.randn_like(p.data) * 0.01
    si_params2 = si._get_named_params(tiny_model)
    loss = si.compute_loss(si_params2)
    assert float(loss.detach()) > 0.0, "SI loss must be > 0 after param displacement from snapshot"


# ── alpha_freeze is a float ───────────────────────────────────────────────────

def test_alpha_freeze_is_float(tiny_model, tiny_cfg):
    """§2.19 B17: bank.alpha_freeze must be a Python float (not tensor)."""
    assert isinstance(tiny_model.bank.alpha_freeze, float), \
        f"alpha_freeze must be float, got {type(tiny_model.bank.alpha_freeze)}"


def test_alpha_freeze_in_unit_interval(tiny_model, tiny_cfg):
    """§2.19: alpha_freeze must be in [0, 1]."""
    opts = build_optimizers_v605(tiny_model, tiny_cfg)
    si = _build_si(tiny_cfg)
    batch = _make_batch(tiny_cfg)
    train_step_v605(batch, tiny_model, opts, si, phase=0, step=0,
                    total_steps=100, cfg=tiny_cfg)
    af = tiny_model.bank.alpha_freeze
    assert 0.0 <= af <= 1.0, f"alpha_freeze={af} out of [0,1]"


# ── Exemplar reactivation ─────────────────────────────────────────────────────

def test_dormancy_buf_exists(tiny_model):
    """§2.21: model must have dormancy_buf attribute for exemplar reactivation."""
    assert hasattr(tiny_model, 'dormancy_buf'), \
        "model must have dormancy_buf for exemplar reactivation"


def test_exemplar_reactivation_restores_mu(tiny_model, tiny_cfg):
    """§2.21: Dormancy add_from_history + reactivation must restore mu_c_l for new slot."""
    dormancy = tiny_model.dormancy_buf
    dyn = tiny_model.dyn
    bank = tiny_model.bank
    # Save unit 0 as dormancy exemplar
    saved = dormancy.add_from_history(bank, unit_idx=0)
    if not saved:
        pytest.skip("No free dormancy slot")
    slot = (dormancy._next_slot - 1) % dormancy.capacity
    mu_saved = dormancy.centroids[slot].clone()
    x_rep = torch.randn(1, tiny_cfg['d_c'], dtype=torch.cfloat)
    from cfln.training.train_step import _reactivate_from_exemplars
    ni = _reactivate_from_exemplars(slot, dormancy, bank, dyn, x_rep)
    if ni < 0:
        pytest.skip("Reactivation returned -1 (N_max reached)")
    # mu_c_l for new slot must match the saved centroid
    torch.testing.assert_close(bank.mu_c_l.data[ni], mu_saved, rtol=0.0, atol=1e-5)


# ── SI snapshot ───────────────────────────────────────────────────────────────

def test_si_save_task_snapshot_sets_active(tiny_model, tiny_cfg):
    """§2.19: save_task_snapshot must set si.active=True."""
    si = _build_si(tiny_cfg)
    assert not si.active
    si_params = si._get_named_params(tiny_model)
    si.save_task_snapshot(si_params)
    assert si.active, "si.active must be True after save_task_snapshot"


def test_si_snapshot_stores_theta_star(tiny_model, tiny_cfg):
    """§2.19: save_task_snapshot must populate theta_star for all protected params."""
    si = _build_si(tiny_cfg)
    si_params = si._get_named_params(tiny_model)
    si.save_task_snapshot(si_params)
    for k in si_params:
        assert k in si.theta_star, f"theta_star missing key {k} after snapshot"
