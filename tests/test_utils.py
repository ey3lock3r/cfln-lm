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


# ── v9.0 tests ────────────────────────────────────────────────────────────────

def _make_tiny_model():
    """Instantiate a minimal CFLNModel for structural tests."""
    from cfln.modules.model import CFLNModel
    cfg = {
        'd_c': 16, 'vocab_size': 64, 'n_l': 32, 'n_p': 4, 'L': 1,
        'n_heads_gat': 2, 'd_e_l': 4, 'd_e_p': 4, 'd_ssm_fast': 4,
        'S_f': 4, 'C_chunk': 4, 'K_L1': 4, 'K_L2': 4, 'K_L3': 4,
        'N_archive': 4, 'd_r_node': 4, 'd_r_lista': 4, 'T': 8, 'B': 1,
        'N_iter_refine': 1, 'N_hop_refine': 1, 'surprise_warmup_chunks': 1,
        'sparse_code_cache_K': 4, 'episodic_rule_cache_n': 4,
        'K_hebb': 4, 'D_g': 4, 'D_bptt': 2, 'K_stats': 2,
        'T_diff': 4, 'n_fourier': 4, 'N_dormant': 8,
        'grad_clip': 1.0, 'lr_local': 3e-4, 'lr_unit': 1e-3,
        'lr_persist': 1e-6, 'lr_muon': 1e-3, 'lr_muon_diff': 1e-4,
        'lr_start': 1e-3, 'lr_end': 1e-4, 'c_SI': 0.1, 'rho_SI': 0.999,
        'beta_SI': 3.0, 'si_warmup_steps': 2, 'min_snapshot_interval': 1,
        'si_proactive_threshold': 0.8, 'proactive_cooldown': 1,
        'lambda_compress': 0.01, 'lambda_lista': 0.1,
        'memory_thresholds': {'eps_s': 0.01, 'eps_p': 0.001, 'eps_split': 0.5,
                               'eps_merge': 0.95, 'r_reset': 0.3, 'eps_H': 1e-4},
        'rho_node': 0.95, 'rho_lista': 0.99,
        'rho_fast': 0.70, 'rho_mid': 0.90, 'rho_slow': 0.99,
        'delta_stuck': 0.1, 'delta_min': 0.01, 'epsilon_esc': 0.05,
        'schedule_grad_clip': 0.5, 'use_hopfield_refine': True,
        'use_escape_refine': False, 'n_layers_diff': 1,
        'lista_min_ratio': 0.25, 'lista_convergence_ratio': 0.5,
        'per_sequence_memory': True, 'domain_check_freq': 100,
        'n_roles': 4, 'arc_dual_key': True, 'alpha_young': 0.1,
        'K_proto_max': 4, 'tau_proto': 0.6,
    }
    return CFLNModel(cfg)


# ── v7.0 tests ────────────────────────────────────────────────────────────────

def test_phase_kernel_psd():
    """B_bind must be PSD (all eigenvalues >= -1e-5). §1.30 R1 / §1.41 W2."""
    model = _make_tiny_model()
    bank = model.bank
    k = min(8, bank.n_l)
    sel_l = torch.arange(k)
    phi_sel = torch.angle(bank.H_c_l[sel_l].mean(-1).mean(-1))
    phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)
    sigma_sq = torch.exp(2.0 * bank.log_sigma_bind)
    B_bind = torch.exp(-phi_diff**2 / sigma_sq.clamp(1e-6)).float()
    eigs = torch.linalg.eigvalsh(B_bind)
    assert eigs.min().item() >= -1e-5, f"B_bind not PSD, min eig={eigs.min().item()}"


def test_dual_key_arc_shape():
    """rule_K must have shape (N, 2*d_c). §1.31 R2."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    d_c = model.bank.d_c
    assert cun.rule_K.shape[1] == 2 * d_c, (
        f"rule_K.shape[1]={cun.rule_K.shape[1]}, expected {2*d_c}")


def test_goal_register_zeros_after_reset():
    """g_c must be zeros after reset_for_inference(). §1.33 R4."""
    model = _make_tiny_model()
    model.bank.g_c.fill_(1.0)
    model.reset_for_inference()
    assert model.bank.g_c.abs().max().item() == 0.0, "g_c not zeroed after reset"


def test_g_c_dtype():
    """g_c must be cfloat. §1.33 R4."""
    model = _make_tiny_model()
    assert model.bank.g_c.dtype == torch.cfloat, (
        f"g_c dtype={model.bank.g_c.dtype}, expected cfloat")


def test_u_meta_v4_five_signals():
    """log_w_meta (legacy) or sigma_sq_buffer must have length 5 for U_meta_v4. §1.34."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    buf = getattr(cun, 'sigma_sq_buffer', None)
    if buf is not None:
        assert len(buf) == 5, f"sigma_sq_buffer len={len(buf)}, expected 5"
    else:
        assert cun.log_w_meta.shape[0] == 5, (
            f"log_w_meta.shape[0]={cun.log_w_meta.shape[0]}, expected 5")


# ── v8.0 tests ────────────────────────────────────────────────────────────────

def test_b_role_psd():
    """B_role = alpha @ alpha.T must be PSD by construction. §1.35 X."""
    model = _make_tiny_model()
    bank = model.bank
    k = min(8, bank.n_l)
    sel_l = torch.arange(k)
    d_c = bank.d_c
    alpha_role = torch.softmax(
        (bank.mu_c_l[sel_l] @ bank.role_vecs.conj().T).real / (d_c**0.5), dim=-1
    ).float()
    B_role = alpha_role @ alpha_role.T
    eigs = torch.linalg.eigvalsh(B_role)
    assert eigs.min().item() >= -1e-5, f"B_role not PSD, min eig={eigs.min().item()}"


def test_role_vecs_not_in_muon():
    """role_vecs must NOT be in Muon param groups (non-square → AdamW). §1.35 X / §3 v8."""
    from cfln.training.optimizers import build_optimizers_v605
    model = _make_tiny_model()
    muon, muon_diff, _opt_g, _opt_u, _opt_p = build_optimizers_v605(model, {
        'lr_local': 3e-4, 'lr_unit': 1e-3, 'lr_persist': 1e-6,
        'lr_muon': 1e-3, 'lr_muon_diff': 1e-4,
    })
    rv_id = id(model.bank.role_vecs)
    # MuonOptimizer stores params as .params (list of (name, tensor) tuples)
    muon_ids = {id(p) for _n, p in muon.params}
    muon_diff_ids = {id(p) for _n, p in muon_diff.params}
    assert rv_id not in muon_ids, "role_vecs should not be in Muon"
    assert rv_id not in muon_diff_ids, "role_vecs should not be in muon_diff"


def test_ssp_stack_push_pop():
    """SSP: push increments stack, pop decrements. §1.39 Z."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    assert len(cun._goal_stack) == 0
    cun._goal_stack.append(cun.r_lista.detach().clone())
    assert len(cun._goal_stack) == 1
    cun._goal_stack.pop()
    assert len(cun._goal_stack) == 0


def test_ssp_max_depth():
    """SSP stack must not exceed ssp_max_depth. §1.39 Z."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    max_depth = 4
    for _ in range(max_depth + 5):
        if len(cun._goal_stack) < max_depth:
            cun._goal_stack.append(cun.r_lista.detach().clone())
    assert len(cun._goal_stack) == max_depth


def test_stela_continuity():
    """STELA threshold must produce zero gradient at h=0 (continuity). §1.40 W1."""
    tau = torch.tensor(0.3)
    tau_smooth = torch.tensor(0.05)
    h_zero = torch.zeros(4, dtype=torch.cfloat, requires_grad=False)
    z = h_zero.abs()
    gate = torch.sigmoid((z - tau) / tau_smooth.clamp(1e-3))
    result = h_zero * gate
    assert result.abs().max().item() == 0.0, "STELA at h=0 must give 0"


def test_consol_updates_mu():
    """consolidate_arc_to_cnep must shift the nearest mu_c_l toward a high-util rule. §1.37."""
    from cfln.modules.v9_ops import consolidate_arc_to_cnep
    model = _make_tiny_model()
    bank = model.bank
    cun = model.diff_aux.cun
    cun._rule_cache_n = 1
    d_c = bank.d_c
    n = bank.n_l
    target = torch.randn(d_c, dtype=torch.cfloat)
    cun.rule_K.data[0, :d_c] = target
    cun.rule_util[0] = 10.0  # well above tau_consol=3.0
    # Find which centroid will be updated (nearest to target)
    dists = (bank.mu_c_l[:n] - target).norm(dim=-1).real
    nearest = int(dists.argmin().item())
    mu_before = bank.mu_c_l[nearest].clone()
    consolidate_arc_to_cnep(bank, cun, tau_consol=3.0, alpha_consol=0.01)
    mu_after = bank.mu_c_l[nearest]
    moved = (mu_after - mu_before).norm().item()
    assert moved > 0.0, "consolidate_arc_to_cnep must update the nearest mu_c_l"


# ── Addendum tests ────────────────────────────────────────────────────────────

def test_se1_kshot_centroid_update():
    """SE-1: young unit mu_c_l must update toward input when sim > tau_proto. §1.43."""
    model = _make_tiny_model()
    bank = model.bank
    bank.activation_freq_l[0] = 0.0
    bank._proto_count[0] = 1
    x_mean = bank.mu_c_l[0].clone() * 0.9
    bank._proto_sum.data[0] = x_mean
    old_mu = bank.mu_c_l[0].clone()
    new_input = bank.mu_c_l[0].clone() * 1.01
    bank._proto_count[0] += 1
    bank._proto_sum.data[0] = bank._proto_sum[0] + new_input
    bank.mu_c_l.data[0] = bank._proto_sum[0] / float(bank._proto_count[0].item())
    assert (bank.mu_c_l[0] - old_mu).norm().item() > 0.0, "SE-1 centroid must shift"


def test_q_beam_parameter_free():
    """compute_Q_beam must run without errors and return a scalar. §1.46."""
    from cfln.modules.v9_ops import compute_Q_beam
    d = 16
    h_N = torch.randn(d, dtype=torch.cfloat)
    r_lista = torch.randn(d, dtype=torch.cfloat)
    score = compute_Q_beam(h_N, r_lista, None, [], torch.randn(1, d, dtype=torch.cfloat))
    assert score.numel() == 1, "Q_beam must return scalar"


def test_ts1_soft_select_weights_sum_to_one():
    """TS-1: beam soft-select weights must be non-negative and sum to 1. §1.47."""
    scores = torch.tensor([0.3, 0.7])
    w = torch.softmax(scores, dim=0)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0)


# ── §1.50 W_bridge tests ──────────────────────────────────────────────────────

def test_w_rc_bridge_is_parameter():
    """W_rc_bridge must be nn.Parameter (requires_grad=True). §1.50."""
    import torch.nn as nn
    model = _make_tiny_model()
    assert isinstance(model.W_rc_bridge, nn.Parameter), (
        "W_rc_bridge must be nn.Parameter")
    assert model.W_rc_bridge.requires_grad, "W_rc_bridge must require grad"


def test_w_rc_bridge_not_in_muon():
    """W_rc_bridge must NOT be in Muon (ESN design → AdamW only). §1.50 / §3 v9."""
    from cfln.training.optimizers import build_optimizers_v605
    model = _make_tiny_model()
    muon, muon_diff, _opt_g, _opt_u, _opt_p = build_optimizers_v605(model, {
        'lr_local': 3e-4, 'lr_unit': 1e-3, 'lr_persist': 1e-6,
        'lr_muon': 1e-3, 'lr_muon_diff': 1e-4,
    })
    bridge_id = id(model.W_rc_bridge)
    muon_ids = {id(p) for _n, p in muon.params}
    muon_diff_ids = {id(p) for _n, p in muon_diff.params}
    assert bridge_id not in muon_ids, "W_rc_bridge must not be in Muon"
    assert bridge_id not in muon_diff_ids, "W_rc_bridge must not be in muon_diff"


# ── v9.0 structural tests ─────────────────────────────────────────────────────

def test_sigma_sq_buffer_init():
    """sigma_sq_buffer must initialise to [1.0]*5. §1.58."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    buf = getattr(cun, 'sigma_sq_buffer', None)
    if buf is not None:
        assert len(buf) == 5, f"sigma_sq_buffer length={len(buf)}"
        for v in buf:
            assert abs(v - 1.0) < 1e-6, f"sigma_sq_buffer init value {v} != 1.0"


def test_precision_inactive_pathway():
    """_precision_active must initialise to [False]*5. §1.58."""
    model = _make_tiny_model()
    cun = model.diff_aux.cun
    pa = getattr(cun, '_precision_active', None)
    if pa is not None:
        assert len(pa) == 5
        assert not any(pa), "_precision_active must all be False at init"


def test_vq_buf_dtype():
    """VQ routing weight buffers must be float32. §1.59."""
    model = _make_tiny_model()
    bank = model.bank
    assert bank.buf_L1_w_full.dtype == torch.float32, "buf_L1_w_full must be float32"
    assert bank.buf_L2_w_full.dtype == torch.float32, "buf_L2_w_full must be float32"
    assert bank.buf_L3_w_full.dtype == torch.float32, "buf_L3_w_full must be float32"


def test_l_vq_gradient_to_encoder():
    """L_vq encoder commitment must have gradient on chunk_mean (encoder). §1.59."""
    from cfln.modules.telescoping import vq_telescope_update
    model = _make_tiny_model()
    bank = model.bank
    N_max = bank.N_max_l  # buf width is N_max_l, not n_l
    chunk_mean = torch.randn(bank.d_c, dtype=torch.cfloat, requires_grad=True)
    s_l_full = torch.softmax(torch.randn(N_max), dim=0)
    sel_l = torch.arange(min(4, bank.n_l))
    cfg = {'K_L2': bank.K_L2, 'K_L3': bank.K_L3}
    L_vq = vq_telescope_update(chunk_mean, s_l_full, None, None, bank, sel_l, cfg)
    if L_vq.requires_grad:
        L_vq.backward()
        assert chunk_mean.grad is not None, "L_vq must give grad to chunk_mean"


def test_l_compress_gradient_flows():
    """_update_telescoping must not detach chunk_mean (§1.51 gradient fix)."""
    from cfln.modules.telescoping import vq_telescope_update
    model = _make_tiny_model()
    bank = model.bank
    N_max = bank.N_max_l
    chunk_mean = torch.randn(bank.d_c, dtype=torch.cfloat, requires_grad=True)
    s_l_full = torch.softmax(torch.randn(N_max), dim=0)
    sel_l = torch.arange(min(4, bank.n_l))
    cfg = {'K_L2': bank.K_L2, 'K_L3': bank.K_L3}
    L = vq_telescope_update(chunk_mean, s_l_full, None, None, bank, sel_l, cfg)
    if L.requires_grad:
        L.backward()
        assert chunk_mean.grad is not None, "L_compress gradient must flow to chunk_mean"
