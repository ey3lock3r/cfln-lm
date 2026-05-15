"""Smoke test: full pipeline on CPU with notebook SMOKE config.

Catches integration-level failures (circular module refs, shape mismatches
after unit spawns, checkpoint round-trips) that unit tests miss.
Runs in ~30s on CPU, no GPU required.
"""
import gc
import math
import random
import torch
from dataclasses import asdict

from cfln.config import CFLNConfig
from cfln.modules.model import CFLNModel
from cfln.modules.si import SynapticIntelligence
from cfln.modules.monitoring import DocumentStreamingContext
from cfln.modules.psc_loss import PSCLoss
from cfln.training.optimizers import build_optimizers_v605
from cfln.training.train_step import train_step_v605, psc_train_step
from cfln.utils import verify_stiefel
from cfln.modules.si import compute_domain_confidence


SMOKE_CFG = CFLNConfig(
    d_c=32, n_l=128, n_p=16, L=2,
    vocab_size=256, C_chunk=16,
    K_L1=8, K_L2=4, K_L3=4,
    d_ssm_fast=8, S_f=8,
    d_e_l=8, d_e_p=8,
    d_r_node=4, d_r_lista=16,
    k_l=8, k_l_min=4, k_l_max=8,
    n_heads_gat=2,
    K_hebb=4, K_sparse=8, N_rules=16,
    N_archive=16, N_iter=2, N_iter_refine=2,
    n_fourier=8, T_diff=10, D_g=4,
    T=32, B=2, stage=0,
    si_warmup_steps=2, min_snapshot_interval=1,
)


def _make_batches(n, cfg, seed=0):
    rng = random.Random(seed)
    return [
        {'input_ids': torch.tensor(
            [[rng.randint(1, cfg.vocab_size - 1) for _ in range(cfg.T)]
             for _ in range(cfg.B)], dtype=torch.long)}
        for _ in range(n)
    ]


def _build(cfg, expand_vocab=False):
    d = asdict(cfg)
    model = CFLNModel(d).to('cpu')
    model.train()
    if expand_vocab:
        model.expand_vocabulary()
    si = SynapticIntelligence(c_SI=cfg.c_SI)
    doc_ctx = DocumentStreamingContext(model)
    opts = build_optimizers_v605(model, d)
    warmup = d.get('warmup_steps', 500)
    muon, muon_diff, opt_g, opt_u, opt_p = opts
    schedulers = {
        'sched_g': torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: min(1.0, (s + 1) / warmup)),
        'sched_u': torch.optim.lr_scheduler.LambdaLR(opt_u, lambda s: min(1.0, (s + 1) / warmup)),
    }
    return model, opts, si, doc_ctx, schedulers


# ── model.to('cpu') does not recurse ─────────────────────────────────────────

def test_model_to_does_not_recurse():
    """CFLNModel must have no circular nn.Module references (caused RecursionError on .to())."""
    model = CFLNModel(asdict(SMOKE_CFG))
    model.to('cpu')  # raises RecursionError if si._model_ref is a registered child


# ── forward pass: shape and no NaN ───────────────────────────────────────────

def test_smoke_forward():
    model, *_ = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]['input_ids']
    with torch.no_grad():
        logits, _, _ = model(batch, training=False)
    assert logits.shape == (SMOKE_CFG.B, SMOKE_CFG.T, SMOKE_CFG.vocab_size)
    assert not torch.isnan(logits).any(), "NaN in smoke forward pass"


# ── one train step: finite loss ───────────────────────────────────────────────

def test_smoke_train_step():
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]
    info = train_step_v605(
        batch, model, opts, si,
        phase='pretrain', step=0, total_steps=10,
        cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx,
    )
    assert 'L_task' in info
    assert not math.isnan(info['L_task']), f"NaN loss at smoke train step: {info['L_task']}"


# ── Stiefel constraint holds after step ──────────────────────────────────────

def test_smoke_stiefel():
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]
    train_step_v605(batch, model, opts, si, phase='pretrain', step=0,
                    total_steps=10, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx)
    assert verify_stiefel(model.bank.W_l[0], tol=1e-4), "W_l Stiefel violated after smoke step"


# ── 10-step loop: loss stays finite ──────────────────────────────────────────

def test_smoke_10_steps():
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG)
    batches = _make_batches(10, SMOKE_CFG)
    cfg_dict = asdict(SMOKE_CFG)
    for step, batch in enumerate(batches):
        info = train_step_v605(batch, model, opts, si, phase='pretrain',
                               step=step, total_steps=10, cfg=cfg_dict, doc_ctx=doc_ctx)
        assert not math.isnan(info.get('L_task', float('nan'))), \
            f"NaN loss at step {step}"
    gc.collect()


# ── checkpoint round-trip ─────────────────────────────────────────────────────

def test_smoke_checkpoint_roundtrip(tmp_path):

    model, opts, si, doc_ctx, schedulers = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]
    train_step_v605(batch, model, opts, si, phase='pretrain', step=0,
                    total_steps=5, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx)

    ckpt = tmp_path / 'smoke.pt'
    torch.save({
        'model': model.state_dict(),
        'step': 1, 'stage': 'smoke',
    }, ckpt)

    model2, opts2, si2, doc_ctx2, _ = _build(SMOKE_CFG)
    # _x_c_prev_bank is resized on first forward; prime it before loading state.
    with torch.no_grad():
        model2(batch['input_ids'], training=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model2.load_state_dict(state['model'])

    with torch.no_grad():
        logits, _, _ = model2(batch['input_ids'], training=False)
    assert not torch.isnan(logits).any(), "NaN after checkpoint round-trip"


# ── MDLM: no double-backward across steps ────────────────────────────────────

def test_smoke_mdlm_no_double_backward():
    """MDLM (p_mask>0) must not cause 'backward through freed graph' on step 2+.

    Catches cross-step live-graph leaks: _null_aux_loss, _last_L_bridge,
    _last_beam_diversity, _L_compress_accum holding freed grad_fn nodes that
    get woven into the next step's L_pass1 before the new forward overwrites them.
    """
    from dataclasses import replace
    cfg = replace(SMOKE_CFG, p_mask=0.15)
    model, opts, si, doc_ctx, _ = _build(cfg)
    cfg_dict = asdict(cfg)
    cfg_dict['mask_token_id'] = 1
    batches = _make_batches(5, cfg)
    for step, batch in enumerate(batches):
        try:
            info = train_step_v605(batch, model, opts, si, phase='pretrain',
                                   step=step, total_steps=10, cfg=cfg_dict, doc_ctx=doc_ctx)
        except RuntimeError as e:
            if 'freed' in str(e) or 'second time' in str(e):
                raise AssertionError(
                    f"Double-backward at step {step}: {e}\n"
                    "A live graph node from a previous step leaked into L_pass1."
                ) from e
            raise
        assert not math.isnan(info.get('L_task', float('nan'))), \
            f"NaN loss at MDLM step {step}"


# ── Fix D: compute_domain_confidence NaN guard ───────────────────────────────

def test_domain_confidence_nan_guard():
    """Fix D: NaN inputs must return 0.0, not silently propagate NaN."""
    assert compute_domain_confidence(float('nan'), 1.0) == 0.0
    assert compute_domain_confidence(1.0, float('nan')) == 0.0
    assert compute_domain_confidence(float('nan'), float('nan')) == 0.0
    c = compute_domain_confidence(5.0, 1.0)
    assert 0.0 <= c <= 1.0, f"Valid inputs must return value in [0,1], got {c}"


# ── Fix E: begin_document resets EMA baselines ───────────────────────────────

def test_begin_document_resets_ema_baselines():
    """Fix E: begin_document must reset EMA baselines to prevent cross-document bleed."""
    model = CFLNModel(asdict(SMOKE_CFG))
    model.bank._e_min_ema.fill_(999.0)
    model.bank._h_route_ema.fill_(999.0)
    model.bank._ema_delta_bank.fill_(999.0)
    ctx = DocumentStreamingContext(model)
    ctx.begin_document()
    assert (model.bank._e_min_ema == 1.0).all(), "_e_min_ema not reset"
    assert (model.bank._h_route_ema == 1.0).all(), "_h_route_ema not reset"
    assert (model.bank._ema_delta_bank <= 1e-5).all(), "_ema_delta_bank not reset"


# ── PSC train step: finite loss, no double-backward ──────────────────────────

def test_smoke_psc_train_step():
    """Stage 1 psc_train_step must return finite loss and not crash on step 2+."""
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG, expand_vocab=True)
    psc_loss_fn = PSCLoss(d_c=SMOKE_CFG.d_c, d_r_lista=SMOKE_CFG.d_r_lista)
    cfg_dict = asdict(SMOKE_CFG)
    # vocab_size grows by 6 after expand_vocabulary(); batches must stay in range
    cfg_dict['vocab_size'] = model.sti_head.W_vocab.weight.shape[0]
    batches = _make_batches(5, SMOKE_CFG)
    for step, batch in enumerate(batches):
        info = psc_train_step(
            batch, model, psc_loss_fn, opts, si,
            phase='psc', step=step, total_steps=10,
            cfg=cfg_dict, doc_ctx=doc_ctx,
        )
        assert 'L_task' in info
        assert not math.isnan(info['L_task']), f"NaN L_task at PSC step {step}"


# ── Full notebook checkpoint round-trip (all optimizer + SI state) ────────────

def _save_checkpoint(model, opts, si, schedulers, step, stage, path):
    muon, muon_diff, opt_g, opt_u, opt_p = opts
    ckpt = {
        'step': step, 'stage': stage,
        'model_state':     model.state_dict(),
        'muon_state':      muon.state_dict(),
        'muon_diff_state': muon_diff.state_dict(),
        'opt_g_state':     opt_g.state_dict(),
        'opt_u_state':     opt_u.state_dict(),
        'opt_p_state':     opt_p.state_dict(),
        'sched_g_state':   schedulers['sched_g'].state_dict(),
        'sched_u_state':   schedulers['sched_u'].state_dict(),
        'si_omega':        {n: p.cpu().clone() for n, p in si.omega.items()},
        'si_theta_star':   {n: p.cpu().clone() for n, p in si.theta_star.items()},
        'titans_M':        model.encoder.titans.M.cpu().clone(),
        'bank_u_epi_mu':   model.bank._u_epi_mu.detach().cpu().clone(),
        'bank_u_epi_var':  model.bank._u_epi_var.detach().cpu().clone(),
        'bank_emin_mean':  torch.tensor(model.bank._Emin_mean),
        'bank_emin_var':   torch.tensor(model.bank._Emin_var),
        'bank_emin_n':     model.bank._Emin_n,
    }
    torch.save(ckpt, path)


def _load_checkpoint(path, model, opts, si, schedulers, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    muon, muon_diff, opt_g, opt_u, opt_p = opts
    muon.load_state_dict(ckpt['muon_state'])
    muon_diff.load_state_dict(ckpt['muon_diff_state'])
    opt_g.load_state_dict(ckpt['opt_g_state'])
    opt_u.load_state_dict(ckpt['opt_u_state'])
    opt_p.load_state_dict(ckpt['opt_p_state'])
    schedulers['sched_g'].load_state_dict(ckpt['sched_g_state'])
    schedulers['sched_u'].load_state_dict(ckpt['sched_u_state'])
    for n, p in ckpt['si_omega'].items():      si.omega[n]      = p.to(device)
    for n, p in ckpt['si_theta_star'].items(): si.theta_star[n] = p.to(device)
    model.encoder.titans.M.copy_(ckpt['titans_M'].to(device))
    model.bank._u_epi_mu.fill_(float(ckpt['bank_u_epi_mu']))
    model.bank._u_epi_var.fill_(float(ckpt['bank_u_epi_var']))
    model.bank._Emin_mean = float(ckpt['bank_emin_mean'])
    model.bank._Emin_var  = float(ckpt['bank_emin_var'])
    model.bank._Emin_n    = int(ckpt['bank_emin_n'])
    return ckpt['step'], ckpt['stage']


def test_smoke_full_checkpoint_roundtrip(tmp_path):
    """Full notebook checkpoint (model + all 5 opts + SI + session state) must survive round-trip."""
    model, opts, si, doc_ctx, schedulers = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]
    train_step_v605(batch, model, opts, si, phase='pretrain', step=0,
                    total_steps=5, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx)
    # Let SI accumulate omega so it's non-trivial to restore
    si.save_task_snapshot(si._get_named_params(model))

    ckpt = tmp_path / 'full.pt'
    _save_checkpoint(model, opts, si, schedulers, step=1, stage='smoke', path=ckpt)

    model2, opts2, si2, doc_ctx2, schedulers2 = _build(SMOKE_CFG)
    with torch.no_grad():
        model2(batch['input_ids'], training=False)
    step_back, stage_back = _load_checkpoint(ckpt, model2, opts2, si2, schedulers2)

    assert step_back == 1
    assert stage_back == 'smoke'

    # SI omega restored (at least one param should be non-zero after a step)
    assert len(si2.omega) > 0, "SI omega dict empty after load"

    # Model produces finite output after restore
    with torch.no_grad():
        logits, _, _ = model2(batch['input_ids'], training=False)
    assert not torch.isnan(logits).any(), "NaN after full checkpoint round-trip"

    # Training continues without crash
    info = train_step_v605(batch, model2, opts2, si2, phase='pretrain', step=2,
                           total_steps=5, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx2)
    assert not math.isnan(info['L_task']), "NaN loss on first step after checkpoint load"


# ── SI task snapshot survives stage transition ────────────────────────────────

def test_smoke_si_task_snapshot():
    """SI omega must be non-empty after save_task_snapshot and protect params."""
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG)
    cfg_dict = asdict(SMOKE_CFG)
    for step, batch in enumerate(_make_batches(3, SMOKE_CFG)):
        train_step_v605(batch, model, opts, si, phase='pretrain', step=step,
                        total_steps=10, cfg=cfg_dict, doc_ctx=doc_ctx)
    si.save_task_snapshot(si._get_named_params(model))
    assert len(si.omega) > 0, "omega empty after snapshot"
    assert len(si.theta_star) > 0, "theta_star empty after snapshot"
    # All omega values must be finite and non-negative
    for name, val in si.omega.items():
        assert torch.isfinite(val).all(), f"omega[{name}] has non-finite values"
        assert (val >= 0).all(), f"omega[{name}] has negative values"


# ── reset_for_inference: no NaN, training state unaffected ───────────────────

def test_smoke_reset_for_inference():
    """reset_for_inference must produce finite output and not corrupt training state."""
    model, opts, si, doc_ctx, _ = _build(SMOKE_CFG)
    batch = _make_batches(1, SMOKE_CFG)[0]
    train_step_v605(batch, model, opts, si, phase='pretrain', step=0,
                    total_steps=5, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx)

    model.eval()
    model.reset_for_inference()
    with torch.no_grad():
        logits, _, _ = model(batch['input_ids'], training=False)
    assert not torch.isnan(logits).any(), "NaN after reset_for_inference"

    # Training can resume after inference reset
    model.train()
    info = train_step_v605(batch, model, opts, si, phase='pretrain', step=1,
                           total_steps=5, cfg=asdict(SMOKE_CFG), doc_ctx=doc_ctx)
    assert not math.isnan(info['L_task']), "NaN loss after resuming from inference mode"


# ── BUG-1: VQ pointer wrap-around — archive stores slot indices, not raw counters ─

def test_vq_archive_slot_indices():
    """After K_L1 chunks wrap, _vq_ptrs must contain slot indices < K_L1."""
    from dataclasses import replace
    cfg = replace(SMOKE_CFG, K_L1=4, K_L2=2, K_L3=2)  # tiny buffer to force wrap quickly
    model, opts, si, doc_ctx, _ = _build(cfg)
    cfg_dict = asdict(cfg)
    # Force enough VQ writes to wrap _L1_ptr past K_L1
    model.bank._Emin_n = 100  # skip warmup so surprise_archive.add_vq fires
    model.bank._Emin_mean = 0.0
    model.bank._Emin_var = 0.001
    model.surprise_archive.W_warmup = 0  # skip warmup
    model.surprise_archive._chunk_count = 9999
    batches = _make_batches(8, cfg)
    for step, batch in enumerate(batches):
        train_step_v605(batch, model, opts, si, phase='pretrain',
                        step=step, total_steps=20, cfg=cfg_dict, doc_ctx=doc_ctx)
    # All stored pointers must be valid slot indices
    vq_ptrs = getattr(model.surprise_archive, '_vq_ptrs', [])
    K_L1 = cfg.K_L1
    for p in vq_ptrs:
        assert 0 <= p < K_L1, f"vq_ptr {p} is not a valid slot index (K_L1={K_L1})"
    # retrieve_vq must return a finite vector
    s_q = torch.zeros(model.bank.N_max_l, dtype=torch.float32)
    result = model.surprise_archive.retrieve_vq(s_q, model.bank)
    assert torch.isfinite(result).all(), "retrieve_vq returned non-finite values"


# ── BUG-2: MDLM forward must not contaminate VQ / Welford session state ──────────

def test_mdlm_state_isolation():
    """MDLM sub-forward must not add extra _L1_ptr increments beyond what Pass 1 writes.

    Strategy: prime two identical models to the same state, then run the same step
    on one with p_mask=0 (no MDLM) and one with p_mask=0.15 (MDLM fires). The final
    _L1_ptr must be identical — MDLM's increment is rolled back by the finally block.
    """
    from dataclasses import replace
    import copy
    base_cfg = SMOKE_CFG
    batches = _make_batches(4, base_cfg, seed=42)

    # Build and prime both models identically
    model_a, opts_a, si_a, doc_ctx_a, _ = _build(base_cfg)
    model_b, opts_b, si_b, doc_ctx_b, _ = _build(base_cfg)

    for step, batch in enumerate(batches[:3]):
        cfg_base = dict(asdict(base_cfg), mask_token_id=1)
        cfg_base['p_mask'] = 0.0
        train_step_v605(batch, model_a, opts_a, si_a, phase='pretrain',
                        step=step, total_steps=100, cfg=cfg_base, doc_ctx=doc_ctx_a)
        train_step_v605(batch, model_b, opts_b, si_b, phase='pretrain',
                        step=step, total_steps=100, cfg=cfg_base, doc_ctx=doc_ctx_b)

    # Run the trigger step: model_a without MDLM, model_b with MDLM
    cfg_no_mdlm   = dict(asdict(base_cfg), p_mask=0.0,  mask_token_id=1)
    cfg_with_mdlm = dict(asdict(base_cfg), p_mask=0.99, mask_token_id=1)  # p_mask=0.99 → almost certain MDLM
    trigger_batch = batches[3]

    train_step_v605(trigger_batch, model_a, opts_a, si_a, phase='pretrain',
                    step=3, total_steps=100, cfg=cfg_no_mdlm, doc_ctx=doc_ctx_a)
    train_step_v605(trigger_batch, model_b, opts_b, si_b, phase='pretrain',
                    step=3, total_steps=100, cfg=cfg_with_mdlm, doc_ctx=doc_ctx_b)

    assert model_a.bank._L1_ptr == model_b.bank._L1_ptr, (
        f"_L1_ptr diverged: no-MDLM={model_a.bank._L1_ptr}, with-MDLM={model_b.bank._L1_ptr}. "
        "MDLM sub-forward leaked an extra _L1_ptr increment."
    )
    assert model_a.surprise_archive._n_filled == model_b.surprise_archive._n_filled, (
        f"surprise_archive._n_filled diverged: no-MDLM={model_a.surprise_archive._n_filled}, "
        f"with-MDLM={model_b.surprise_archive._n_filled}"
    )


# ── BUG-3: opt_p gradient isolation — Pass 2 must not bleed into next step's Pass 1 ─

def test_opt_p_gradient_isolation():
    """mu_c_p.grad must be None at the start of Pass 1 each step (no Pass-2 residual)."""
    from dataclasses import replace
    # Use stage ≥ 1 (step >= total_steps//4) to enable diff_aux so Pass 2 runs
    cfg = replace(SMOKE_CFG)
    model, opts, si, doc_ctx, _ = _build(cfg)
    cfg_dict = asdict(cfg)
    model.diff_aux.enable()  # force Pass 2 active
    batches = _make_batches(4, cfg)
    for step, batch in enumerate(batches):
        # Zero grads manually as the train_step does; check state entering backward
        muon, muon_diff, opt_g, opt_u, opt_p = opts
        opt_g.zero_grad(); muon.zero_grad()
        # Verify no residual from previous step
        assert model.bank.mu_c_p.grad is None or model.bank.mu_c_p.grad.abs().max() == 0, \
            f"mu_c_p has residual gradient at step {step} before Pass 1 backward"
        train_step_v605(batch, model, opts, si, phase='pretrain',
                        step=step + 10,  # stage >= 1 to keep diff_aux enabled
                        total_steps=20, cfg=cfg_dict, doc_ctx=doc_ctx)


# ── BUG-4: Hebbian co-activation multi-partner coverage ──────────────────────────

def test_coactivation_multi_partner():
    """With k active units, each unit must record all k-1 co-activation partners."""
    from cfln.modules.coact import CoactivationRegister
    K_hebb = 16
    N = 32
    reg = CoactivationRegister(N_max_l=N, K_hebb=K_hebb)
    # 5 simultaneously active units: indices 0..4
    s_l = torch.zeros(1, N)
    s_l[0, :5] = 1.0  # units 0-4 all active above threshold
    reg.update(s_l, threshold=0.5)
    # Each unit should have 4 distinct partners stored
    for unit in range(5):
        stored = reg.coact_reg[unit]
        valid = stored[stored >= 0].tolist()
        expected_partners = set(range(5)) - {unit}
        assert set(valid) == expected_partners, \
            f"Unit {unit}: expected partners {expected_partners}, got {set(valid)}"


# ── BUG-6: Fisher-KL checkpoint persistence — name-keyed dict survives reload ────

def test_fisher_kl_survives_checkpoint(tmp_path):
    """_fisher_diag_named must be non-empty after checkpoint reload and L_KL must fire."""
    from dataclasses import replace
    cfg = replace(SMOKE_CFG, si_warmup_steps=0)
    model, opts, si, doc_ctx, _ = _build(cfg)
    cfg_dict = asdict(cfg)
    cfg_dict['beta_KL_warmup'] = 0  # make KL penalty active immediately
    cfg_dict['beta_KL'] = 0.1
    batches = _make_batches(3, cfg)
    for step, batch in enumerate(batches):
        train_step_v605(batch, model, opts, si, phase='pretrain',
                        step=step, total_steps=10, cfg=cfg_dict, doc_ctx=doc_ctx)
    assert hasattr(model, '_fisher_diag_named') and len(model._fisher_diag_named) > 0, \
        "_fisher_diag_named not populated after training steps"
    # Save and reload
    ckpt = tmp_path / 'fisher_test.pt'
    torch.save({'model': model.state_dict()}, ckpt)
    model2, opts2, si2, doc_ctx2, _ = _build(cfg)
    with torch.no_grad():
        model2(batches[0]['input_ids'], training=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model2.load_state_dict(state['model'])
    # Run a step; Fisher dict will be rebuilt from named_parameters
    cfg_dict2 = dict(cfg_dict)
    cfg_dict2['beta_KL_warmup'] = 0
    info = train_step_v605(batches[0], model2, opts2, si2, phase='pretrain',
                           step=1, total_steps=10, cfg=cfg_dict2, doc_ctx=doc_ctx2)
    # After step 1 the dict is being built; run step 2 to trigger KL (step > 0)
    train_step_v605(batches[1], model2, opts2, si2, phase='pretrain',
                    step=2, total_steps=10, cfg=cfg_dict2, doc_ctx=doc_ctx2)
    assert hasattr(model2, '_fisher_diag_named') and len(model2._fisher_diag_named) > 0, \
        "_fisher_diag_named empty after checkpoint reload + training step"
    # Verify keys are valid parameter names (not Python id() integers)
    for key in model2._fisher_diag_named:
        assert isinstance(key, str), f"_fisher_diag_named key is not a string: {key!r}"
