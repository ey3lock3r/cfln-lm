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
from cfln.training.optimizers import build_optimizers_v605
from cfln.training.train_step import train_step_v605
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
)


def _make_batches(n, cfg, seed=0):
    rng = random.Random(seed)
    return [
        {'input_ids': torch.tensor(
            [[rng.randint(1, cfg.vocab_size - 1) for _ in range(cfg.T)]
             for _ in range(cfg.B)], dtype=torch.long)}
        for _ in range(n)
    ]


def _build(cfg):
    d = asdict(cfg)
    model = CFLNModel(d).to('cpu')
    model.train()
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
