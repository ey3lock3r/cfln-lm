"""Unit tests for VQ-Telescope memory: §1.59 / OI-7 / OI-8 / FIX-5.

Uses a minimal mock bank so tests stay fast and isolated.
"""
import torch
import pytest
from cfln.modules.telescoping import vq_telescope_update, vq_telescope_retrieve


# ── Minimal mock bank ──────────────────────────────────────────────────────────

def _make_bank(K_L1=8, K_L2=4, K_L3=4, N=16, d_c=8, C_chunk=4):
    """Minimal struct with the fields vq_telescope_update/retrieve require."""
    class FakeBank:
        pass
    b = FakeBank()
    b.K_L1 = K_L1
    b.K_L2 = K_L2
    b.K_L3 = K_L3
    b.n_l  = N
    b.d_c  = d_c
    b.C_chunk = C_chunk
    b.N_max_l = N
    b.buf_L1_w_full = torch.zeros(K_L1, N, dtype=torch.float32)
    b.buf_L2_w_full = torch.zeros(K_L2, N, dtype=torch.float32)
    b.buf_L3_w_full = torch.zeros(K_L3, N, dtype=torch.float32)
    b.buf_L1_ids    = torch.zeros(K_L1, C_chunk, dtype=torch.int32)
    b.mu_c_l        = torch.randn(N, d_c, dtype=torch.cfloat)
    b._L1_ptr       = 0
    b._last_s_l_full = None
    b._Emin_mean    = 0.0
    b._Emin_var     = 0.0
    b._Emin_n       = 0
    b._last_E_min_raw = 0.0
    return b


def _cfg(K_L2=4, K_L3=4):
    return {'K_L2': K_L2, 'K_L3': K_L3}


# ── L1 write tests ────────────────────────────────────────────────────────────

def test_l1_buf_written_at_ptr():
    """§1.59: buf_L1_w_full[ptr] must equal s_l_full after update."""
    bank = _make_bank()
    cfg = _cfg()
    chunk_mean = torch.randn(8, dtype=torch.cfloat)
    s_l_full   = torch.rand(16)
    sel_l      = torch.arange(4)

    vq_telescope_update(chunk_mean, s_l_full, 0.0, None, bank, sel_l, cfg)

    torch.testing.assert_close(bank.buf_L1_w_full[0], s_l_full.float(), rtol=0.0, atol=1e-6)


def test_l1_ptr_increments():
    """§1.59: _L1_ptr increments by 1 per call."""
    bank = _make_bank()
    cfg = _cfg()
    for i in range(3):
        vq_telescope_update(
            torch.randn(8, dtype=torch.cfloat),
            torch.rand(16), 0.0, None, bank,
            torch.arange(4), cfg,
        )
    assert bank._L1_ptr == 3


def test_l1_wraps_around_circular():
    """§1.59: Circular buffer — ptr % K_L1 wraps at capacity."""
    K_L1 = 4
    bank = _make_bank(K_L1=K_L1)
    cfg = _cfg(K_L2=2, K_L3=2)
    sentinel = torch.ones(16) * 99.0
    # Fill buffer and then write one more
    for i in range(K_L1):
        vq_telescope_update(
            torch.randn(8, dtype=torch.cfloat),
            torch.rand(16), 0.0, None, bank, torch.arange(4), cfg,
        )
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        sentinel, 0.0, None, bank, torch.arange(4), cfg,
    )
    # slot 0 should have been overwritten
    torch.testing.assert_close(bank.buf_L1_w_full[0], sentinel.float(), rtol=0.0, atol=1e-6)


def test_chunk_token_ids_written():
    """§1.36 Y1: chunk_token_ids written to buf_L1_ids at correct slot."""
    bank = _make_bank(C_chunk=4)
    cfg = _cfg()
    ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        torch.rand(16), 0.0, ids, bank, torch.arange(4), cfg,
    )
    torch.testing.assert_close(bank.buf_L1_ids[0], ids, rtol=0, atol=0)


# ── last_s_l_full cache (OI-7) ────────────────────────────────────────────────

def test_last_s_l_full_cached_after_update():
    """OI-7: bank._last_s_l_full must be set to s_l_full.detach() after update."""
    bank = _make_bank()
    cfg = _cfg()
    s_l_full = torch.rand(16)
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        s_l_full, 0.0, None, bank, torch.arange(4), cfg,
    )
    assert bank._last_s_l_full is not None
    torch.testing.assert_close(bank._last_s_l_full, s_l_full, rtol=0.0, atol=1e-6)


def test_last_s_l_full_detached():
    """OI-7: _last_s_l_full must be detached (no grad)."""
    bank = _make_bank()
    cfg = _cfg()
    s_l_full = torch.rand(16, requires_grad=True)
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        s_l_full, 0.0, None, bank, torch.arange(4), cfg,
    )
    assert not bank._last_s_l_full.requires_grad


# ── L_vq loss (OI-8 / §1.59) ─────────────────────────────────────────────────

def test_l_vq_is_scalar_tensor():
    """§1.59: L_vq must be a scalar tensor."""
    bank = _make_bank()
    cfg = _cfg()
    L_vq = vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        torch.rand(16), 0.0, None, bank, torch.arange(4), cfg,
    )
    assert L_vq.ndim == 0, "L_vq must be a scalar"


def test_l_vq_gradient_flows_to_chunk_mean():
    """§1.59: Gradient of L_vq must flow to chunk_mean (encoder), not mu_c_l."""
    bank = _make_bank()
    cfg = _cfg()
    chunk_mean = torch.randn(8, dtype=torch.cfloat, requires_grad=True)
    s_l_full   = torch.rand(16)
    sel_l      = torch.arange(4)
    L_vq = vq_telescope_update(chunk_mean, s_l_full, 0.0, None, bank, sel_l, cfg)
    L_vq.backward()
    assert chunk_mean.grad is not None, "L_vq gradient must reach chunk_mean"
    assert not torch.isnan(chunk_mean.grad).any()


def test_l_vq_zero_for_empty_sel_l():
    """§1.59: L_vq = 0 when sel_l is empty (no selected units)."""
    bank = _make_bank()
    cfg = _cfg()
    L_vq = vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        torch.rand(16), 0.0, None,
        bank, torch.zeros(0, dtype=torch.long), cfg,
    )
    assert L_vq.item() == 0.0, "L_vq must be 0 when sel_l is empty"


def test_l_vq_codebook_detached():
    """§1.59: mu_c_l (codebook) must be detached in L_vq — gradient must NOT flow to it."""
    bank = _make_bank()
    bank.mu_c_l = torch.randn(16, 8, dtype=torch.cfloat, requires_grad=True)
    cfg = _cfg()
    chunk_mean = torch.randn(8, dtype=torch.cfloat, requires_grad=True)
    sel_l = torch.arange(4)
    L_vq = vq_telescope_update(chunk_mean, torch.rand(16), 0.0, None, bank, sel_l, cfg)
    L_vq.backward()
    assert bank.mu_c_l.grad is None, "Codebook (mu_c_l) must have no gradient through L_vq"


# ── L2 aggregation (§1.59) ────────────────────────────────────────────────────

def test_l2_updated_after_k_l2_chunks():
    """§1.59: buf_L2_w_full[0] must equal mean of first K_L2 L1 entries after K_L2 writes."""
    K_L2 = 4
    bank = _make_bank(K_L1=16, K_L2=K_L2, K_L3=4)
    cfg = _cfg(K_L2=K_L2, K_L3=4)
    stored = []
    for _ in range(K_L2):
        s = torch.rand(16)
        stored.append(s.clone())
        vq_telescope_update(
            torch.randn(8, dtype=torch.cfloat),
            s, 0.0, None, bank, torch.arange(4), cfg,
        )
    expected_l2 = torch.stack(stored).mean(0)
    torch.testing.assert_close(bank.buf_L2_w_full[0], expected_l2, rtol=0.0, atol=1e-5)


# ── Retrieval (§1.59 OI-8) ───────────────────────────────────────────────────

def test_retrieve_returns_zeros_before_any_write():
    """§1.59: Retrieval before any write must return zero vectors."""
    bank = _make_bank()
    query = torch.rand(16)
    r_L1, r_L2, r_L3 = vq_telescope_retrieve(query, bank)
    assert r_L1.abs().sum().item() == 0.0
    assert r_L2.abs().sum().item() == 0.0
    assert r_L3.abs().sum().item() == 0.0


def test_retrieve_nonzero_after_write():
    """§1.59 OI-8: After at least one write, r_L1 must be non-zero."""
    bank = _make_bank()
    cfg = _cfg()
    s_l_full = torch.rand(16)
    # make sure some centroids have nonzero weight under the stored s_l_full
    bank.mu_c_l = torch.randn(16, 8, dtype=torch.cfloat)
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        s_l_full, 0.0, None, bank, torch.arange(4), cfg,
    )
    r_L1, _, _ = vq_telescope_retrieve(s_l_full, bank)
    assert r_L1.abs().sum().item() > 0.0, "r_L1 must be non-zero after a write"


def test_retrieve_with_return_ids():
    """§1.59: return_ids=True must return a 4-tuple with token ids at index 3."""
    bank = _make_bank(C_chunk=4)
    cfg = _cfg()
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    vq_telescope_update(
        torch.randn(8, dtype=torch.cfloat),
        torch.rand(16), 0.0, ids, bank, torch.arange(4), cfg,
    )
    result = vq_telescope_retrieve(torch.rand(16), bank, return_ids=True)
    assert len(result) == 4, "return_ids=True must return 4-tuple"
    assert result[3] is not None, "Token ids must be returned"
