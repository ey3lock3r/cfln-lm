"""Test suite for ComplexUnitaryDenoisingNet (CUN) and DiffusionAuxiliaryModule."""

import pytest
import torch
from cfln.modules.diffusion import ComplexUnitaryDenoisingNet
from cfln.modules.v9_ops import compute_Q_beam


class TestCUNInitialization:
    def test_sigma_sq_buffer(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        assert hasattr(cun, "sigma_sq_buffer")
        assert len(cun.sigma_sq_buffer) == 5
        assert all(v == 1.0 for v in cun.sigma_sq_buffer)

    def test_precision_active(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        assert hasattr(cun, "_precision_active")
        assert len(cun._precision_active) == 5
        assert all(v is False for v in cun._precision_active)

    def test_goal_stack_empty(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        assert isinstance(cun._goal_stack, list)
        assert len(cun._goal_stack) == 0

    def test_r_lista_buffer(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        assert cun.r_lista.dtype == torch.cfloat
        assert cun.r_lista.shape[0] == cun.d_r_lista

    def test_U1_U2_parameters(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        d_c = cun.d_c
        for pname in ["U1", "U2"]:
            p = getattr(cun, pname)
            assert isinstance(p, torch.nn.Parameter)
            assert p.dtype == torch.cfloat
            assert p.shape == (d_c, d_c)

    def test_log_blend_alpha(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        assert hasattr(cun, "log_blend_alpha")
        assert isinstance(cun.log_blend_alpha, torch.nn.Parameter)


class TestResetListaReservoir:
    def test_reset_zeros_r_lista(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        with torch.no_grad():
            cun.r_lista.copy_(torch.ones_like(cun.r_lista))
        cun.reset_lista_reservoir()
        assert torch.allclose(cun.r_lista, torch.zeros_like(cun.r_lista))

    def test_reset_clears_goal_stack(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        cun._goal_stack.append(torch.randn(cun.d_c, dtype=torch.cfloat))
        cun.reset_lista_reservoir()
        assert len(cun._goal_stack) == 0

    def test_reset_thinking_mode_false(self, tiny_model):
        cun = tiny_model.diff_aux.cun
        cun._in_thinking_mode = True
        cun.reset_lista_reservoir()
        assert cun._in_thinking_mode is False


class TestListaForward:
    def test_output_shape_returns_tuple(self, tiny_model, tiny_cfg):
        cun = tiny_model.diff_aux.cun
        B = tiny_cfg["B"]
        d_c = tiny_cfg["d_c"]
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        result = cun.lista_forward(x_c, bank=tiny_model.bank)
        assert isinstance(result, tuple)
        assert len(result) == 3
        x_ref, h_N, info = result
        assert h_N.shape == (B, d_c)
        assert h_N.dtype == torch.cfloat

    def test_no_nan(self, tiny_model, tiny_cfg):
        cun = tiny_model.diff_aux.cun
        B = tiny_cfg["B"]
        d_c = tiny_cfg["d_c"]
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        x_ref, h_N, info = cun.lista_forward(x_c, bank=tiny_model.bank)
        assert not torch.isnan(h_N).any()

    def test_Q_BEAM_score(self, tiny_model, tiny_cfg):
        cun = tiny_model.diff_aux.cun
        B = tiny_cfg["B"]
        d_c = tiny_cfg["d_c"]
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        x_ref, h_N, info = cun.lista_forward(x_c, bank=tiny_model.bank)
        assert isinstance(cun._last_Q_BEAM_score, (float, int))


class TestComputeQBeam:
    def test_returns_scalar(self, tiny_model, tiny_cfg):
        d_c = tiny_cfg["d_c"]
        d_r = tiny_cfg["d_r_lista"]
        h_N = torch.randn(d_c, dtype=torch.cfloat)
        r = torch.randn(d_r, dtype=torch.cfloat)
        x_c = torch.randn(1, d_c, dtype=torch.cfloat)
        cun = tiny_model.diff_aux.cun
        score = compute_Q_beam(h_N, r, None, [], x_c, log_w_beam=cun.log_w_beam)
        assert isinstance(score, torch.Tensor)

    def test_mdl_negative(self, tiny_cfg):
        d_c = tiny_cfg["d_c"]
        d_r = tiny_cfg["d_r_lista"]
        h_N = torch.randn(d_c, dtype=torch.cfloat)
        r = torch.randn(d_r, dtype=torch.cfloat)
        x_c = torch.randn(1, d_c, dtype=torch.cfloat)
        score = compute_Q_beam(h_N, r, None, [], x_c)
        assert float(score.item()) < 0

    def test_with_goal_proxy(self, tiny_cfg):
        d_c = tiny_cfg["d_c"]
        d_r = tiny_cfg["d_r_lista"]
        h_N = torch.randn(d_c, dtype=torch.cfloat)
        r = torch.randn(d_r, dtype=torch.cfloat)
        rg = torch.randn(d_r, dtype=torch.cfloat)
        x_c = torch.randn(1, d_c, dtype=torch.cfloat)
        score = compute_Q_beam(h_N, r, rg, [], x_c)
        assert float(score.item()) < 0


class TestDiffAuxIntegration:
    def test_cun_reference(self, tiny_model):
        assert hasattr(tiny_model, "diff_aux")
        assert isinstance(tiny_model.diff_aux.cun, ComplexUnitaryDenoisingNet)

    def test_cun_d_c_matches(self, tiny_model, tiny_cfg):
        cun = tiny_model.diff_aux.cun
        cfg_d_c = tiny_cfg["d_c"]
        assert cun.d_c == cfg_d_c

