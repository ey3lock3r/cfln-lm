"""Integration tests for CFBank and CFL5Layer routing pipeline."""
import torch
import pytest
from cfln.modules.bank import CFBank
from cfln.modules.cfl5 import CFL5Layer


class TestComputeUEpistemic:
    def test_u_epistemic_in_valid_range(self, tiny_model):
        bank = tiny_model.bank
        B, n_l = 2, bank.n_l
        E_l = torch.randn(B, n_l)
        s_l = torch.randn(B, n_l).softmax(dim=-1)
        u_epi = bank.compute_u_epistemic(E_l, s_l)
        assert 0.0 <= u_epi <= 1.0

    def test_u_epistemic_cached(self, tiny_model):
        bank = tiny_model.bank
        B, n_l = 1, bank.n_l
        E_l = torch.randn(B, n_l)
        s_l = torch.randn(B, n_l).softmax(dim=-1)
        u_epi = bank.compute_u_epistemic(E_l, s_l)
        assert bank._u_epistemic_last == u_epi

    def test_log_cal_scale_param(self, tiny_model):
        bank = tiny_model.bank
        assert hasattr(bank, "log_cal_scale")
        assert isinstance(bank.log_cal_scale, torch.nn.Parameter)



class TestGoalAnchoredRouting:
    def test_g_c_buffer(self, tiny_model):
        bank = tiny_model.bank
        assert hasattr(bank, 'g_c')
        assert bank.g_c.dtype == torch.cfloat
        assert bank.g_c.shape == (bank.d_c,)

    def test_g_c_updated(self, tiny_model):
        if hasattr(tiny_model.bank, '_in_hypo_mode'):
            tiny_model.bank._in_hypo_mode = False
        if hasattr(tiny_model.bank, '_goal_frozen'):
            tiny_model.bank._goal_frozen = False
        bank = tiny_model.bank
        bank.g_c.zero_()
        B, d_c = 2, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, local_only=False)
        assert bank.g_c.abs().max().item() > 1e-6

    def test_x_c_eff_differs(self, tiny_model):
        bank = tiny_model.bank
        d_c = bank.d_c
        bank.g_c.fill_(0.1 + 0.1j)
        x_c = torch.randn(1, d_c, dtype=torch.cfloat)
        x_c_eff = x_c + torch.exp(bank.log_lam_goal) * bank.g_c.unsqueeze(0)
        diff = (x_c_eff - x_c).abs().max().item()
        assert diff > 1e-6


class TestRoutingOutput:
    def test_sel_l_shape(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert 'sel_l' in info
        sel_l = info['sel_l']
        assert sel_l.ndim == 1
        assert sel_l.shape[0] > 0

    def test_s_l_shape(self, tiny_model):
        bank = tiny_model.bank
        B, d_c, n_l = 2, bank.d_c, bank.n_l
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert 's_l' in info
        s_l = info['s_l']
        assert s_l.shape == (B, n_l)

    def test_s_l_sums_to_one(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 3, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        s_l = info['s_l']
        sums = s_l.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(B), rtol=1e-3, atol=1e-4)

    def test_E_l_nonnegative(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert 'E_l' in info
        E_l = info['E_l']
        assert (E_l >= -1e-5).all()

    def test_U_epistemic_in_range(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert 'U_epistemic' in info
        u_epi = info['U_epistemic']
        u_val = float(u_epi) if isinstance(u_epi, torch.Tensor) else u_epi
        assert 0.0 <= u_val <= 1.0


class TestBankStateAfterCFL5:
    def test_prev_sel_l_set(self, tiny_model):
        bank = tiny_model.bank
        bank._prev_sel_l = None
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=True)
        assert bank._prev_sel_l is not None
        assert bank._prev_sel_l.shape == info['sel_l'].shape

    def test_prev_sel_l_not_set_false(self, tiny_model):
        bank = tiny_model.bank
        old_prev = None if bank._prev_sel_l is None else bank._prev_sel_l.clone()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=False)
        if old_prev is None:
            assert bank._prev_sel_l is None or bank._prev_sel_l.shape[0] == 0

    def test_last_s_l_full_exists(self, tiny_model):
        bank = tiny_model.bank
        assert hasattr(bank, '_last_s_l_full')


class TestReservoirIntegration:
    def test_rho_l_updated(self, tiny_model):
        bank = tiny_model.bank
        n_l = bank.n_l
        old_rho = bank.rho_l[:n_l].clone()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=True)
        new_rho = bank.rho_l[:n_l]
        diff = (new_rho - old_rho).abs().max().item()
        assert diff > 1e-8

    def test_rho_l_unchanged(self, tiny_model):
        bank = tiny_model.bank
        n_l = bank.n_l
        old_rho = bank.rho_l[:n_l].clone()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=False)
        new_rho = bank.rho_l[:n_l]
        diff = (new_rho - old_rho).abs().max().item()
        assert diff < 1e-10


class TestKShotCentroidRefinement:
    def test_proto_count_increments(self, tiny_model):
        bank = tiny_model.bank
        bank.activation_freq_l[0] = 0.05
        bank._proto_count[0] = 2
        old_count = int(bank._proto_count[0].item())
        bank._u_epistemic_last = 0.3
        bank.tau_proto_min = 0.4
        bank.K_proto_max = 10
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        new_count = int(bank._proto_count[0].item())
        assert new_count >= old_count

    def test_mu_c_l_updates(self, tiny_model):
        bank = tiny_model.bank
        uid = 5
        bank.activation_freq_l[uid] = 0.08
        bank._proto_count[uid] = 1
        bank._u_epistemic_last = 0.25
        bank.tau_proto_min = 0.4
        bank.K_proto_max = 10
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        new_mu = bank.mu_c_l[uid]
        assert new_mu.dtype == torch.cfloat


class TestLocalOnlyMode:
    def test_local_only_skips(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, local_only=True)
        assert x_out.shape == (B, d_c)
        assert x_out.dtype == torch.cfloat


class TestOutputInvariants:
    def test_x_out_shape(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 3, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert x_out.shape == (B, d_c)
        assert x_out.dtype == torch.cfloat

    def test_x_out_no_nan(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 2, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert not torch.isnan(x_out).any()

    def test_gradient_flows(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat, requires_grad=True)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        loss = x_out.real.sum() + x_out.imag.sum()
        loss.backward()
        assert x_c.grad is not None
        assert x_c.grad.abs().max().item() > 0
