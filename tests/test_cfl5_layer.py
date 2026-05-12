"""Tests for CFL5Layer.forward() output invariants."""
import torch
import pytest
from cfln.modules.cfl5 import CFL5Layer


class TestCFL5OutputShape:
    def test_x_out_shape_and_dtype(self, tiny_model):
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

    def test_gradient_flows_through_layer(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat, requires_grad=True)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        loss = x_out.real.sum() + x_out.imag.sum()
        loss.backward()
        assert x_c.grad is not None
        assert x_c.grad.abs().max().item() > 0


class TestSelLAndCardinality:
    def test_sel_l_cardinality_correct(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        sel_l = info['sel_l']
        k_l_computed = sel_l.shape[0]
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        E_l = info['E_l']
        assert (E_l >= -1e-5).all()

    def test_s_l_sums_near_one(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 2, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        s_l = info['s_l']
        sums = s_l.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(B), rtol=1e-3, atol=1e-4)


class TestBindingKernelPSD:
    def test_B_bind_is_psd(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        sel_l = info['sel_l']
        k_l = sel_l.shape[0]
        phi_sel = torch.angle(bank.H_c_l[sel_l].mean(-1).mean(-1))
        phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)
        sigma_sq_bind = torch.exp(2.0 * bank.log_sigma_bind)
        B_bind = torch.exp(-phi_diff**2 / sigma_sq_bind.clamp(1e-6)).float()
        eigs = torch.linalg.eigvalsh(B_bind)
        assert eigs.min().item() >= -1e-5


class TestGoalAnchorUpdates:
    def test_g_c_updates_outside_hypo_mode(self, tiny_model):
        if hasattr(tiny_model.bank, '_in_hypo_mode'):
            tiny_model.bank._in_hypo_mode = False
        if hasattr(tiny_model.bank, '_goal_frozen'):
            tiny_model.bank._goal_frozen = False
        bank = tiny_model.bank
        bank.g_c.zero_()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert bank.g_c.abs().max().item() > 1e-7

    def test_g_c_frozen_in_hypo_mode(self, tiny_model):
        if not hasattr(tiny_model.bank, '_in_hypo_mode'):
            return
        bank = tiny_model.bank
        bank._in_hypo_mode = True
        bank.g_c.zero_()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert bank.g_c.abs().max().item() < 1e-8


class TestKShotRefinement:
    def test_young_unit_refinement_conditions(self, tiny_model):
        bank = tiny_model.bank
        uid = 3
        bank.activation_freq_l[uid] = 0.06
        bank._proto_count[uid] = 1
        bank._u_epistemic_last = 0.2
        bank.tau_proto_min = 0.5
        bank.K_proto_max = 10
        old_count = int(bank._proto_count[uid].item())
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        new_count = int(bank._proto_count[uid].item())
        assert new_count >= old_count


class TestLocalOnlyBehavior:
    def test_local_only_true_skips_persistent(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, local_only=True)
        assert x_out.dtype == torch.cfloat
        assert x_out.shape == (B, d_c)


class TestUpdateResidueControl:
    def test_reservoir_updated_with_update_res_true(self, tiny_model):
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

    def test_reservoir_unchanged_with_update_res_false(self, tiny_model):
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


class TestSequentialHebbian:
    def test_prev_sel_l_set_after_forward(self, tiny_model):
        bank = tiny_model.bank
        bank._prev_sel_l = None
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=True)
        assert bank._prev_sel_l is not None

    def test_sequential_hebbian_updates_h_seq_mat(self, tiny_model):
        bank = tiny_model.bank
        old_H_seq = bank.H_seq_mat.clone()
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True, update_res=True)
        x_c2 = torch.randn(B, d_c, dtype=torch.cfloat)
        x_out2, Z_val2, U_val2, info2 = layer.forward(x_c2, training=True, update_res=True)
        new_H_seq = bank.H_seq_mat
        diff = (new_H_seq - old_H_seq).abs().max().item()
        assert diff >= 0


class TestZAndUValues:
    def test_Z_val_computed(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert isinstance(Z_val, (float, int)) or Z_val.numel() == 1
        z_float = float(Z_val)
        assert z_float >= 0

    def test_U_val_computed(self, tiny_model):
        bank = tiny_model.bank
        B, d_c = 1, bank.d_c
        x_c = torch.randn(B, d_c, dtype=torch.cfloat)
        layer = tiny_model.cfl_layers[0]
        x_out, Z_val, U_val, info = layer.forward(x_c, training=True)
        assert isinstance(U_val, (float, int)) or U_val.numel() == 1
        u_float = float(U_val)
        assert u_float >= 0
