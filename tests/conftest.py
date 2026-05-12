"""Shared fixtures for CFLN-LM test suite. Mirrors _make_tiny_model() in test_utils.py exactly."""
import pytest
import torch
from cfln.modules.model import CFLNModel


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)


@pytest.fixture
def tiny_cfg():
    return {
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
        'memory_thresholds': {
            'eps_s': 0.01, 'eps_p': 0.001, 'eps_split': 0.5,
            'eps_merge': 0.95, 'r_reset': 0.3, 'eps_H': 1e-4,
        },
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


@pytest.fixture
def tiny_model(tiny_cfg):
    return CFLNModel(tiny_cfg)
