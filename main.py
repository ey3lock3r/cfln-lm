"""main.py — CFLN v9.0 CLI entry point.

Usage:
    python main.py --train [--config path.json]
    python main.py --generate [--config path.json] [--prompt "text"]
"""
import argparse
import json
import sys


def _load_cfg(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _make_tiny_cfg(overrides=None):
    """Tiny config for smoke-test / default runs when no config is supplied."""
    cfg = {
        'd_c': 32, 'vocab_size': 256, 'n_l': 80, 'n_p': 8, 'L': 2,
        'n_heads_gat': 2, 'd_e_l': 8, 'd_e_p': 8, 'd_ssm_fast': 8,
        'S_f': 8, 'C_chunk': 8, 'K_L1': 8, 'K_L2': 4, 'K_L3': 4,
        'N_archive': 8, 'd_r_node': 4, 'd_r_lista': 8, 'T': 16, 'B': 2,
        'N_iter_refine': 2, 'N_hop_refine': 1, 'surprise_warmup_chunks': 2,
        'sparse_code_cache_K': 4, 'episodic_rule_cache_n': 4,
        'K_hebb': 4, 'D_g': 4, 'D_bptt': 2, 'K_stats': 2,
        'T_diff': 10, 'n_fourier': 4, 'N_dormant': 16,
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
        # ── v7.0 ──────────────────────────────────────────────────────────
        'arc_dual_key': True, 'use_goal_context': True,
        # hypo/push/pop IDs assigned at runtime by expand_vocabulary
        # ── v8.0 ──────────────────────────────────────────────────────────
        'n_roles': 4, 'tau_consol': 3.0, 'alpha_consol': 0.001,
        'persist_archive': False, 'ssp_max_depth': 4,
        'lambda_recon': 0.01, 'psd_apply_every': 10,
        # ── Addendum SE/Q-BEAM/TS ─────────────────────────────────────────
        'K_proto_max': 10, 'tau_proto': 0.6, 'alpha_young': 0.1,
        'p_mask': 0.0, 'lambda_mlm': 0.3, 'beam_B': 2, 'lambda_diversity': 0.01,
        # ── v9.0 ──────────────────────────────────────────────────────────
        'lambda_bridge': 0.1, 'lambda_vq': 0.01,
        'beta_KL': 0.5, 'beta_SI_stiefel': 0.25, 'beta_KL_warmup': 500,
        'lambda_prec': 0.001, 'lambda_lipschitz': 0.001, 'lambda_sigma_reg': 0.001,
        'k_l_min': 10, 'k_l_max': 40, 'beam_B_max': 3, 'tau_proto_min': 0.4,
        'alpha_micro': 0.0001,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def run_train(cfg_path):
    import torch
    from cfln.modules.model import CFLNModel
    from cfln.training.optimizers import build_optimizers_v605
    from cfln.training.train_step import train_step_v605

    file_cfg = _load_cfg(cfg_path)
    cfg = _make_tiny_cfg(file_cfg if file_cfg else None)

    print(f"Instantiating CFLNModel (d_c={cfg['d_c']}, L={cfg['L']}, "
          f"vocab_size={cfg['vocab_size']})...")
    model = CFLNModel(cfg)
    # model.train() skipped: shared submodule refs in model cause nn.Module recursion
    # CFLNModel.__init__ sets training=True by default (nn.Module default)

    si = model.si
    opts = build_optimizers_v605(model, cfg)

    T = cfg['T']
    B = cfg['B']
    total_steps = 3
    print(f"Running {total_steps} smoke-test train steps "
          f"(B={B}, T={T}, random data)...")

    for step in range(total_steps):
        batch = {
            'input_ids': torch.randint(0, cfg['vocab_size'], (B, T)),
        }
        info = train_step_v605(
            batch, model, opts, si,
            phase='pretrain', step=step,
            total_steps=total_steps, cfg=cfg)
        print(f"  step {step}: L_task={info['L_task']:.4f} "
              f"L_SI={info['L_SI']:.4f} "
              f"U_epistemic={info['U_epistemic']:.4f}")

    print("Smoke test passed.")


def run_generate(cfg_path, prompt_text):
    import torch
    from cfln.modules.model import CFLNModel
    from cfln.inference.ctp import generate_cfln_ctp

    file_cfg = _load_cfg(cfg_path)
    cfg = _make_tiny_cfg(file_cfg if file_cfg else None)

    print("Instantiating CFLNModel for generation...")
    model = CFLNModel(cfg)
    # model.eval() skipped: shared submodule refs in model cause nn.Module recursion
    # Set training=False directly on the top-level module
    model.training = False

    # Expand vocabulary to enable CTP + HYPO/SSP tokens (v9.0: n_new=6)
    model.expand_vocabulary(n_new=6)

    # Encode prompt as token IDs (character-level for demo)
    if prompt_text:
        vocab_size = cfg['vocab_size']
        prompt_ids = torch.tensor(
            [[ord(c) % vocab_size for c in prompt_text[:8]]],
            dtype=torch.long)
    else:
        prompt_ids = torch.randint(0, cfg['vocab_size'], (1, 4))

    print(f"Generating with CTP (prompt shape={list(prompt_ids.shape)})...")
    with torch.no_grad():
        out = generate_cfln_ctp(
            model, prompt_ids,
            max_new_tokens=8,
            max_think_tokens=4,
            think_threshold=0.5,
            temperature=1.0,
            top_k=10)
    print(f"Output shape: {list(out.shape)}")
    print(f"Generated token IDs: {out[0].tolist()}")


def main():
    parser = argparse.ArgumentParser(description='CFLN v6.0.9')
    parser.add_argument('--train', action='store_true',
                        help='Run training (smoke test with random data if no config)')
    parser.add_argument('--generate', action='store_true',
                        help='Run CTP generation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    args = parser.parse_args()

    if not args.train and not args.generate:
        parser.print_help()
        sys.exit(0)

    if args.train:
        run_train(args.config)

    if args.generate:
        run_generate(args.config, args.prompt)


if __name__ == '__main__':
    main()
