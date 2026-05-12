"""
v9.0 standalone utility functions.
Imported by model.py, diffusion.py, train_step.py — kept here to avoid circular imports.
"""
import torch
import torch.nn.functional as F


def consolidate_arc_to_cnep(bank, cun, tau_consol=3.0, alpha_consol=0.001):
    """§1.37 Y2 — Promote high-utility ARC rules into nearest μ_c_l centroid.

    SI-protected: update magnitude gated by (1 - activation_freq / alpha_freeze_thresh).
    Call BEFORE clearing session state in reset_for_inference().
    """
    n_r = getattr(cun, '_rule_cache_n', 0)
    if n_r == 0:
        return
    with torch.no_grad():
        for idx in range(n_r):
            if float(cun.rule_util[idx].item()) < tau_consol:
                continue
            k_rule = cun.rule_K[idx, :bank.d_c]  # concept key (d_c,)
            dists = (bank.mu_c_l[:bank.n_l] - k_rule).norm(dim=-1).real
            nearest = int(dists.argmin().item())
            mu_target = bank.mu_c_l[nearest]
            si_gate = 1.0
            if hasattr(bank, '_si_omega_unit'):
                si_gate = float((1.0 - bank._si_omega_unit[nearest].clamp(0, 1)).item())
            _fisher = float(bank.fisher_unit[nearest].item()) if hasattr(bank, 'fisher_unit') else 0.0
            alpha_eff = alpha_consol / (1.0 + _fisher)
            delta = alpha_eff * si_gate * (k_rule - mu_target)
            bank.mu_c_l.data[nearest] = mu_target + delta


def micro_consolidate_arc(bank, cun, cfg):
    """§1.54 KA-MC — CLS micro-consolidation: top-1 ARC rule → μ_c_l per chunk.

    Runs at every chunk boundary (inside _update_telescoping).
    Uses alpha_micro=0.0001 (much smaller than CONSOL-1's alpha_consol=0.001).
    """
    n_r = getattr(cun, '_rule_cache_n', 0)
    if n_r == 0:
        return
    tau = cfg.get('tau_consol', 3.0)
    alpha_micro = cfg.get('alpha_micro', 0.0001)
    alpha_young = cfg.get('alpha_young', 0.1)

    utils = cun.rule_util[:n_r]
    best = int(utils.argmax().item())
    if float(utils[best].item()) < tau:
        return

    k_rule = cun.rule_K[best, :bank.d_c].detach()
    with torch.no_grad():
        dists = (bank.mu_c_l[:bank.n_l] - k_rule).norm(dim=-1).real
        nearest = int(dists.argmin().item())
        freq = float(bank.activation_freq_l[nearest].item())
        si_gate = max(0.0, 1.0 - freq / max(alpha_young, 1e-8))
        _fisher = float(bank.fisher_unit[nearest].item()) if hasattr(bank, 'fisher_unit') else 0.0
        alpha_eff = alpha_micro / (1.0 + _fisher)
        delta = alpha_eff * si_gate * (k_rule - bank.mu_c_l[nearest])
        bank.mu_c_l.data[nearest] += delta


def update_fisher_magnitude_freeze(bank, fisher_diag, k_sigma=1.5):
    """§1.63 C1: Fisher-magnitude parallel freeze. Runs every 100 steps.

    Uses W_l per-unit gradient magnitude accumulated separately (W_l excluded from AdamW loop).
    High Fisher(W_l) = unit actively shaping its projection = mature, worth protecting.
    fisher_diag key: id(bank.W_l), value shape: (n_l,) per-unit scalar EMA.
    """
    n = bank.n_l
    if n == 0:
        return
    wl_id = id(bank.W_l)
    if wl_id not in fisher_diag or fisher_diag[wl_id].shape[0] < n:
        return
    with torch.no_grad():
        unit_fisher = fisher_diag[wl_id][:n].float().abs()  # (n,) — already per-unit scalar
        bank.fisher_unit[:n] = unit_fisher
        if unit_fisher.std() < 1e-8:
            return  # early training guard
        threshold = unit_fisher.mean() + k_sigma * unit_fisher.std()
        new_frozen = (unit_fisher > threshold) & ~bank.is_sensory_l[:n]
        if new_frozen.any():
            bank.is_sensory_l[:n] |= new_frozen
            bank.sensory_domain_id[:n][new_frozen] = -1


def compute_Q_beam(h_N, r_lista, r_goal_proxy, goal_stack, x_c,
                   W_bridge=None, E_min_raw=None, H_route_raw=None,
                   log_w_beam=None, phi_rel=None):
    """§1.46 Q-BEAM + §1.53 D1 — Multi-field beam quality composite.

    F3 (MDL):          -||h_N||₁            information theory / sparsity
    F4 (Lyapunov):     -||r_lista - r_goal||²  control theory / goal proximity
    F5 (CSP):          min cosine_sim over SSP stack  arc-consistency
    F2 (predictive):   -||(x_c_mean - W_bridge @ r_lista)||  optional, needs trained W_bridge
    F1 (thermodynamic):-E_min_raw × H_route_raw   optional, routing free energy
    D1 (relational):   phi_rel.norm()             §1.53, relational richness

    Weights: equal 1/N (parameter-free) unless log_w_beam provided.
    log_w_beam: up to 3 scalars for F3+F4+F5 core signals; optional signals always equal-weighted.
    """
    signals = []

    # F3: MDL — sparse code economy (always computed)
    mdl = -h_N.abs().sum()
    signals.append(mdl)

    # F4: Lyapunov goal proximity (only if goal proxy available)
    if r_goal_proxy is not None:
        lyap = -(r_lista - r_goal_proxy).norm() ** 2
        signals.append(lyap)

    # F5: CSP arc-consistency (only if SSP stack non-empty)
    if goal_stack:
        sims = [
            F.cosine_similarity(r_lista.real.unsqueeze(0), s.real.unsqueeze(0)).item()
            for s in goal_stack
        ]
        signals.append(torch.tensor(float(min(sims))))

    # F2: Predictive coding (optional — requires trained W_bridge)
    if W_bridge is not None:
        x_pred = W_bridge @ r_lista
        rc_res = -(x_c.mean(0) - x_pred).norm()
        signals.append(rc_res)

    # F1: Thermodynamics / free energy (optional)
    if E_min_raw is not None and H_route_raw is not None:
        signals.append(torch.tensor(float(-(E_min_raw * H_route_raw))))

    # D1: phi_rel relational richness (§1.53)
    if phi_rel is not None:
        signals.append(torch.tensor(float(phi_rel.norm().item())))

    if not signals:
        return torch.tensor(0.0)

    signals_t = torch.stack([
        s if isinstance(s, torch.Tensor) else torch.tensor(float(s))
        for s in signals
    ])

    # Apply learned weights to core signals (F3, F4, F5) if provided
    n_core = min(3, len(signals_t))
    if log_w_beam is not None and len(log_w_beam) >= n_core and n_core > 0:
        w_core = torch.softmax(log_w_beam[:n_core], dim=0)
        core_score = (w_core * signals_t[:n_core]).sum()
        if len(signals_t) > n_core:
            extra_score = signals_t[n_core:].mean()
            return (core_score + extra_score) / 2.0
        return core_score
    else:
        return signals_t.mean()
