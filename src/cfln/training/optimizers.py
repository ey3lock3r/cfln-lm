"""optimizers.py — build_optimizers_v605 (§3.1 of CFLN v6.0.9 spec)."""
import torch
from cfln.modules.muon import MuonOptimizer


def build_optimizers_v605(model, cfg):
    """
    5-tuple: (muon, muon_diff, opt_g, opt_u, opt_p). v5.9.7 changes:
    - muon_params: explicitly add W_dec_res (d_c×d_r_node) from CFBank (trained readout).
      W_enc_res is now a fixed buffer (v5.9.5 B8) — not trained, excluded from Muon.
    - muon_diff: picks up W_rs (d_c×d_r_lista) from diff_aux.cun via is_matrix check.
      W_ri is now a fixed buffer (v5.9.5) — excluded automatically.
      W_rc_bridge is now a fixed buffer (v5.9.7 C3) — removed from Muon.
    - opt_g: picks up log_hop_blend (scalar) automatically via diff_aux.named_parameters.
    - opt_g: log_beta_rs (scalar in diff_aux.cun) picked up automatically.
    - opt_u: adds log_scale_l (per-unit temporal influence scale) alongside log_alp_l.
    """
    stiefel_ids={id(model.bank.W_l),id(model.bank.W_p),  # v6.0.8: W_g removed
                  id(model.diff_aux.cun.U1),id(model.diff_aux.cun.U2)}
    # Pre-reserve unit-tier and persistent-tier param IDs so that sweeping
    # refine.named_parameters() / cfl_layers (which share bank) does NOT
    # claim them for muon/g1.  They are explicitly added to g2/g3 below.
    _unit_reserved = {id(getattr(model.bank, n))
                      for n in ['mu_c_l','log_alp_l','log_alpha_rq_l',
                                'log_ell_l','log_scale_l','log_decode_scale']
                      if getattr(model.bank, n, None) is not None}
    _pers_reserved = {id(getattr(model.bank, n))
                      for n in ['mu_c_p','log_alp_p','log_ell_p']
                      if getattr(model.bank, n, None) is not None}
    seen=_unit_reserved | _pers_reserved
    def is_matrix(p): return p.dim()>=2 and min(p.shape)>=4
    def add_m(name,p,grp):
        if id(p) not in seen and id(p) not in stiefel_ids and p.requires_grad:
            seen.add(id(p)); grp.append((name,p)); return True
        return False
    def add_g(p,grp):
        if p is not None and id(p) not in seen and id(p) not in stiefel_ids and p.requires_grad:
            seen.add(id(p)); grp.append(p); return True
        return False

    muon_params=[]
    for n,p in model.encoder.named_parameters():
        if is_matrix(p): add_m(f'encoder.{n}',p,muon_params)
    for n,p in model.bank.gat.named_parameters():
        if is_matrix(p): add_m(f'bank.gat.{n}',p,muon_params)
    # NODE RESERVOIR (v5.9.5 B8): only W_dec_res (trained readout) -> Muon
    # W_enc_res is now a fixed buffer — no gradient, excluded from Muon
    for n in ['W_dec_res']:
        p=getattr(model.bank,n,None)
        if p is not None: add_m(f'bank.{n}',p,muon_params)
    for n,p in model.highway.named_parameters():
        if is_matrix(p): add_m(f'highway.{n}',p,muon_params)
    for n in ['W_compress_L1','W_compress_L2','W_compress_L3']:
        p=getattr(model,n,None)
        if p is not None: add_m(n,p,muon_params)
    add_m('W_gate_mem',model.W_gate_mem,muon_params)
    for n,p in model.sti_head.named_parameters():
        if is_matrix(p): add_m(f'sti_head.{n}',p,muon_params)
    muon=MuonOptimizer(muon_params,lr=cfg.get('lr_muon',1e-3),
                        momentum=cfg.get('muon_momentum',0.95),ns_steps=cfg.get('muon_ns_steps',5))

    muon_diff_params=[]
    # W_rc_bridge: now a fixed buffer (v5.9.7 C3) — removed from Muon
    # (had zero gradient in v5.9.6 — was inside no_grad block, same bug as W_ri pre-v5.9.5)
    for n,p in model.diff_aux.named_parameters():
        # v5.9.5: W_ri is fixed buffer (excluded); W_rs is parameter and is picked up
        if is_matrix(p) and id(p) not in stiefel_ids:
            add_m(f'diff_aux.{n}',p,muon_diff_params)
    for n,p in model.refine.named_parameters():
        if is_matrix(p) and id(p) not in stiefel_ids:
            add_m(f'refine.{n}',p,muon_diff_params)
    muon_diff=MuonOptimizer(muon_diff_params,
                             lr=cfg.get('lr_muon_diff',cfg.get('lr_muon',1e-3)*0.1),
                             momentum=cfg.get('muon_momentum',0.95),
                             ns_steps=cfg.get('muon_ns_steps',5))

    g1=[]
    for n,p in model.encoder.named_parameters():
        if not is_matrix(p): add_g(p,g1)
    for p in model.sti_head.parameters():
        if not is_matrix(p): add_g(p,g1)
    for p in model.unc_module.parameters(): add_g(p,g1)
    for p in model.lam_p_schedule.parameters(): add_g(p,g1)
    for layer in model.cfl_layers:
        for n,p in layer.named_parameters():
            if not is_matrix(p): add_g(p,g1)
    add_g(model.w_outer_gate,g1)
    add_g(model.w_commit,g1)   # v6.0.1 C4: DCG+ commit score calibration (3 scalars) → opt_g
    add_g(model.encoder.titans.log_null_threshold,g1)
    add_g(model.encoder.titans.log_domain_threshold,g1)
    for n,p in model.highway.named_parameters():
        if not is_matrix(p): add_g(p,g1)
    for n,p in model.diff_aux.named_parameters():
        # log_beta_rs, log_w_meta, W_cache_gate, log_cache_gate_bias picked up automatically
        if id(p) not in stiefel_ids and not is_matrix(p): add_g(p,g1)
    for n,p in model.refine.named_parameters():
        if not is_matrix(p): add_g(p,g1)
    opt_g=torch.optim.AdamW(g1,lr=cfg.get('lr_local',3e-4),weight_decay=0.01,betas=(0.9,0.999))  # v6.0.8: lr_global removed

    g2=[]
    for n in ['mu_c_l','log_alp_l','log_alpha_rq_l','log_ell_l']:  # v6.0.8: removed mu_c_g,log_alp_g,log_kap_g
        p=getattr(model.bank,n,None)
        if p is not None and p.requires_grad: g2.append(p)
    # NODE RESERVOIR (v5.9.4): per-unit temporal scale -> unit optimizer
    p=getattr(model.bank,'log_scale_l',None)
    if p is not None and p.requires_grad: g2.append(p)
    g2.append(model.bank.log_decode_scale)  # v6.0.6: per-unit spectral filter
    opt_u=torch.optim.AdamW(g2,lr=cfg.get('lr_unit',1e-3),weight_decay=0.0,betas=(0.9,0.999))

    g3=[]
    for n in ['mu_c_p','log_alp_p','log_ell_p']:
        p=getattr(model.bank,n,None)
        if p is not None and p.requires_grad: g3.append(p)
    opt_p=torch.optim.AdamW(g3,lr=cfg.get('lr_persist',1e-6),weight_decay=0.0)
    model._optimizers_built=True   # v6.0.2 H3: guard for expand_vocabulary ordering
    return muon,muon_diff,opt_g,opt_u,opt_p
