"""train_step.py — psc_train_step, memory_update_v605, train_step_v605 (§3.2b, §3.3, §3.5)."""
import math
import torch
import torch.nn.functional as F

from cfln.utils import (normalize_complex_center, batched_cayley_with_per_unit_lr, cayley_retraction_single,
                         stiefel_update_all_v51, detect_domain_boundary)
from cfln.modules.v9_ops import update_fisher_magnitude_freeze
from cfln.modules.si import compute_domain_confidence


# ── Stiefel updates ────────────────────────────────────────────────────────────

def stiefel_update_v58(bank, si, lr_stiefel, beta_SI=3.0):
    if bank.W_l.grad is None: return
    n=bank.n_l; sensory=bank.is_sensory_l[:n]; learner=~sensory
    li=learner.nonzero(as_tuple=True)[0]
    if len(li)==0: bank.W_l.grad=None; return
    om_n=si.get_unit_importance('bank.W_l',n)
    lr_per=lr_stiefel/(1.0+beta_SI*om_n[li])
    bank.W_l.data[li]=batched_cayley_with_per_unit_lr(bank.W_l.data[li],bank.W_l.grad[li],lr_per)
    bank.W_l.grad=None


def stiefel_update_cun(diff_aux, lr):
    for W in [diff_aux.cun.U1, diff_aux.cun.U2]:
        if W.grad is not None:
            W.data.copy_(cayley_retraction_single(W.data, W.grad, lr))
            W.grad=None


def _resolve_opt_grads(opt):
    """Resolve lazy conjugate bits on all gradients in an optimizer's param groups.
    Required before AdamW.step() when params may have complex-valued gradients
    (view_as_real fails on unresolved conjugate tensors)."""
    for pg in opt.param_groups:
        for p in pg['params']:
            if p.grad is not None and p.grad.is_conj():
                p.grad = p.grad.resolve_conj()


# ── Memory thresholds default ──────────────────────────────────────────────────

# v5.9.5 D5: default memory thresholds — prevents KeyError if cfg lacks 'memory_thresholds'
DEFAULT_MEMORY_THRESHOLDS = {
    'eps_s':0.01,'eps_p':0.001,'eps_split':0.5,'eps_merge':0.95,'r_reset':0.3,'eps_H':1e-4
}


# ── memory_update_v605 ─────────────────────────────────────────────────────────

def _find_merge_pair(bank, n, eps_merge, approx_sample=64):
    non_sensory=(~bank.is_sensory_l[:n]).nonzero(as_tuple=True)[0]
    if len(non_sensory)<2: return -1,-1
    if len(non_sensory)<=approx_sample: idx=non_sensory
    else:
        perm=torch.randperm(len(non_sensory),device=non_sensory.device)
        idx=non_sensory[perm[:approx_sample]]
    mu_s=normalize_complex_center(bank.mu_c_l[idx]); cs=(mu_s@mu_s.conj().T).real
    cs.fill_diagonal_(-1.0); mx=cs.max()
    if not torch.isfinite(mx) or mx<=eps_merge: return -1,-1
    nz=(cs==mx).nonzero()
    if nz.shape[0]==0: return -1,-1
    pair=nz[0]; return idx[pair[0]].item(),idx[pair[1]].item()


def _reactivate_from_exemplars(slot, dormancy, bank, dyn, x_rep):
    if dyn.n_active>=dyn.N_max: return -1
    cen=dormancy.centroids[slot]; ni=dyn.n_active
    with torch.no_grad():
        bank.mu_c_l.data[ni]=cen.to(bank.mu_c_l.device)
        bank.W_l.data[ni]=dormancy.W_l_saved[slot].to(bank.W_l.device)
        bank.log_alp_l.data[ni]=math.log(1e-3)
        bank.log_alpha_rq_l.data[ni]=bank.log_ell_l.data[ni]=0.0
        bank.is_sensory_l[ni]=False; bank.activation_freq_l[ni]=0.0
        bank.H_c_l[ni].zero_(); bank.h_c_l[ni].zero_()
        bank.rho_l[ni].zero_()          # v5.9.4: fresh reservoir on reactivation
        bank.log_scale_l.data[ni]=-3.0  # v5.9.4: reset temporal scale
        bank.active_mask_l[ni]=True
    dyn.n_active+=1; bank.n_l=dyn.n_active; return ni


def memory_update_v605(bank, dyn, dormancy, x_c, s_l, a_l_rq,
                        U_current, phase, cfg,
                        cached_grad_norms=None, si=None) -> dict:
    eps_s=cfg['eps_s']; eps_p=cfg['eps_p']; eps_split=cfg['eps_split']
    eps_merge=cfg['eps_merge']; r_reset=cfg['r_reset']
    eps_H=cfg.get('eps_H',1e-4); n=bank.n_l
    ops={'spawned':0,'pruned':0,'reactivated':0,'split':0,'merged':0,'reset':0,'new_sensory':0}
    alpha=torch.exp(bank.log_alp_l[:n]).clamp(1e-6,1.0)
    bank.alpha_histogram.update(alpha)
    bank.alpha_freeze=bank.alpha_histogram.get_alpha_freeze(cfg.get('sensory_fraction',0.15))
    x_rep=x_c.mean(0,keepdim=True)
    slots=dormancy.check_reactivation(x_rep,U_current,
                                       cfg.get('U_reactivate',0.7),cfg.get('cos_reactivate',0.7))
    for slot in slots[:2]:
        ni=_reactivate_from_exemplars(slot,dormancy,bank,dyn,x_rep)
        if ni>=0: ops['reactivated']+=1; dormancy.release_slot(slot)
    n=bank.n_l
    # §1.69: Welford-based spawn threshold replaces fixed eps_s check
    _e_last = getattr(bank, '_last_E_min_raw', None)
    _emin_n = int(getattr(bank, '_Emin_n', 0))
    if _e_last is not None and _emin_n >= 10:
        _sigma = (bank._Emin_var / max(_emin_n - 1, 1)) ** 0.5
        _spawn_cond = float(_e_last) > bank._Emin_mean + 2.5 * _sigma
    else:
        _spawn_cond = s_l.max().item() < eps_s  # fallback before stats stabilise
    if _spawn_cond and n < dyn.N_max and ops['reactivated'] == 0:
        dyn.spawn(x_c); ops['spawned'] += 1
    with torch.no_grad():
        act=(s_l.mean(0)>1.0/n).float()
        bank.log_alp_l.data[:n]+=0.01*act[:n]; bank.log_alp_l.data[:n].clamp_(-5,0)
        bank.update_activation_freq(s_l)
    ops['new_sensory']=bank.update_sensory_mask(cfg.get('sensory_fraction',0.15))
    alpha=torch.exp(bank.log_alp_l[:n]).clamp(1e-6,1.0); rq=a_l_rq.mean(0)[:n]
    sens=bank.is_sensory_l[:n]
    h_var=(bank.H_c_l[:n].abs().var(dim=-1).mean(-1) if bank.H_c_l is not None else torch.zeros(n,device=bank.is_sensory_l.device))
    keep=((sens)|(alpha>eps_p)|(rq>eps_p)|(h_var>eps_H))
    ki=keep.nonzero(as_tuple=True)[0]
    if len(ki)<n:
        ops['pruned']+=n-len(ki)
        dyn.prune(ki,dormancy=dormancy,si=si)  # remaps rho_l and log_scale_l (v5.9.4)
        n=bank.n_l
    if cached_grad_norms is not None and len(cached_grad_norms)>0:
        _ng=min(n,len(cached_grad_norms)); lg=cached_grad_norms[:_ng]*(~bank.is_sensory_l[:_ng]).float()
        for idx in (lg>eps_split).nonzero(as_tuple=True)[0][:3].tolist():
            dyn.split(idx); ops['split']+=1
        n=bank.n_l
    ia,ib=_find_merge_pair(bank,n,eps_merge,approx_sample=cfg.get('merge_sample',64))
    if ia>=0: dyn.merge(ia,ib,si=si); ops['merged']+=1; n=bank.n_l
    # v5.9.5 D2: use register_buffer + _has_prev_mu flag (survives checkpoint load)
    with torch.no_grad():
        if bank._has_prev_mu.item():
            drift=(bank.mu_c_l[:n].real-bank.mu_c_l_prev[:n].real).norm(dim=-1)
            rm=(drift>r_reset)&~bank.is_sensory_l[:n]
            if rm.any():
                ri=rm.nonzero(as_tuple=True)[0]
                bank.H_c_l[ri].zero_(); bank.h_c_l[ri].zero_()
                bank.rho_l[ri].zero_()  # reset reservoir for drifted units
                ops['reset']=int(rm.sum().item())
        bank.mu_c_l_prev[:n].copy_(bank.mu_c_l[:n].data)
        bank._has_prev_mu.fill_(True)
    bank.enforce_constraints()
    return ops


# ── _update_lam_p_corrections ──────────────────────────────────────────────────

def _update_lam_p_corrections(model, monitor_diag, cfg):
    E_D_thr=cfg.get('E_D_threshold',0.3); rate=cfg.get('lam_p_correction_rate',0.1)
    max_corr=cfg.get('lam_p_max_correction',3.0)
    for l,E_D in enumerate(monitor_diag.get('E_D_per_layer',[])):
        if l>=len(model.lam_p_corrections): break
        if E_D<E_D_thr: model.lam_p_corrections[l]=min(model.lam_p_corrections[l]*(1+rate),max_corr)
        else: model.lam_p_corrections[l]=(model.lam_p_corrections[l]*(1-rate*0.1)+rate*0.1*1.0)
        model.cfl_layers[l]._lam_p_correction=float(model.lam_p_corrections[l].item())


# ── _compute_local_losses ──────────────────────────────────────────────────────

def _compute_local_losses(model,input_ids,all_infos,stage,step,opts,cfg,lr_s,phase):
    _muon,_muon_diff,_opt_g,opt_u,_opt_p=opts
    if stage==0: return torch.tensor(0.0),torch.tensor(0.0)
    bank=model.bank; n=bank.n_l; L_local=torch.tensor(0.0,device=input_ids.device)
    # v5.9.7 M9: vectorized — was Python loop (16 iters), now 3 tensor ops
    n_sample=min(16,n); unit_idx=torch.randperm(n,device=input_ids.device)[:n_sample]
    valid=unit_idx[bank.H_c_l[unit_idx].abs().sum(dim=(-2,-1))>1e-8]
    if len(valid)>0:
        h_curr=bank.H_c_l[valid,:,-1]; h_prev=bank.H_c_l[valid,:,max(0,bank.D_g-2)]
        nc=h_curr.norm(dim=-1).clamp(1e-8); np_=h_prev.norm(dim=-1).clamp(1e-8)
        pos_sim=((h_curr.conj()*h_prev).real.sum(-1)/(nc*np_)).mean()
        L_local=L_local-pos_sim*0.01
    if n>=2:
        from cfln.utils import normalize_complex_center as _ncc
        mu_n=_ncc(bank.mu_c_l[:n]); cs=(mu_n@mu_n.conj().T).real
        L_div=((cs-torch.eye(n,device=cs.device))**2).mean()*0.001
    else: L_div=torch.tensor(0.0)
    L_total=L_local+L_div
    if L_total.requires_grad: opt_u.zero_grad(); L_total.backward(); opt_u.step()
    return L_local,L_div


# ── psc_train_step ─────────────────────────────────────────────────────────────

def psc_train_step(batch, model, psc_loss_fn, opts, si, phase, step,
                   total_steps, cfg, doc_ctx=None,
                   K_psc: int=4, u_epi_threshold: float=0.5) -> dict:
    """v6.0.5: PSC pre-training step. Wraps train_step_v605 and adds L_PSC.
    CE_baseline is FREE from Pass 1 (no extra compute).
    L_improve only triggers for U_epi > u_epi_threshold tokens (~15%).
    Expected overhead: +15% on PSC phase = +1.5% of total training budget.
    """
    # ── Pass 1: normal train step → CE_baseline free ──────────────────────
    info=train_step_v605(batch,model,opts,si,phase,step,total_steps,cfg,doc_ctx)
    ce_baseline=torch.tensor(info['L_task'],device=batch['input_ids'].device)
    u_epi_now=float(info['U_epistemic'])
    if u_epi_now<u_epi_threshold:
        return {**info,'L_PSC':0.0,'L_improve':0.0,'L_economy':0.0,'L_predictive':0.0}

    # ── Pass 2: thinking forward → CE_thinking (differentiable) ──────────
    model.train(); input_ids=batch['input_ids']; B,T=input_ids.shape
    device=input_ids.device
    # Snapshot r_lista_0 before thinking
    r_lista_0=model.diff_aux.cun.r_lista.detach().clone()
    # Deterministic thinking: K_psc LISTA steps with fixed THINK_START seed
    think_emb=model.encoder.embed(
        torch.tensor([[model.THINK_START_ID]]*B,device=device))[:,0,:]  # (B,d_c)
    model._in_thinking_mode=True
    model.diff_aux.cun._in_thinking_mode=True
    if hasattr(model.encoder,'titans'): model.encoder.titans._in_thinking_mode=True
    for _ in range(K_psc):
        # Single token LISTA step with think embedding as input
        think_emb.unsqueeze(1)          # (B,1,d_c)
        _=model(input_ids[:,:1].fill_(model.THINK_START_ID),training=True)
    r_lista_K=model.diff_aux.cun.r_lista.clone()  # (d_r_lista,) updated by K steps
    model._in_thinking_mode=False
    model.diff_aux.cun._in_thinking_mode=False
    if hasattr(model.encoder,'titans'): model.encoder.titans._in_thinking_mode=False
    # Thinking-augmented logits
    logits_think,_,_=model(input_ids,training=True)
    targets=input_ids[:,1:]
    ce_thinking=F.cross_entropy(
        logits_think[:,:-1].reshape(-1,logits_think.size(-1)),
        targets.reshape(-1),reduction='mean',
        ignore_index=cfg.get('pad_id',-100))

    # ── Future h_N targets for L_predictive ───────────────────────────────
    # Use h_N values already computed from Pass 1 forward (stored in cun state)
    n_fut=psc_loss_fn.n_future
    # Proxy: zero targets (conservative — no interference with main training)
    future_h_N=torch.zeros(n_fut,model.diff_aux.cun.U2.shape[0],
                            dtype=torch.cfloat,device=device)
    future_u_epi=torch.ones(n_fut,device=device)*u_epi_now

    # ── L_PSC backward (separate from Pass 1 backward) ────────────────────
    muon,muon_diff,opt_g,opt_u,opt_p=opts
    L_PSC=psc_loss_fn(ce_baseline,ce_thinking,r_lista_K,r_lista_0,
                       u_epi_now,future_h_N,future_u_epi)
    opt_g.zero_grad(); muon.zero_grad()
    L_PSC.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for pg in opt_g.param_groups for p in pg['params'] if p.grad is not None],
        cfg.get('grad_clip',1.0), foreach=False)
    _resolve_opt_grads(opt_g); opt_g.step(); muon.step()

    return {**info,
            'L_PSC':float(L_PSC),
            'L_improve':float(psc_loss_fn.alpha*
                             (-torch.log(torch.sigmoid(
                              ce_baseline.detach()-ce_thinking+psc_loss_fn.margin)))),
            'L_economy':0.0,'L_predictive':0.0}


# ── train_step_v605 ────────────────────────────────────────────────────────────

def train_step_v605(batch, model, opts, si, phase, step,
                     total_steps, cfg, doc_ctx=None) -> dict:
    """
    CFLN v6.0.7 training step.
    Cumulative changes through v5.9.6:
    - Function renamed to v596.
    - Node reservoir updated automatically in CFL5Layer.forward() — no explicit call.
    - LISTA reservoir updated automatically in lista_forward() — no explicit call.
    - opt_u.step() triggered after Pass 2 if log_scale_l has gradient.
    - warm_start_norm added to return dict (reservoir diagnostic).
    - W_enc_res/W_ri: fixed random buffers (ESN design, C1+C3)
    - opt_u.step() after Pass 1 (H1: log_scale_l + mu_c_l SI gradients applied)
    - opt_p.step() after Pass 1 (H2: persistent tier mu_c_p now updates)
    - domain_handler.on_domain_boundary: passes cun for reservoir reset (H3)
    - _seq_mode gates LISTA warm start (M1)
    - DEFAULT_MEMORY_THRESHOLDS guards KeyError (D5)
    - All 14 ordering invariants unchanged.
    """
    muon,muon_diff,opt_g,opt_u,opt_p=opts
    input_ids=batch['input_ids']; B,T=input_ids.shape
    stage=(0 if step<total_steps//4 else 1 if step<total_steps//2 else 2)
    if stage>=1 and not model.diff_aux._enabled: model.diff_aux.enable()
    # v5.9.6 I6: stage-0 freeze — let W_dec_res calibrate before log_scale_l grows
    # Prevents chicken-and-egg: W_dec_res can't learn while scale is tiny; scale can't grow
    # while W_dec_res is noisy. Stage 0 = setup phase; log_scale_l unfreezes at stage 1.
    if stage==0 and model.bank.log_scale_l.requires_grad:
        model.bank.log_scale_l.requires_grad_(False)
    elif stage>=1 and not model.bank.log_scale_l.requires_grad:
        model.bank.log_scale_l.requires_grad_(True)

    lr_start=cfg.get('lr_start',1e-3); lr_end=cfg.get('lr_end',1e-4)
    lr_ratio=lr_end/lr_start
    lr_s=lr_start*(lr_ratio)**(step/max(total_steps,1))
    lr_cun=lr_s*0.1
    lr_muon_s=cfg.get('lr_muon',lr_start)*(lr_ratio)**(step/max(total_steps,1))

    if step==0 and cfg.get('si_warmup_steps',1000)>=total_steps*0.9:
        import warnings
        warnings.warn(f"si_warmup_steps={cfg.get('si_warmup_steps')} >= 90% of "
                       f"total_steps={total_steps}. SI snapshots will never fire.")

    # 1. Conditional memory/STI reset (reservoirs NOT reset between batches)
    if doc_ctx is None or not doc_ctx.is_active:
        model.telescoping_mem.reset(); model.surprise_archive.reset()
    else: doc_ctx.record_window(n_chunks=T//model.encoder.C_chunk)
    model.sti_head.reset()
    # v5.9.5 M1: gate LISTA warm start based on sequential context
    model.diff_aux.cun._seq_mode=(doc_ctx is not None and doc_ctx.is_active)
    # v5.9.6 I2: set salience gate from Titans surprise signal (updated each step)
    _s_norm=getattr(model.encoder.titans,'_s_norm_last',1.0)
    model.bank._last_salience=min(2.0,max(0.3,float(_s_norm) if _s_norm else 1.0))

    # 2. Snapshot SI params
    si_params=si._get_named_params(model)
    prev_params={k:p.data.clone().detach() for k,p in si_params.items()}

    # ═══ PASS 1 ══════════════════════════════════════════════════════════
    logits,U_fin,aux=model(input_ids,training=True)
    all_infos=aux['all_infos']; targets=input_ids[:,1:]
    ce=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),
                        reduction='none',ignore_index=cfg.get('pad_id',-100)).reshape(B,T-1)
    L_task=ce.mean(); L_SI=si.compute_loss(si_params)
    L_null=model.encoder.titans._null_aux_loss or torch.tensor(0.0,device=input_ids.device)
    L_pass1=L_task+L_SI+L_null

    # ── v9.0 auxiliary losses ──────────────────────────────────────────────
    _bank=model.bank; _cun=model.diff_aux.cun

    # L_bridge per CFL layer (§1.50)
    _lb_sum=torch.tensor(0.0,device=input_ids.device)
    for _lay in model.cfl_layers:
        _lb=getattr(_lay,'_last_L_bridge',None)
        if _lb is not None and isinstance(_lb,torch.Tensor) and _lb.requires_grad:
            _lb_sum=_lb_sum+_lb
    L_pass1=L_pass1+cfg.get('lambda_bridge',0.1)*_lb_sum

    # L_vq VQ commitment (§1.59) — _L_compress_accum sums L_vq across ALL chunks in forward()
    _lvq=model._L_compress_accum
    if _lvq is not None and isinstance(_lvq,torch.Tensor) and _lvq.requires_grad:
        L_pass1=L_pass1+cfg.get('lambda_vq',0.01)*_lvq
    model._L_compress_accum=None  # clear after use

    # L_diversity beam anti-collapse (§1.47)
    _ldiv=getattr(_cun,'_last_beam_diversity',None)
    if _ldiv is not None and isinstance(_ldiv, torch.Tensor):
        L_pass1=L_pass1+cfg.get('lambda_diversity',0.01)*(-_ldiv**2)

    # ROB-L Lipschitz young units (§1.56)
    _n9=_bank.n_l; _ym=_bank.activation_freq_l[:_n9]<cfg.get('alpha_young',0.1)
    if _ym.any():
        L_pass1=L_pass1+cfg.get('lambda_lipschitz',0.001)*_bank.log_alp_l[:_n9][_ym].mean()

    # ROB-S learned phase width (§1.56)
    if hasattr(_bank,'log_sigma_bind'):
        L_pass1=L_pass1+cfg.get('lambda_sigma_reg',0.001)*torch.exp(-_bank.log_sigma_bind)

    # L_precision monitoring (§1.58) — non-differentiable tracking term
    _lp_list=getattr(_cun,'log_precision',None)
    if _lp_list:
        _prec_sum=sum(math.exp(p.item() if isinstance(p,torch.Tensor) else float(p)) for p in _lp_list)
        L_pass1=L_pass1+cfg.get('lambda_prec',0.001)*_prec_sum

    # Fisher-KL penalty from previous step (§1.57)
    _bkl_wu=cfg.get('beta_KL_warmup',500)
    if step>_bkl_wu and hasattr(model,'_fisher_diag') and model._fisher_diag:
        _kl_w=cfg.get('beta_KL',0.5)*min(1.0,(step-_bkl_wu)/max(_bkl_wu,1))
        _L_KL=torch.tensor(0.0,device=input_ids.device)
        for _kp in (p for pg in opt_g.param_groups for p in pg['params']):
            _fid=id(_kp)
            if _fid in model._fisher_diag and _fid in model._fisher_ref:
                _d=_kp-model._fisher_ref[_fid]
                _dr=_d.real if _kp.is_complex() else _d
                _L_KL=_L_KL+(model._fisher_diag[_fid]*_dr.pow(2)).sum()
        L_pass1=L_pass1+_kl_w*_L_KL

    # SE-2 MDLM masking (stage 0 only, §1.44)
    if stage==0 and cfg.get('p_mask',0.0)>0.0:
        _pm=cfg.get('p_mask',0.15); _mtok=cfg.get('mask_token_id',1)
        _mpos=torch.bernoulli(torch.full((B,T),_pm,device=input_ids.device)).bool()
        if _mpos.any():
            _mid=input_ids.clone(); _mid[_mpos]=_mtok
            # Clear W_ll cache before second forward: cache holds grad_fn nodes from
            # the first forward; reusing them causes "backward through graph twice" error.
            for _layer in model.cfl_layers: _layer._W_ll_cache.clear()
            _lm,_,_=model(_mid,training=False)
            _tgtm=input_ids.reshape(-1).clone(); _tgtm[~_mpos.reshape(-1)]=-100
            _Lmlm=F.cross_entropy(_lm.reshape(-1,_lm.size(-1)),_tgtm,ignore_index=-100)
            L_pass1=L_pass1+cfg.get('lambda_mlm',0.3)*_Lmlm

    opt_g.zero_grad(); muon.zero_grad(); L_pass1.backward()

    # Accumulate Fisher diagonal (§1.57) — after backward, before clip
    if not hasattr(model,'_fisher_diag'): model._fisher_diag={}; model._fisher_ref={}
    _stiefel_ids={id(model.bank.W_l),id(model.bank.W_p)}
    for _kp in (p for pg in opt_g.param_groups for p in pg['params']):
        if id(_kp) in _stiefel_ids or _kp.grad is None: continue
        _gr=_kp.grad.real if _kp.is_complex() else _kp.grad
        _f2=(_gr.detach()**2)
        _fid=id(_kp)
        if _fid not in model._fisher_diag:
            model._fisher_diag[_fid]=_f2.clone(); model._fisher_ref[_fid]=_kp.detach().clone()
        else:
            model._fisher_diag[_fid].mul_(0.99).add_(_f2,alpha=0.01)

    # §1.63: accumulate W_l per-unit scalar Fisher (W_l excluded from AdamW loop above)
    if model.bank.W_l.grad is not None:
        _wl_id = id(model.bank.W_l)
        _wl_g2 = model.bank.W_l.grad[:_bank.n_l].abs().mean(dim=(-2, -1)).detach()  # (n_l,)
        if _wl_id not in model._fisher_diag:
            model._fisher_diag[_wl_id] = _wl_g2.clone()
        else:
            model._fisher_diag[_wl_id].mul_(0.99).add_(_wl_g2, alpha=0.01)
    # Fisher-magnitude freeze check — every 100 steps (§1.63 C1)
    if step % 100 == 0:
        update_fisher_magnitude_freeze(_bank, model._fisher_diag)

    si.update_embed_omega(model,input_ids)
    torch.nn.utils.clip_grad_norm_(list(model.lam_p_schedule.parameters()),
                                    cfg.get('schedule_grad_clip',0.5), foreach=False)
    torch.nn.utils.clip_grad_norm_(
        [p for pg in opt_g.param_groups for p in pg['params']
         if p.grad is not None and p not in set(model.lam_p_schedule.parameters())],
        cfg['grad_clip'], foreach=False)
    _cached_w_l_grad_norms=None
    if model.bank.W_l.grad is not None:
        _cached_w_l_grad_norms=model.bank.W_l.grad[:model.bank.n_l].norm(dim=(-2,-1)).detach().clone()
    stiefel_update_v58(model.bank,si,lr_s,cfg.get('beta_SI_stiefel',0.25))
    stiefel_update_all_v51(model.bank,lr_l=0,lr_p=cfg.get('lr_persist',1e-6))  # v6.0.9: lr_g removed (global tier gone)
    muon.step(lr=lr_muon_s)
    _resolve_opt_grads(opt_g); opt_g.step()
    # v5.9.5 H1: step opt_u after Pass 1 so L_SI gradient reaches mu_c_l + log_scale_l gets L_task grad
    _resolve_opt_grads(opt_u); opt_u.step(); opt_u.zero_grad()
    # v5.9.5 H2: step opt_p after Pass 1 so mu_c_p gets L_SI gradient (was never stepped)
    _resolve_opt_grads(opt_p); opt_p.step()
    # v6.0.3 C3: clear W_ll overlap cache — routing params updated above → W_ll now stale
    # W_ll = apply_psd((a_rq(alpha,ell))^2) depends on log_alpha_rq_l and log_ell_l
    for _layer in model.cfl_layers: _layer._W_ll_cache.clear(); opt_p.zero_grad()
    model.encoder.fast_lru.enforce_stability()
    si.update_omega(si_params,prev_params)

    # ═══ PASS 2: Diffusion + LISTA ═════════════════════════════════════════
    L_diff=torch.tensor(0.0,device=input_ids.device)
    L_lista=torch.tensor(0.0,device=input_ids.device)
    warm_start_norm=0.0
    if model.diff_aux._enabled:
        t_sample=torch.randint(0,max(T-1,1),(1,)).item()
        xm=aux['x_c_final'][:,t_sample,:].detach()
        x_out=model.forward_single_position(xm)
        L_diff=model.diff_aux(x_out,training=True)
        L_lista=model.refine.compute_lista_loss(xm,x_out)
        L_pass2=L_diff+cfg.get('lambda_lista',0.1)*L_lista
        muon_diff.zero_grad(); opt_g.zero_grad(); opt_u.zero_grad()
        L_pass2.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for pg in opt_g.param_groups for p in pg['params'] if p.grad is not None],
            cfg['grad_clip'], foreach=False)
        stiefel_update_cun(model.diff_aux,lr_cun)
        muon_diff.step(lr=lr_muon_s*0.1)
        _resolve_opt_grads(opt_g); opt_g.step()
        # Step opt_u if log_scale_l received gradient via psi_for -> L_task path
        if model.bank.log_scale_l.grad is not None:
            _resolve_opt_grads(opt_u); opt_u.step()
        opt_u.zero_grad()
        # Warm-start diagnostic from last lista_forward call
        warm_start_norm=float(getattr(model.diff_aux.cun,'_last_warm_norm',0.0))

    # 13. Local losses
    L_local,_=_compute_local_losses(model,input_ids,all_infos,stage,step,opts,cfg,lr_s,phase)

    # 14. Maintenance
    model.bank.enforce_constraints()
    # v5.9.7 H5: average s_l across ALL layers and positions (was layer-0 pos-0 only)
    s_l_list=[all_infos[l][t].get('s_l') for l in range(len(all_infos))
              for t in range(len(all_infos[l])) if all_infos[l][t].get('s_l') is not None]
    s_l_avg=(torch.stack(s_l_list).mean(0) if s_l_list
             else torch.zeros(1,model.bank.n_l,device=input_ids.device))
    model.bank.update_activation_freq(s_l_avg)
    model.bank.alpha_histogram.update(torch.exp(model.bank.log_alp_l[:model.bank.n_l]).clamp(1e-6,1.0))

    # v5.9.8 CL.A: Proactive SI snapshot at high epistemic uncertainty
    # Fires BEFORE Titans EMA detects domain shift — protects parameters at point of novelty
    _u_epi_now=model.bank._u_epistemic_last
    _si_proactive_thr=cfg.get('si_proactive_threshold',0.8)
    _proactive_cooldown=cfg.get('proactive_cooldown',20)
    if (_u_epi_now>_si_proactive_thr
            and step>=cfg.get('si_warmup_steps',1000)
            and (step-model._last_proactive_snapshot)>_proactive_cooldown
            and not getattr(model,'_in_thinking_mode',False)):  # v6.0 CTP: no snapshot during thinking
        si.save_task_snapshot(si_params)   # protect current parameters
        model._last_proactive_snapshot=step

    if stage>=1 and step%50==0:
        xr=aux['x_c_final'].detach().mean(0).mean(0,keepdim=True)
        ar=all_infos[0][0].get('a_l_rq',torch.zeros(1,model.bank.n_l)).mean(0,keepdim=True)
        memory_update_v605(model.bank,model.dyn,model.dormancy_buf,xr,
                            s_l_avg[:1],ar,float(U_fin.mean()),phase,
                            cfg.get('memory_thresholds',DEFAULT_MEMORY_THRESHOLDS),  # v5.9.5 D5
                            cached_grad_norms=_cached_w_l_grad_norms,si=si)

    mon=model.monitor.step(step,[info[0] for info in all_infos])
    _update_lam_p_corrections(model,mon,cfg)
    si_warmup=cfg.get('si_warmup_steps',1000)

    if model.encoder.titans.domain_shift_detected:
        steps_since=step-model._last_domain_step
        if step>=si_warmup and steps_since>=cfg.get('min_snapshot_interval',50):
            _conf1=compute_domain_confidence(                    # v5.9.6 I8
                float(model.encoder.titans._s_domain_ema.item()),
                float(model.encoder.titans.domain_threshold.item()))   # v5.9.7 H1: use learned threshold
            model.domain_handler.on_domain_boundary(step,model.bank,si,si_params,
                                                     cun=model.diff_aux.cun,confidence=_conf1,
                                                     r_lista_attn=cfg.get('r_lista_attenuation_factor',None))  # v5.9.7 M7
            model._last_domain_step=step
            model.encoder.titans.domain_shift_detected=False

    slow_drift=model.slow_drift_detector.update(float(model.encoder.titans._s_domain_ema.item()),step)
    m4_boundary=(step%cfg.get('domain_check_freq',100)==0 and detect_domain_boundary(model.monitor))
    if slow_drift or m4_boundary:
        steps_since=step-model._last_domain_step
        if step>=si_warmup and steps_since>=cfg.get('min_snapshot_interval',50):
            _conf2=compute_domain_confidence(                    # v5.9.6 I8
                float(model.encoder.titans._s_domain_ema.item()),
                float(model.encoder.titans._tau_dom if hasattr(model.encoder.titans,'_tau_dom') else 3.0),
                routing_drop=float(mon.get('E_D_per_layer',[0])[0]) if mon else 0.0)
            model.domain_handler.on_domain_boundary(step,model.bank,si,si_params,
                                                     cun=model.diff_aux.cun,confidence=_conf2,
                                                     r_lista_attn=cfg.get('r_lista_attenuation_factor',None))  # v5.9.7 M7
            model._last_domain_step=step
            model.encoder.titans.domain_shift_detected=False

    return {
        'L_task':float(L_task.detach()),
        'L_SI':float(L_SI.detach()) if isinstance(L_SI,torch.Tensor) else 0.0,
        'L_diff':float(L_diff.detach() if isinstance(L_diff,torch.Tensor) else L_diff),
        'L_lista':float(L_lista.detach() if isinstance(L_lista,torch.Tensor) else L_lista),
        'L_compress':0.0,
        'L_null':float(L_null.detach()) if isinstance(L_null,torch.Tensor) else 0.0,
        'L_local':float(L_local.detach()) if isinstance(L_local,torch.Tensor) else 0.0,
        'U_mean':float(U_fin.detach().mean()),
        'n_sensory':int(model.bank.is_sensory_l[:model.bank.n_l].sum()),
        'n_dormant':model.dormancy_buf.n_dormant,
        'domain':model.domain_handler.current_domain,
        's_domain_ema':float(model.encoder.titans._s_domain_ema.item()),
        'tele_l2_fill':model.telescoping_mem._fill_L2,
        'tele_l3_fill':model.telescoping_mem._fill_L3,
        'lista_U_meta':float(aux.get('meta_refine',{}).get('U_meta',0.0) or 0.0),
        'warm_start_norm':warm_start_norm,
        'U_epistemic':float(model.bank._u_epistemic_last),   # v5.9.8 R2.A
        'U_hopfield':float(getattr(model.diff_aux.cun,'_last_confidence_stored',0.0)),  # v5.9.8 R2.B
    }
