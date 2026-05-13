import torch
import torch.nn as nn


class SynapticIntelligence(nn.Module):
    """
    Online SI. v5.9.5 changes:
    - _get_named_params: W_dec_res + log_scale_l (node readout), W_rs (LISTA readout)
    - W_enc_res and W_ri removed — now fixed buffers (no gradient, no SI protection needed)
    All other logic unchanged from v5.9.3 (displacement-only omega, _omega_scales in __init__).
    """
    def __init__(self,c_SI=0.5,rho_SI=0.999,beta_SI=3.0):
        super().__init__()
        self.c_SI=c_SI; self.rho_SI=rho_SI; self.beta_SI=beta_SI
        self.theta_star={}; self.omega={}; self.active=False
        self._omega_scales={'sti_head.W_vocab.weight':0.3}
        object.__setattr__(self, '_model_ref', None)  # bypass nn.Module registry to avoid circular child ref
        self._embed_omega_real=None; self._embed_omega_imag=None
        self._embed_theta_star_real=None; self._embed_theta_star_imag=None
        self._embed_omega_scale=0.1

    def _get_named_params(self,model) -> dict:
        object.__setattr__(self, '_model_ref', model); protected={}
        for n in ['W_c_proj','B_dec_fast']:
            p=getattr(model.encoder,n,None)
            if p is not None: protected[f'encoder.{n}']=p
        for n in ['B_c','C_c','log_nu','theta']:
            p=getattr(model.encoder.fast_lru,n,None)
            if p is not None: protected[f'encoder.fast_lru.{n}']=p
        for n in ['W_K','W_V','W_Q','log_eta','w_theta']:
            p=getattr(model.encoder.titans,n,None)
            if p is not None: protected[f'encoder.titans.{n}']=p
        for n in ['B_c_out','w_c_g','C_proj']:
            p=getattr(model.sti_head,n,None)
            if p is not None: protected[f'sti_head.{n}']=p
        if hasattr(model.sti_head,'W_vocab') and model.sti_head.W_vocab is not None:
            protected['sti_head.W_vocab.weight']=model.sti_head.W_vocab.weight
        for n in ['mu_c_l','mu_c_p','W_l','W_p']:  # v6.0.8: mu_c_g, W_g removed
            p=getattr(model.bank,n,None)
            if p is not None: protected[f'bank.{n}']=p
        # NODE RESERVOIR (v5.9.5): W_dec_res trained readout only (W_enc_res now fixed buffer)
        for n in ['W_dec_res']:
            p=getattr(model.bank,n,None)
            if p is not None: protected[f'bank.{n}']=p
        p=getattr(model.bank,'log_scale_l',None)
        if p is not None: protected['bank.log_scale_l']=p
        for n in ['W_compress_L1','W_compress_L2','W_compress_L3']:
            p=getattr(model,n,None)
            if p is not None: protected[n]=p
        p=getattr(model,'W_gate_mem',None)
        if p is not None: protected['W_gate_mem']=p
        for li in range(model.highway.L):
            for mat in ['w_a','w_c']:
                p=getattr(model.highway,mat)[li]
                protected[f'highway.{mat}_{li}']=p
        # LISTA RESERVOIR (v5.9.5): W_rs trained readout only (W_ri now fixed buffer)
        for n in ['W_rs']:
            p=getattr(model.diff_aux.cun,n,None)
            if p is not None: protected[f'diff_aux.cun.{n}']=p
        # W_rc_bridge: fixed buffer (v5.9.7 C3) — not in SI
        # log_hop_blend: warm-start blend, intentionally adaptive — not SI-protected
        # ROUTING SHAPE PARAMS (v5.9.7 M8): protect against routing drift post-shift
        for n in ['log_alpha_rq_l','log_ell_l']:
            p=getattr(model.bank,n,None)
            if p is not None: protected[f'bank.{n}']=p
        # v6.0.1 H3: protect metacognition weights against CL drift
        # log_w_meta: U_meta_v2 composition (affects reasoning quality consistency)
        p=getattr(model.diff_aux.cun,'log_w_meta',None)
        if p is not None: protected['diff_aux.cun.log_w_meta']=p
        # log_lam_seq_gat (per CFL layer): sequential Hebbian GAT weight
        for li,layer in enumerate(model.cfl_layers):
            p=getattr(layer,'log_lam_seq_gat',None)
            if p is not None: protected[f'cfl_layers.{li}.log_lam_seq_gat']=p
        # W_vocab.bias (v6.0.1 C1 follow-up): protect expanded bias rows
        if (hasattr(model.sti_head,'W_vocab') and model.sti_head.W_vocab is not None
                and model.sti_head.W_vocab.bias is not None):
            protected['sti_head.W_vocab.bias']=model.sti_head.W_vocab.bias
        for k,p in protected.items():
            if k not in self.omega:
                self.omega[k]=torch.zeros_like(p.data,dtype=torch.float32)
        return protected

    def remap_after_prune(self,keep_idx: torch.Tensor) -> None:
        k=len(keep_idx)
        for name in [n for n in list(self.omega.keys())
                      if any(s in n for s in ['W_l','mu_c_l','log_scale_l'])]:
            if name in self.omega:
                om=self.omega[name]; new_om=torch.zeros_like(om); new_om[:k]=om[keep_idx]; self.omega[name]=new_om
            if name in self.theta_star:
                ts=self.theta_star[name]; new_ts=ts.clone(); new_ts[:k]=ts[keep_idx]; self.theta_star[name]=new_ts

    @torch.no_grad()
    def update_omega(self,named_params: dict,prev_params: dict):
        for n,p in named_params.items():
            if n not in self.omega: continue
            dp=p.data-prev_params[n].to(p.device)
            c=(dp.conj()*dp).real.float() if dp.dtype==torch.cfloat else dp.float().pow(2)
            self.omega[n]=self.rho_SI*self.omega[n]+(1-self.rho_SI)*c

    @torch.no_grad()
    def update_embed_omega(self,model,input_ids):
        if model.embed.embed_real.weight.grad is None: return
        if self._embed_omega_real is None:
            vs=model.embed.embed_real.weight.shape[0]; dev=model.embed.embed_real.weight.device
            self._embed_omega_real=torch.zeros(vs,dtype=torch.float32,device=dev)
            self._embed_omega_imag=torch.zeros(vs,dtype=torch.float32,device=dev)
        unique=input_ids.view(-1).unique()
        g_r=model.embed.embed_real.weight.grad[unique]; g_i=model.embed.embed_imag.weight.grad[unique]
        if self._embed_theta_star_real is not None:
            dp_r=model.embed.embed_real.weight.data[unique]-self._embed_theta_star_real[unique].to(g_r.device)
            dp_i=model.embed.embed_imag.weight.data[unique]-self._embed_theta_star_imag[unique].to(g_i.device)
            # §1.12: displacement-only Ω for non-Stiefel params (embed is AdamW, not Stiefel)
            self._embed_omega_real[unique]=(self.rho_SI*self._embed_omega_real[unique]
                                             +(1-self.rho_SI)*dp_r.pow(2).sum(-1))
            self._embed_omega_imag[unique]=(self.rho_SI*self._embed_omega_imag[unique]
                                             +(1-self.rho_SI)*dp_i.pow(2).sum(-1))

    def save_task_snapshot(self,named_params: dict):
        for n,p in named_params.items(): self.theta_star[n]=p.data.clone().detach()
        if self._model_ref is not None:
            self._embed_theta_star_real=self._model_ref.embed.embed_real.weight.data.clone()
            self._embed_theta_star_imag=self._model_ref.embed.embed_imag.weight.data.clone()
        self.active=True

    def compute_loss(self,named_params: dict) -> torch.Tensor:
        if not self.active: return torch.tensor(0.0)
        dev=next(iter(named_params.values())).device; loss=torch.tensor(0.0,device=dev)
        for n,p in named_params.items():
            if n not in self.theta_star or n not in self.omega: continue
            if 'W_l' in n: continue  # v6.0.8: W_g check removed (no W_g)
            diff=p-self.theta_star[n].to(dev); om=self.omega[n].to(dev)
            sq=(diff.real**2+diff.imag**2) if diff.dtype==torch.cfloat else diff**2
            scale=self._omega_scales.get(n,1.0); loss=loss+scale*(om*sq).sum()
        if self._embed_omega_real is not None and self._embed_theta_star_real is not None and self._model_ref is not None:
            m=self._model_ref
            dr=m.embed.embed_real.weight-self._embed_theta_star_real.to(dev)
            di=m.embed.embed_imag.weight-self._embed_theta_star_imag.to(dev)
            omr=self._embed_omega_real.to(dev).unsqueeze(1); omi=self._embed_omega_imag.to(dev).unsqueeze(1)
            loss=loss+self._embed_omega_scale*(omr*dr**2+omi*di**2).sum()
        return (self.c_SI/2)*loss

    def get_unit_importance(self,W_name: str,n_units: int) -> torch.Tensor:
        if W_name not in self.omega: return torch.zeros(n_units)
        pu=self.omega[W_name][:n_units].reshape(n_units,-1).sum(-1)
        return pu/pu.max().clamp(1e-8)


class ExemplarDormancyBuffer:
    """Exemplar-based dormancy. Unchanged from v5.9.3."""
    def __init__(self,d_c,d_e_l,D_g=8,capacity=512):
        self.d_c=d_c; self.d_e_l=d_e_l; self.D_g=D_g; self.capacity=capacity
        self.exemplars  =torch.zeros(capacity,D_g,d_c,dtype=torch.cfloat)
        self.centroids  =torch.zeros(capacity,d_c,dtype=torch.cfloat)
        self.W_l_saved  =torch.zeros(capacity,d_e_l,d_c,dtype=torch.cfloat)
        self.active_mask=torch.zeros(capacity,dtype=torch.bool)
        self._next_slot=0; self.n_dormant=0

    def add_from_history(self,bank,unit_idx):
        attempts=0
        while self.active_mask[self._next_slot] and attempts<self.capacity:
            self._next_slot=(self._next_slot+1)%self.capacity; attempts+=1
        if attempts>=self.capacity: return False
        slot=self._next_slot; W_i=bank.W_l.data[unit_idx]; mu_i=bank.mu_c_l[unit_idx]
        H_c=bank.H_c_l[unit_idx]; W_H=W_i.conj().T
        with torch.no_grad():
            for k in range(self.D_g): self.exemplars[slot,k]=W_H@H_c[:,k]+mu_i
            self.centroids[slot]=self.exemplars[slot].mean(0)
            self.W_l_saved[slot]=W_i.clone()
            self.active_mask[slot]=True
        self._next_slot=(self._next_slot+1)%self.capacity; self.n_dormant+=1; return True

    def check_reactivation(self,x_rep,U_curr,U_thr=0.7,cos_thr=0.7):
        if self.n_dormant==0 or U_curr<=U_thr: return []
        slots=self.active_mask.nonzero(as_tuple=True)[0]
        if len(slots)==0: return []
        dev=x_rep.device; cents=self.centroids[slots].to(dev)
        xn=x_rep/x_rep.norm().clamp(1e-8); cn=cents/cents.norm(dim=1,keepdim=True).clamp(1e-8)
        return slots[((xn.conj()*cn).real.sum(-1)>cos_thr)].tolist()

    def release_slot(self,slot):
        with torch.no_grad(): self.active_mask[slot]=False; self.n_dormant-=1


def compute_domain_confidence(s_domain_ema: float, tau_dom: float,
                               routing_drop: float=0.0,
                               slow_drift_mag: float=0.0) -> float:
    """v5.9.6 I8: Unified domain shift confidence score in [0,1].
    0=no shift, 1=definite shift. Combines all 3 detection channels.
    Uses max-pool: any channel confident → overall confident.
    c_titans: soft threshold via sigmoid around tau_dom
    c_m4:     routing diversity drop magnitude (0-1)
    c_drift:  slow drift magnitude scaled to [0,1]
    """
    import math as _math
    import torch as _torch
    if _math.isnan(s_domain_ema) or _math.isnan(tau_dom): return 0.0
    c_titans=float(_torch.sigmoid(_torch.tensor((s_domain_ema-tau_dom)*3.0)).item())
    c_m4=min(1.0,float(routing_drop)*2.0)
    c_drift=min(1.0,float(slow_drift_mag)*2.0)
    return float(max(c_titans,c_m4*0.8,c_drift*0.6))


class DomainTransitionHandler:
    def __init__(self,max_history=20):
        self.current_domain=0; self.domain_history=[]; self._max_history=max_history

    def on_domain_boundary(self,step,bank,si,si_params,new_domain_id=None,cun=None,
                           confidence: float=1.0, r_lista_attn: float=None):
        """v5.9.6 I8: graded response based on domain shift confidence.
        confidence in [0,1]: 0=no action, 0.4-0.8=partial attenuate, >0.8=full reset.
        This is strictly more expressive than binary reset (v5.9.5 always full-reset).
        """
        if new_domain_id is None: new_domain_id=self.current_domain+1
        si.save_task_snapshot(si_params)
        bank.release_domain_sensory(self.current_domain)
        # v5.9.6 I8: graded reservoir response
        if confidence > 0.8:
            bank.reset_reservoir()                  # definite shift: full reset
            if cun is not None: cun.reset_lista_reservoir()
        elif confidence > 0.4:
            bank.attenuate_reservoir(0.5)           # moderate shift: half attenuate
            if cun is not None:
                with torch.no_grad():
                    factor=r_lista_attn if r_lista_attn is not None else (1.0-confidence)
                    cun.r_lista.mul_(factor)  # v5.9.7 M7: configurable attenuation factor
        # else: gentle shift (confidence <= 0.4), preserve reservoir context
        self.current_domain=new_domain_id
        self.domain_history.append((step,new_domain_id))
        if len(self.domain_history)>self._max_history: self.domain_history=self.domain_history[-self._max_history:]
        return new_domain_id
