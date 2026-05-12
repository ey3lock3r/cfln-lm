import math
import torch
import torch.nn as nn

from cfln.utils import (compute_energies, rq_routing, entmax15_with_floor,
                         apply_psd_to_weight_matrix, compute_direction_angles_complex)


class CFL5Layer(nn.Module):
    """
    One CFL-5 layer. v5.9.4 changes over v5.9.3:
    - psi_for_local: uses predicted prototype (mu_pred) as expansion center AND return center
    - psi_for_local: uses reservoir phase instead of h_c_l phase
    - reservoir update called AFTER routing (uses final s_l)
    - H_c_l update RETAINED for dormancy exemplar reconstruction
    - Routing E_l still uses STATIC mu_c_l (content-driven, no circular dependency)

    Architectural decision (Dr. G + Dr. D): prediction enters via psi_for (contribution),
    NOT via E_l (selection). This preserves interpretable content-driven routing while
    giving units temporally-predictive contribution signals to GAT.
    """
    def __init__(self, bank, layer_idx, lam_p_schedule):
        super().__init__()
        self.bank=bank; self.layer_idx=layer_idx; self.lam_p_schedule=lam_p_schedule
        # log_lam_LG removed v6.0.8 (h_g blend removed with global tier)
        self.log_lambda_hebb =nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_alpha_res   =nn.Parameter(torch.tensor(math.log(0.2)))
        self.log_lam_seq_gat =nn.Parameter(torch.tensor(math.log(0.05)))   # v5.9.8 R3.A H_seq GAT weight
        self._lam_p_correction=1.0
        self._W_ll_cache: dict = {}

    def _get_W_ll(self, bank, sel_l: torch.Tensor) -> torch.Tensor:
        key=tuple(sel_l.sort().values.tolist())   # v5.9.5 H5: full sel_l key (was first 10 only)
        if key in self._W_ll_cache: return self._W_ll_cache[key]
        E_cross=compute_energies(bank.mu_c_l[sel_l],bank.W_l.data[sel_l],bank.mu_c_l[sel_l])
        a_rq_c=rq_routing(E_cross,bank.log_alpha_rq_l[sel_l],bank.log_ell_l[sel_l])
        W_ll=apply_psd_to_weight_matrix((a_rq_c*a_rq_c.T).sqrt().float())
        self._W_ll_cache[key]=W_ll
        if len(self._W_ll_cache)>32: self._W_ll_cache.pop(next(iter(self._W_ll_cache)))
        return W_ll

    def forward(self, x_c, training=True, lam_p=0.1, local_only=False, update_res=True):  # v5.9.5 B5
        B,d_c=x_c.shape; bank=self.bank; n_l=bank.n_l; device=x_c.device
        x_c_mean=x_c.mean(0)   # (d_c,) used in psi_for and reservoir update

        # §1.33 R4: goal-anchored context — update g_c, compute x_c_eff BEFORE routing
        _freeze_goal = getattr(bank, '_in_hypo_mode', False) or getattr(bank, '_goal_frozen', False)
        if not _freeze_goal:
            g_t = torch.sigmoid(bank.W_goal_detect @ x_c_mean.real.unsqueeze(-1)).squeeze()  # scalar
            with torch.no_grad():
                bank.g_c = g_t * x_c_mean + (1.0 - g_t) * bank.g_c
        x_c_eff = x_c + torch.exp(bank.log_lam_goal) * bank.g_c.unsqueeze(0)  # (B, d_c)

        # ── ROUTING: 2-tier (local + persistent) v6.0.8 ─────────────────────
        E_l =compute_energies(x_c_eff,bank.W_l.data[:n_l],bank.mu_c_l[:n_l])
        a_rq=rq_routing(E_l,bank.log_alpha_rq_l[:n_l],bank.log_ell_l[:n_l])
        s_l =entmax15_with_floor(a_rq*torch.exp(bank.log_alp_l[:n_l]).unsqueeze(0),1e-4)
        if not local_only:
            E_p=compute_energies(x_c_eff,bank.W_p.data,bank.mu_c_p)
            s_p=torch.softmax(
                torch.exp(-E_p/torch.exp(2*bank.log_ell_p))*torch.exp(bank.log_alp_p),dim=-1)

        _u_cal = float(getattr(bank, '_u_epistemic_last', 0.5))
        _k_min = int(getattr(bank, 'k_l_min', 20))
        _k_max = int(getattr(bank, 'k_l_max', 60))
        k_l = min(n_l, _k_min + round((_k_max - _k_min) * _u_cal))
        _,sel_l=torch.topk(s_l.mean(0),k_l)
        # v6.0.7 MC-3: U_temporal — representation drift rate
        x_c_mean_d=x_c_mean.detach()
        if bank._x_c_prev_bank.shape==x_c_mean_d.shape:
            delta_t=(x_c_mean_d-bank._x_c_prev_bank).norm()/(bank._x_c_prev_bank.norm().clamp(1e-8))
            bank._ema_delta_bank=0.95*bank._ema_delta_bank+0.05*delta_t
            _u_temporal_val=float(torch.sigmoid(2.0*(delta_t/(bank._ema_delta_bank.clamp(1e-8))-1.0)).item())
        else:
            _u_temporal_val=0.0
        with torch.no_grad(): bank._x_c_prev_bank=x_c_mean_d.clone()

        # ── PSI_FOR: local tier uses PREDICTIVE prototype + RESERVOIR phase ──
        # (Decision D1: prediction enters via contribution, not selection)
        def psi_for_local_rc(sel: torch.Tensor) -> torch.Tensor:
            """v5.9.4: expansion center = mu_pred_i. v5.9.7 C1: phase = H_c_l (stable, not reservoir which scrambles)."""
            mu_pred = bank.get_psi_expansion(sel)              # (k_l, d_c)
            W_s     = bank.W_l.data[sel]                       # (k_l, d_e_l, d_c)
            delta   = x_c_mean.unsqueeze(0) - mu_pred          # (k_l, d_c): prediction error
            proj    = torch.einsum('ned,nd->ne', W_s, delta)   # (k_l, d_e_l)
            ph      = torch.exp(1j*torch.angle(bank.h_c_l[sel].mean(-1)))  # v5.9.7 C1: mean over d_e_l → (k_l,) scalar phase per unit
            # Return: W^H(ph * proj) + mu_pred (unit reports FROM predicted position)
            return (torch.einsum('ned,ne->nd',
                                  W_s.resolve_conj(),
                                  ph.unsqueeze(-1)*proj)
                    + mu_pred)                                  # (k_l, d_c)

        def psi_for_static(W_b, mu_b, sel) -> torch.Tensor:
            """Persistent tier: standard psi_for (no RC). v6.0.8: global tier removed."""
            mu_s=mu_b[sel]; W_s=W_b[sel]; delta=x_c_mean.unsqueeze(0)-mu_s
            proj=torch.einsum('ned,nd->ne',W_s,delta)
            return torch.einsum('ned,ne->nd',W_s.resolve_conj(),proj)+mu_s

        # Build psi_all: local only (v6.0.8 — global tier removed, CS-GAT k²=1,600 not 10,816)
        psi_l=psi_for_local_rc(sel_l)
        psi_all=psi_l                                          # (k_l, d_c)

        mu_all=bank.mu_c_l[sel_l]
        theta_all=compute_direction_angles_complex(mu_all)

        # ── OVERLAP GRAPH + HEBBIAN ───────────────────────────────────────────
        W_ll=self._get_W_ll(bank,sel_l)
        W_full=W_ll                                            # v6.0.8: no global tier block
        lam_h=torch.exp(self.log_lambda_hebb).clamp(max=0.5)   # v6.0.2 C4: grad flows; v6.0.4 C3: bounded ≤0.5
        H_mat=bank.coact_register.get_hebbian_matrix(sel_l.cpu()).to(device)
        W_ll2=W_full[:k_l,:k_l]+lam_h*H_mat[:k_l,:k_l]
        mx=W_full[:k_l,:k_l].max().clamp(1e-8); W_full=W_full.clone()
        W_full[:k_l,:k_l]=W_ll2/W_ll2.max().clamp(1e-8)*mx
        # v5.9.8 R3.A: augment W_full with sequential Hebbian H_seq
        K_h=bank.K_hebb; sel_k=sel_l%K_h
        H_seq_sub=bank.H_seq_mat[sel_k][:,sel_k]   # (k_l,k_l) — pure GPU indexing
        lam_sg=torch.exp(self.log_lam_seq_gat).clamp(max=0.5)   # v6.0.2 C3: grad flows; v6.0.4 C3: bounded ≤0.5
        H_seq_norm=H_seq_sub*(mx/H_seq_sub.max().clamp(1e-8))
        W_full[:k_l,:k_l]=W_full[:k_l,:k_l]+lam_sg*H_seq_norm
        # §1.31 R2: relational eigenvector for dual-key ARC (requires sel_k, only available here)
        try:
            _,_eigvecs=torch.linalg.eigh(H_seq_sub.real.float())
            bank._phi_rel_cache=_eigvecs[:,-1].detach().to(torch.cfloat)  # (k_l,)
        except Exception:
            bank._phi_rel_cache=None

        # §1.43 SE-1 / §1.70 C7: k-shot centroid refinement for young units
        K_proto_max = bank.__dict__.get('K_proto_max', 10) or 10
        tau_proto_min = float(getattr(bank, 'tau_proto_min', 0.4))  # §1.70 C7: U_epi gate threshold
        alpha_young = 0.1
        u_epi_cal = float(getattr(bank, '_u_epistemic_last', 1.0))  # §1.70: gate on calibrated U_epi
        with torch.no_grad():
            for _i, _uid in enumerate(sel_l.tolist()):
                freq = float(bank.activation_freq_l[_uid].item())
                cnt  = int(bank._proto_count[_uid].item())
                if freq < alpha_young and cnt < K_proto_max and u_epi_cal < tau_proto_min:
                    # cosine similarity guard (threshold 0.4 per §1.70)
                    _xm = x_c_mean / x_c_mean.norm().clamp(1e-8)
                    _mu = bank.mu_c_l[_uid] / bank.mu_c_l[_uid].norm().clamp(1e-8)
                    if float((_xm.conj() * _mu).real.sum().item()) > 0.4:
                        bank._proto_count[_uid] += 1
                        bank._proto_sum.data[_uid] += x_c_mean
                        bank.mu_c_l.data[_uid] = bank._proto_sum[_uid] / float(bank._proto_count[_uid].item())

        # §1.30 R1 + §1.41 W2: phase binding kernel B_bind (PSD by Bochner's theorem)
        phi_sel = torch.angle(bank.H_c_l[sel_l].mean(-1).mean(-1))  # (k_l,) real phases
        phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)       # (k_l, k_l)
        sigma_sq_bind = torch.exp(2.0 * bank.log_sigma_bind)
        B_bind = torch.exp(-phi_diff**2 / sigma_sq_bind.clamp(1e-6)).float()  # (k_l, k_l)
        W_full[:k_l,:k_l] = W_full[:k_l,:k_l] + torch.exp(bank.log_lam_bind) * B_bind.to(W_full.dtype)

        # §1.35 X: role binding B_role = α @ α.T (PSD by construction)
        alpha_role = torch.softmax(
            (bank.mu_c_l[sel_l] @ bank.role_vecs.conj().T).real / (d_c**0.5), dim=-1
        )  # (k_l, R)
        B_role = (alpha_role @ alpha_role.T).float()  # (k_l, k_l)
        W_full[:k_l,:k_l] = W_full[:k_l,:k_l] + torch.exp(bank.log_lam_role) * B_role.to(W_full.dtype)

        # §1.55 COMP-H: Hadamard composition B_comp = B_bind ⊙ B_role (PSD by Schur product theorem)
        B_comp = B_bind * B_role  # (k_l, k_l)
        W_full[:k_l,:k_l] = W_full[:k_l,:k_l] + torch.exp(bank.log_lam_composition) * B_comp.to(W_full.dtype)

        # ── GAT AGGREGATION (k_l=40 only; 6.76× cheaper than k_l+k_g=104) ──
        h_filt=bank.gat(psi_all,theta_all,W_full)
        h_l=(s_l[:,sel_l].to(torch.cfloat).unsqueeze(-1)*h_filt[:k_l].unsqueeze(0)).sum(1)
        if not local_only:
            delta_p=x_c.unsqueeze(1)-bank.mu_c_p.unsqueeze(0)
            z_p=torch.einsum('ned,bnd->bne',bank.W_p.data,delta_p)
            proj_p=torch.einsum('ned,bne->bnd',bank.W_p.data.resolve_conj(),z_p)+bank.mu_c_p.unsqueeze(0)
            h_p=(s_p.to(torch.cfloat).unsqueeze(-1)*proj_p).sum(1)
        else: h_p=torch.zeros(B,d_c,dtype=torch.cfloat,device=device)

        if local_only: x_out=h_l
        else:
            lam_p_eff=lam_p*self._lam_p_correction
            x_out=(h_l+lam_p_eff*h_p)/(1.0+lam_p_eff)        # v6.0.8: removed lam_LG*h_g

        # ── NODE RESERVOIR UPDATE (v5.9.4/v5.9.5) ──────────────────────────────
        if update_res:
            bank.update_reservoir(x_c_mean.detach(), s_l.detach(), sel_l,
                                    salience_gate=getattr(bank,'_last_salience',1.0))  # v5.9.6 I2

        # ── H_c_l UPDATE (dormancy exemplar + psi_for phase injection) ──────
        with torch.no_grad():
            prs=torch.einsum('ned,nd->ne',bank.W_l.data[sel_l],
                              x_c_mean.unsqueeze(0)-bank.mu_c_l[sel_l]).detach() # (k_l,d_e_l)
            if training:
                bank.H_c_l[sel_l]=torch.roll(bank.H_c_l[sel_l],-1,dims=-1)
                bank.H_c_l[sel_l,:,-1]=prs
            bank.h_c_l[sel_l]=prs

        Z_val=float(s_l.sum(-1).mean().item()); U_val=1.0/(1.0+Z_val)
        u_epi=bank.compute_u_epistemic(E_l.detach(),s_l.detach())
        if update_res and training and bank._prev_sel_l is not None:
            bank.update_sequential_hebbian(bank._prev_sel_l,sel_l)
        if update_res: bank._prev_sel_l=sel_l.detach()
        info={'s_l':s_l.detach(),'E_l':E_l.detach(),'a_l_rq':a_rq.detach(),
               'alp_l':torch.exp(bank.log_alp_l[:n_l]).detach(),'sel_l':sel_l,'B':B,
               'U_epistemic':u_epi,
               'Z_val':Z_val}
        return x_out,Z_val,U_val,info
