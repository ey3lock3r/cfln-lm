import math
import torch
import torch.nn as nn
from cfln.utils import (init_stiefel, rq_routing, compute_energies,  # noqa: F401
                         entmax15_with_floor, apply_psd_to_weight_matrix)  # noqa: F401
from cfln.modules.coact import CoactivationRegister
from cfln.modules.alpha_hist import AlphaHistogram
from cfln.modules.gat import ComplexGATLayer


class CFBank(nn.Module):
    """
    Three-tier CNEP bank + node Fourier reservoir. v5.9.4.

    v5.9.4 additions:
    - lambda_node: fixed Fourier eigenvalues (d_r_node,) cfloat buffer
    - rho_l: per-unit reservoir states (N_MAX_L, d_r_node) cfloat buffer
    - W_enc_res: shared projection-error -> reservoir encoder (d_r_node, d_e_l) cfloat param
    - W_dec_res: shared reservoir -> prototype-shift decoder (d_c, d_r_node) cfloat param
    - log_scale_l: per-unit temporal scale (N_MAX_L,) float param (init=-3 -> scale~0.05)
    New methods: update_reservoir, get_reservoir_phase, get_psi_expansion, reset_reservoir
    H_c_l/h_c_l RETAINED for dormancy exemplar reconstruction.
    """
    N_MAX_L=16384

    def __init__(self, n_l, n_p, d_c, d_e_l=32, d_e_p=64,  # v6.0.8: n_g, d_e_g removed
                 D_g=8, K_hebb=16, d_r_node=8, rho_node=0.95,
                 n_heads_gat=4, **kwargs):   # v5.9.6: **kwargs for rho_fast/mid/slow
        super().__init__()
        N=self.N_MAX_L
        self.n_l=n_l; self.n_p=n_p  # v6.0.8: self.n_g removed
        self.d_c=d_c; self.d_e_l=d_e_l; self.d_e_p=d_e_p
        self.D_g=D_g; self.alpha_freeze=0.7; self.d_r_node=d_r_node

        # LOCAL TIER
        W_l_init=torch.zeros(N,d_e_l,d_c,dtype=torch.cfloat)
        for i in range(n_l): W_l_init[i]=init_stiefel(d_e_l,d_c)
        self.W_l          =nn.Parameter(W_l_init)
        self.mu_c_l       =nn.Parameter((torch.randn(N,d_c)+1j*torch.randn(N,d_c)).to(torch.cfloat)*0.1)
        self.log_alp_l    =nn.Parameter(torch.zeros(N))
        self.log_alpha_rq_l=nn.Parameter(torch.zeros(N))
        self.log_ell_l    =nn.Parameter(torch.zeros(N))
        self.register_buffer('H_c_l',torch.zeros(N,d_e_l,D_g,dtype=torch.cfloat))
        self.register_buffer('h_c_l',torch.zeros(N,d_e_l,dtype=torch.cfloat))
        self.register_buffer('active_mask_l',torch.zeros(N,dtype=torch.bool))
        self.active_mask_l[:n_l]=True
        self.register_buffer('is_sensory_l',torch.zeros(N,dtype=torch.bool))
        self.register_buffer('activation_freq_l',torch.zeros(N))
        self.register_buffer('sensory_domain_id',torch.full((N,),-1,dtype=torch.long))
        # v5.9.5 D1: mu_c_l_prev as register_buffer (survives checkpoint load)
        self.register_buffer('mu_c_l_prev',torch.zeros(N,d_c,dtype=torch.cfloat))
        self.register_buffer('_has_prev_mu',torch.zeros(1,dtype=torch.bool))

        # NODE FOURIER RESERVOIR (v5.9.4)
        # v5.9.6 I5: multi-scale spectral radii (4 groups of d_r_node//4 dims)
        assert d_r_node % 4 == 0, f"d_r_node={d_r_node} must be divisible by 4 for multi-scale rho"
        g=d_r_node//4
        rho_scales=torch.cat([
            torch.full((g,),kwargs.get('rho_fast',0.85)),   # v5.9.7 M6: ~6 tok (was 0.70=~3 tok, shorter than H_c_l D_g=8)
            torch.full((g,),kwargs.get('rho_mid', 0.90)),   # medium: ~10 tok
            torch.full((g,),rho_node),                      # default: ~20 tok
            torch.full((g,),kwargs.get('rho_slow',0.99)),   # slow: ~100 tok
        ])
        k_idx=torch.arange(d_r_node,dtype=torch.float32)+0.5   # offset for non-trivial phi_0
        lambda_node=(rho_scales*torch.exp(1j*2*math.pi*k_idx/d_r_node)).to(torch.cfloat)
        self.register_buffer('lambda_node',lambda_node)           # (d_r,) FIXED multi-scale
        self.register_buffer('rho_l',torch.zeros(N,d_r_node,dtype=torch.cfloat))  # (N,d_r)
        # Shared encoder/decoder: W_enc_res FIXED random buffer (ESN design v5.9.5 B3)
        # Fixed W_enc_res: had zero gradient (inside @no_grad); fixed random is standard ESN
        W_enc_init=((torch.randn(d_r_node,d_e_l)+1j*torch.randn(d_r_node,d_e_l)).to(torch.cfloat)
                    /d_e_l**0.5)
        self.register_buffer('W_enc_res',W_enc_init)             # (d_r, d_e_l) FIXED
        self.W_dec_res=nn.Parameter(
            (torch.randn(d_c,d_r_node)+1j*torch.randn(d_c,d_r_node)).to(torch.cfloat)
            /d_r_node**0.5)                                       # (d_c, d_r)
        self.log_scale_l=nn.Parameter(torch.full((N,),-3.0))     # (N,) init -> scale~0.05
        # v6.0.6: per-unit spectral frequency filter for node reservoir readout
        self.log_decode_scale=nn.Parameter(torch.zeros(N,d_r_node))  # (N,d_r) init 0 -> uniform weighting

        # GLOBAL TIER REMOVED v6.0.8 — subsumed by alpha_freeze-protected local + persistent softmax.
        # Performance: CS-GAT k^2 drops 10,816->1,600 (6.76x); saves 21% per-token flops. See §1.2.

        # PERSISTENT TIER (lr_persist=1e-6 + SI protection)
        self.W_p      =nn.Parameter(torch.stack([init_stiefel(d_e_p,d_c) for _ in range(n_p)]))
        self.mu_c_p   =nn.Parameter((torch.randn(n_p,d_c)+1j*torch.randn(n_p,d_c)).to(torch.cfloat)*0.1)
        self.log_alp_p=nn.Parameter(torch.zeros(n_p))
        self.log_ell_p=nn.Parameter(torch.zeros(n_p))

        self.gat            =ComplexGATLayer(d_c,n_heads=n_heads_gat)   # v5.9.5 D4
        self.coact_register =CoactivationRegister(N,K_hebb)
        self.alpha_histogram=AlphaHistogram(N)

        self._last_salience=1.0   # v5.9.6 I2: Titans surprise salience gate (set by train_step)
        # v5.9.8 R2.A/R3.A: epistemic uncertainty + sequential Hebbian
        self._u_epistemic_last: float = 0.0   # U_epistemic from last CFL-stack pass
        self._prev_sel_l = None               # sel_l from previous token (for H_seq update)
        self.K_hebb = K_hebb                  # store K_hebb for sequential Hebbian indexing
        self.register_buffer('_e_min_ema', torch.tensor(1.0))
        self.register_buffer('_h_route_ema', torch.tensor(1.0))
        # v6.0.7 MC-1: calibration rolling stats for U_epi_cal normalisation
        self.register_buffer('_u_epi_mu',  torch.tensor(0.5))   # rolling mean of U_epi
        self.register_buffer('_u_epi_var', torch.tensor(0.01))  # rolling variance
        # v6.0.7 MC-3: per-bank x_c prev for U_temporal computation
        self.register_buffer('_x_c_prev_bank', torch.zeros(1,dtype=torch.cfloat))  # shape sentinel
        # First call: shape (1,) != x_c_mean (d_c,) -> else branch (u_temporal=0)
        # After first call: updated to (d_c,) -> U_temporal computed normally
        self.register_buffer('_ema_delta_bank', torch.tensor(1e-6))  # running mean of delta_t
        self.register_buffer('H_seq_mat',
            torch.zeros(K_hebb, K_hebb, dtype=torch.float32))   # (K_hebb,K_hebb) transition counts

    def compute_u_epistemic(self, E_l: 'torch.Tensor', s_l: 'torch.Tensor',
                             alpha: float=2.0) -> float:
        """v5.9.8 R2.A: Epistemic uncertainty from routing energy and entropy.
        v6.0.7 MC-1: Post-hoc calibration normalisation keeps output near [0.35, 0.65].
        E_l: (B,n_l) energies, s_l: (B,n_l) routing weights -> float in [0,1].
        High = token poorly covered by any unit AND routing diffuse = genuinely uncertain.
        """
        n=self.n_l
        active=(s_l[:,:n]>(1.0/max(n,1)))              # (B,n_l) bool
        e_masked=E_l[:,:n].clone()
        e_masked[~active]=1e8                            # mask inactive units
        E_min=e_masked.min(dim=-1).values.mean()         # scalar: avg-over-batch min energy
        s_n=s_l[:,:n].clamp(1e-10)
        H_route=(-(s_n*s_n.log()).sum(-1)).mean()        # scalar: avg routing entropy
        with torch.no_grad():
            self._e_min_ema=0.95*self._e_min_ema+0.05*float(E_min.item())  # v6.0.6: 0.99->0.95
            self._h_route_ema=0.95*self._h_route_ema+0.05*float(H_route.item())
        e_norm=float(E_min.item())/(float(self._e_min_ema.item())+1e-8)
        h_norm=float(H_route.item())/(float(self._h_route_ema.item())+1e-8)
        u_raw=float(torch.sigmoid(torch.tensor(alpha*(e_norm*h_norm-1.0))).item())
        # v6.0.7 MC-1: rolling normalisation -> keeps U_epi near [0.35, 0.65]
        with torch.no_grad():
            # Update rolling mean and variance (W=256 token window via EMA)
            _old_mu=float(self._u_epi_mu.item())           # v6.0.9: store BEFORE update (Welford)
            self._u_epi_mu =0.99*self._u_epi_mu  + 0.01*u_raw
            self._u_epi_var=0.99*self._u_epi_var + 0.01*(u_raw-_old_mu)**2  # use OLD mu
            u_std=float(self._u_epi_var**0.5)+1e-6
        u_cal=float(torch.sigmoid(torch.tensor((u_raw-float(self._u_epi_mu))/u_std*0.15+0.5)).item())
        self._u_epistemic_last=u_cal
        self._last_u_epi=u_cal   # v6.0.7: for NR-1 trigger B in lista_forward
        return u_cal

    @torch.no_grad()
    def update_sequential_hebbian(self, prev_sel: 'torch.Tensor',
                                   curr_sel: 'torch.Tensor',
                                   eta: float=0.01, decay: float=0.005) -> None:
        """v5.9.8 R3.A: Vectorised sequential Hebbian update.
        Increments H_seq_mat[i%K,j%K] for all (i in prev_sel, j in curr_sel) pairs.
        """
        K=self.K_hebb
        # One-hot (K,) vectors for prev and curr selections (modulo K for safety)
        prev_hot=torch.zeros(K,dtype=torch.float32,device=self.H_seq_mat.device)
        curr_hot=torch.zeros(K,dtype=torch.float32,device=self.H_seq_mat.device)
        prev_hot.scatter_(0,(prev_sel%K).clamp(0,K-1),1.0)
        curr_hot.scatter_(0,(curr_sel%K).clamp(0,K-1),1.0)
        self.H_seq_mat.add_(eta*torch.outer(prev_hot,curr_hot))
        self.H_seq_mat.mul_(1.0-decay).clamp_(0.0,1.0)

    def enforce_constraints(self):
        """v6.0.8: clamp local-tier log_alp_l only (log_alp_g/log_kap_g removed)."""
        with torch.no_grad():
            n=self.n_l
            self.log_alp_l.data[:n].clamp_(-5,0)

    # ─── NODE RESERVOIR METHODS ────────────────────────────────────────────────

    @torch.no_grad()
    def update_reservoir(self, x_c_mean: torch.Tensor, s_l: torch.Tensor,
                          sel_l: torch.Tensor, salience_gate: float=1.0):  # v5.9.6 I2
        """
        Update Fourier reservoir for selected units after routing.
        1. Decay all active-slot units first (lambda o rho)
        2. Add projection-error input for units active above threshold
        Called AFTER routing (needs s_l), does NOT affect E_l computation.
        """
        n=self.n_l; eps_act=1.0/max(n,1)
        # Step 1: decay all units in active slots (including inactive this step)
        self.rho_l[:n]=self.lambda_node.unsqueeze(0)*self.rho_l[:n]  # (n,d_r)
        # Step 2: accumulate error signal for active selected units
        s_mean=s_l.mean(0)[sel_l]                        # (k_l,)
        active_mask=(s_mean>eps_act)                     # (k_l,) bool
        if not active_mask.any(): return
        active_local=active_mask.nonzero(as_tuple=True)[0]   # indices into sel_l
        active_units=sel_l[active_local]                      # global unit indices
        # Compute projection error: W_i @ (x - mu_i) for each active unit
        mu_act=self.mu_c_l[active_units]                 # (n_act, d_c)
        W_act =self.W_l.data[active_units]               # (n_act, d_e_l, d_c)
        delta =x_c_mean.unsqueeze(0)-mu_act              # (n_act, d_c)
        proj  =torch.einsum('ned,nd->ne',W_act,delta)    # (n_act, d_e_l)
        e_in  =proj@self.W_enc_res.conj().T              # (n_act, d_r)
        self.rho_l[active_units]+=e_in*salience_gate   # v5.9.6 I2: surprise-weighted trace

    def get_reservoir_phase(self, sel_l: torch.Tensor) -> torch.Tensor:
        """
        Phase from reservoir mean for selected units.
        Returns (k_l,) cfloat unit-magnitude tensor for phase injection in psi_for.
        When rho=0: angle=0, exp(0)=1 -> no phase rotation (backward compatible).
        """
        rho_sel=self.rho_l[sel_l]                        # (k_l, d_r)
        mean_r =rho_sel.mean(dim=-1)                     # (k_l,) complex
        return torch.exp(1j*torch.angle(mean_r))         # (k_l,) unit complex

    def get_psi_expansion(self, sel_l: torch.Tensor) -> torch.Tensor:
        """
        Predicted prototypes for k_l selected units.
        Returns (k_l, d_c) cfloat = mu_c_l[sel] + scale * W_dec @ rho_l[sel].
        When rho=0: returns mu_c_l[sel] (backward compatible with v5.9.3).
        v6.0.6 C1: per-unit spectral frequency filter via log_decode_scale.
        """
        rho_sel =self.rho_l[sel_l]                       # (k_l, d_r)
        # v6.0.6: apply per-unit spectral weighting before W_dec projection
        # log_decode_scale[sel_l]: (k_l, d_r) -> exp gives per-unit frequency weights
        if hasattr(self,'log_decode_scale'):
            freq_w = torch.exp(self.log_decode_scale[sel_l])        # (k_l, d_r) real
            rho_sel = rho_sel * freq_w.to(torch.cfloat)             # (k_l, d_r) cfloat
        delta   =rho_sel@self.W_dec_res.conj().T         # (k_l, d_c)
        scale   =torch.exp(self.log_scale_l[sel_l]).unsqueeze(-1)   # (k_l,1) -> (k_l,d_c)
        return self.mu_c_l[sel_l]+scale*delta            # (k_l, d_c)

    @torch.no_grad()
    def reset_reservoir(self):
        """Full reset. Called at document boundaries and definite domain shifts."""
        self.rho_l.zero_()
        self._prev_sel_l=None   # v6.0.3 C2: prevent cross-document H_seq contamination

    @torch.no_grad()
    def attenuate_reservoir(self, factor: float):
        """v5.9.6 I8: Partial attenuation for moderate domain shifts.
        factor in (0,1): rho_l *= factor. factor=0 = full reset, factor=1 = no change."""
        self.rho_l.mul_(factor)

    # ─── EXISTING METHODS (unchanged) ──────────────────────────────────────────

    @torch.no_grad()
    def update_activation_freq(self, s_l, decay=0.995):
        n=self.n_l; active=(s_l.mean(0)[:n]>1.0/n).float()
        self.activation_freq_l[:n]=(decay*self.activation_freq_l[:n]+(1-decay)*active)
        self.coact_register.update(s_l[:,:n],threshold=1.0/n)

    def update_sensory_mask(self, sensory_fraction=0.15, current_domain_id=-1):
        n=self.n_l; alpha=torch.exp(self.log_alp_l[:n]).clamp(1e-6,1.0)
        self.alpha_histogram.update(alpha)
        af=self.alpha_histogram.get_alpha_freeze(sensory_fraction); self.alpha_freeze=af
        new_s=(alpha>af)&~self.is_sensory_l[:n]; self.is_sensory_l[:n]|=new_s
        self.sensory_domain_id[:n][new_s]=current_domain_id
        already=((self.sensory_domain_id[:n]>=0)
                  &(self.sensory_domain_id[:n]!=current_domain_id)
                  &self.is_sensory_l[:n])
        self.sensory_domain_id[:n][already]=-1
        return int(new_s.sum().item())

    def release_domain_sensory(self, domain_id):
        if domain_id<0: return 0
        n=self.n_l; rel=(self.sensory_domain_id[:n]==domain_id)&self.is_sensory_l[:n]
        self.is_sensory_l[:n][rel]=False; self.sensory_domain_id[:n][rel]=-1
        return int(rel.sum().item())
