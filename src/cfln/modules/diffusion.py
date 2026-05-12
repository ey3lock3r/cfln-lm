import math
import torch
import torch.nn as nn

from cfln.utils import complex_layer_norm, init_unitary  # noqa: F401
from cfln.modules.v9_ops import compute_Q_beam  # noqa: F401 — used after lista_forward additions

SIGMA_DATA=math.sqrt(2); LAMBDA_MAX=100.0

def cosine_schedule(T,device=None):
    t=torch.arange(T+1,dtype=torch.float32,device=device)
    ab=torch.cos((t/T)*(math.pi/2))**2; ab=ab/ab[0]
    return ab,(1-ab[1:]/ab[:-1]).clamp(max=0.999)

def t_to_sigma(t,alpha_bar): ab=alpha_bar[t].float(); return ((1-ab)/ab.clamp(1e-8)).sqrt()

def q_sample(x0,t,alpha_bar):
    ab=alpha_bar[t].float()
    eps=torch.complex(torch.randn_like(x0.real)/math.sqrt(2),torch.randn_like(x0.imag)/math.sqrt(2))
    return ab.sqrt().unsqueeze(-1)*x0+(1-ab).sqrt().unsqueeze(-1)*eps,eps

def continuous_noise_conditioning(sigma_t,n_fourier=32):
    logs=torch.log(sigma_t.float().clamp(1e-8))
    f=torch.arange(1,n_fourier+1,dtype=torch.float32,device=sigma_t.device)*(math.pi/n_fourier)
    return torch.cat([torch.sin(logs.unsqueeze(-1)*f),torch.cos(logs.unsqueeze(-1)*f)],dim=-1)

def edm_precondition_complex(x_t,sigma_t,F_theta,sd=SIGMA_DATA):
    s=sigma_t.float(); c_skip=(sd**2/(s**2+sd**2)).unsqueeze(-1)
    c_out=(s*sd/(s**2+sd**2).sqrt()).unsqueeze(-1); c_in=(1.0/(s**2+sd**2).sqrt()).unsqueeze(-1)
    return c_skip*x_t+c_out*F_theta(c_in*x_t,0.25*torch.log(s.clamp(1e-8)))

def edm_loss_weight(sigma_t,sd=SIGMA_DATA,lmax=LAMBDA_MAX):
    return (1/sd**2+1/sigma_t.float().clamp(1e-4)**2).clamp(max=lmax)

def complex_soft_threshold(z,tau): m=z.abs().clamp(1e-8); return z*(m-tau).clamp(0)/m


class ComplexUnitaryDenoisingNet(nn.Module):
    """
    LISTA-extended CUN. v5.9.6 additions over v5.9.5:
    - Per-sequence r_lista: expands to (B, d_r_lista) inside lista_forward (I7)
    - U_meta gate: beta_eff = beta_rs * max(0.1, 1-0.7*U_meta_prev) (I1)
    - _prev_U_meta: float attribute storing previous U_meta for gate computation
    Backward compatible: r_lista=0 and _prev_U_meta=0 gives h_0=0 (v5.9.3 behavior).
    """
    def __init__(self,d_c,n_fourier=32,N_iter=8,rho_max=0.95,
                 delta_stuck=0.1,delta_min=0.01,epsilon_esc=0.05,
                 d_r_lista=None,rho_lista=0.99,
                 sparse_code_cache_K=32,episodic_rule_n=16,   # v5.9.8 new
                 lista_min_ratio=0.25,lista_convergence_ratio=0.5):
        super().__init__()
        self.d_c=d_c; self.N_iter=N_iter; self.rho_max=rho_max
        self.delta_stuck=delta_stuck; self.delta_min=delta_min; self.epsilon_esc=epsilon_esc
        self.n_fourier=n_fourier
        d_r_lista=d_r_lista or d_c//2; self.d_r_lista=d_r_lista

        # Existing parameters (unchanged)
        self.U1         =nn.Parameter(init_unitary(d_c))
        self.U2         =nn.Parameter(init_unitary(d_c))
        self.log_thresh =nn.Parameter(torch.zeros(d_c))
        self.noise_proj =nn.Linear(2*n_fourier,d_c); nn.init.normal_(self.noise_proj.weight,std=0.01)
        self.S               =nn.Parameter(torch.zeros(d_c,d_c,dtype=torch.cfloat))
        self.log_tau_schedule=nn.Parameter(torch.zeros(N_iter))
        self.log_gamma_raw   =nn.Parameter(torch.tensor(0.0))
        self.log_s_scale     =nn.Parameter(torch.tensor(0.0))
        self.w_conv          =nn.Parameter(torch.tensor(0.5))

        # LISTA WARM-START RESERVOIR (v5.9.4/v5.9.5)
        # W_ri: FIXED random buffer (v5.9.5 B4 — had zero gradient: r_lista always detached)
        # ESN design: fixed W_in, trained readout W_rs. Classic and provably correct.
        W_ri_init=((torch.randn(d_r_lista,d_c)+1j*torch.randn(d_r_lista,d_c)).to(torch.cfloat)/d_c**0.5)
        self.register_buffer('W_ri',W_ri_init)                     # (d_r, d_c) FIXED
        self.W_rs=nn.Parameter(                                    # (d_c, d_r): TRAINED readout
            (torch.randn(d_c,d_r_lista)+1j*torch.randn(d_c,d_r_lista)).to(torch.cfloat)/d_r_lista**0.5)
        self.log_beta_rs=nn.Parameter(torch.tensor(-3.0))          # scale sigmoid(-3)≈0.047 initially
        self.log_hop_blend=nn.Parameter(torch.tensor(0.0))          # v5.9.7 C2: blend(temporal,content) sigmoid(0)=0.5
        k_idx=torch.arange(d_r_lista,dtype=torch.float32)
        lambda_lista=(rho_lista*torch.exp(1j*2*math.pi*k_idx/d_r_lista)).to(torch.cfloat)
        self.register_buffer('lambda_lista',lambda_lista)          # (d_r,) FIXED Fourier eigenvalues
        self.register_buffer('r_lista',torch.zeros(d_r_lista,dtype=torch.cfloat))  # session state
        self._prev_U_meta: float = 0.0   # v5.9.6 I1: previous U_meta for warm start gate
        # v5.9.8 R1.B: Sparse code cache (shift-buffer, K most recent h_N)
        self.sparse_code_cache_K=sparse_code_cache_K
        if sparse_code_cache_K>0:
            self.register_buffer('h_cache',torch.zeros(sparse_code_cache_K,d_c,dtype=torch.cfloat))
            self._cache_filled: int = 0
            self.W_cache_gate=nn.Parameter(
                torch.zeros(d_c,dtype=torch.cfloat))  # (d_c,) gate direction
            self.log_cache_gate_bias=nn.Parameter(torch.tensor(-2.0))
        # v6.0.7 NR-3: learned rule cache gate
        self.log_gate_rule=nn.Parameter(torch.tensor(-2.0))   # scalar gate bias
        self.W_gate_rule=nn.Parameter(torch.zeros(d_c))  # (d_c,) real  # v6.0.2 M6: init sigmoid≈0.12
        # v5.9.8 R3.B: Episodic rule cache (ring buffer, N_rules recent successful inferences)
        self.episodic_rule_n=episodic_rule_n
        if episodic_rule_n>0:
            # §1.31 R2: dual-key ARC — rule_K stores [k_concept | k_rel] = 2*d_c
            self.register_buffer('rule_K',torch.zeros(episodic_rule_n,2*d_c,dtype=torch.cfloat))
            self.register_buffer('rule_V',torch.zeros(episodic_rule_n,d_c,dtype=torch.cfloat))
            self.register_buffer('rule_ptr',torch.zeros(1,dtype=torch.long))
            self.register_buffer('rule_util',torch.zeros(episodic_rule_n))  # v6.0.6 ARC: utility score
            self.register_buffer('rule_n',  torch.zeros(1,dtype=torch.long))     # filled count
            self._rule_cache_n: int = 0
        # §1.34: U_meta_v4 — five signals; log_w_meta kept for init then replaced by precision (§1.58)
        # §1.58 precision replaces log_w_meta at runtime; log_w_meta removed from __init__
        # sigma_sq_buffer and log_precision are plain Python lists (not nn.Parameter)
        self.sigma_sq_buffer = [1.0, 1.0, 1.0, 1.0, 1.0]    # §1.58: init 1.0 NOT 0.0
        self.log_precision   = [0.0, 0.0, 0.0, 0.0, 0.0]    # §1.58: equal weights initially
        self._precision_active = [False, False, False, False, False]  # §1.58: gate inactive pathways
        # v5.9.8 R1.A: Adaptive LISTA depth parameters
        self.lista_min_ratio=lista_min_ratio
        self.lista_convergence_ratio=lista_convergence_ratio
        self._in_thinking_mode=False
        self._x_c_prev=None          # v6.0.7 MC-3: U_temporal prev representation
        self._ema_delta=0.0          # v6.0.7 MC-3: running mean of delta_t

        # ── v9.0 / v7.0 / v8.0 additions ────────────────────────────────────
        # §1.40 W1: STELA smooth threshold
        self.tau_smooth = nn.Parameter(torch.tensor(0.1))
        # §1.31 R2: dual-key ARC relational key cache
        self._phi_rel_cache = None
        self._phi_rel_step  = 0
        # §1.31 R2: dual-key mixing weight
        self.log_alpha_arc = nn.Parameter(torch.tensor(0.0))
        # §1.47 TS-1: beam search perturbation scale
        self.eps_beam_scale = nn.Parameter(torch.tensor(0.1))
        # §1.46 Q-BEAM: optional learned beam weights (3 core signals)
        self.log_w_beam = nn.Parameter(torch.zeros(3))
        # §1.39 Z + §1.52 PLAN-B: SSP goal stack and Lyapunov tracking
        self._goal_stack  = []
        self._stuck_count = []
        self._v_prev      = []
        # §1.65 C3: last Q_BEAM score for adaptive SSP merge
        self._last_Q_BEAM_score = 0.0
        # §1.73 C10: learned r_lista blend alpha
        self.log_blend_alpha = nn.Parameter(torch.tensor(math.log(0.8)))

    def reset_lista_reservoir(self):
        """Reset session reservoir. Called at begin_document and reset_for_inference."""
        with torch.no_grad(): self.r_lista.zero_()
        self._prev_U_meta = 0.0
        self._seq_mode = True
        # v6.0.6: reset ARC utility scores at session boundary
        if hasattr(self,'rule_util'): self.rule_util.zero_()
        if hasattr(self,'rule_n'):    self.rule_n.zero_()
        # v5.9.8: reset sparse code cache and episodic rule cache
        if self.sparse_code_cache_K>0:
            with torch.no_grad(): self.h_cache.zero_()
            self._cache_filled=0
        if self.episodic_rule_n>0:
            with torch.no_grad(): self.rule_K.zero_(); self.rule_V.zero_(); self.rule_ptr.zero_()
            self._rule_cache_n=0
        self._in_thinking_mode=False
        self._x_c_prev=None
        self._ema_delta=0.0
        # §1.58: reset precision tracking per session (log_precision NOT reset — long-term)
        self.sigma_sq_buffer   = [1.0, 1.0, 1.0, 1.0, 1.0]
        self._precision_active = [False, False, False, False, False]
        # §1.39 Z + §1.52 PLAN-B: clear SSP stacks
        self._goal_stack  = []
        self._stuck_count = []
        self._v_prev      = []
        # §1.32 R3: clear HYPO state (also cleared on bank)
        self._phi_rel_cache = None
        self._phi_rel_step  = 0

    def init_S_from_unitaries(self):
        with torch.no_grad():
            self.S.data.copy_(torch.eye(self.d_c,dtype=torch.cfloat,device=self.U1.device)
                               -self.U2.data@self.U1.data.conj().T)

    def _S_effective(self) -> torch.Tensor:
        s=torch.exp(self.log_s_scale).clamp(0.01,2.0); S=self.S*s
        with torch.no_grad():
            v=torch.randn(self.d_c,dtype=torch.cfloat,device=self.S.device); v=v/v.norm().clamp(1e-8)
            for _ in range(5): v=S@v; v=v/v.norm().clamp(1e-8)
            sv=(S@v).norm()
        return S*(self.rho_max/sv.clamp(1e-8)).clamp(max=1.0)

    def _tau_k(self,k: int) -> torch.Tensor:
        gamma=torch.sigmoid(self.log_gamma_raw).clamp(min=0.1)
        base =torch.exp(self.log_thresh).clamp(1e-3)
        per_k=torch.exp(self.log_tau_schedule[k]).clamp(0.1,10.0)
        return (base*per_k*(gamma**k)).clamp(min=1e-3)

    def lista_forward(self,x_c,hopfield=None,bank=None,N_hop=4,
                       escape=True,compute_meta: bool=True,u_temporal: float=0.0,
                       u_hypo: float=0.0,r_lista_goal=None,
                       E_min_raw=None,H_route_raw=None):
        B,d_c=x_c.shape; dev=x_c.device
        S_eff=self._S_effective(); x_proj=x_c@self.U1.conj().T
        # §1.31 R2: local alias — bank._phi_rel_cache written by CFL5Layer, None when bank absent
        _phi_rel=getattr(bank,'_phi_rel_cache',None) if bank is not None else None

        # WARM START (v5.9.6): per-sequence r_lista + U_meta gate (I1+I7)
        # §1.32 R3: HYPO mode uses branched r_lista_hypo instead of r_lista
        _r_src = (bank._r_lista_hypo if (bank is not None and getattr(bank, '_in_hypo_mode', False)
                                         and bank._r_lista_hypo is not None)
                  else self.r_lista)
        # I7: expand r_lista to (B, d_r_lista) for per-sequence warm start
        r_lista_B = _r_src.unsqueeze(0).expand(B, -1).detach()   # (B, d_r)
        # I1: gate beta_rs by previous U_meta — poor prior reasoning → trust warm start less
        u_prev    = getattr(self, '_prev_U_meta', 0.0)
        beta_seq  = (torch.sigmoid(self.log_beta_rs)
                     if getattr(self,'_seq_mode',True) else self.log_beta_rs.new_zeros(1).squeeze())
        beta_rs   = beta_seq * max(0.1, 1.0 - 0.7 * float(u_prev))   # floor 0.1, suppress 70%
        warm      = r_lista_B @ self.W_rs.conj().T                     # (B, d_c) per-sequence
        h         = beta_rs * warm                                     # (B, d_c)
        # v5.9.7 C2: blend temporal warm start with Hopfield content warm start
        # Content warm start: always in correct basin (content-addressed via nearest prototype)
        # Temporal warm start: provides specific context for coherent text
        # Together: robust to topic shifts AND maintains reasoning continuity
        if hopfield is not None and bank is not None:
            with torch.no_grad():
                x_hop=hopfield.forward(x_c.detach(),bank.mu_c_l[:bank.n_l])  # (B,d_c)
                h_hop=(x_hop@self.U1.conj().T.detach()).detach()               # LISTA space
            alpha_b=torch.sigmoid(self.log_hop_blend)           # learned blend ratio
            h=alpha_b*h+(1.0-alpha_b)*h_hop                     # blend temporal + content
        # When r_lista=0 and hopfield=None: h=0 → identical to v5.9.3 ✓

        # v6.0.3 H2: self-healing _cache_filled repair after checkpoint load
        # h_cache is a register_buffer (serialized) but _cache_filled is a plain int (not serialized)
        # After load_state_dict, _cache_filled=0 but h_cache may have valid data → repair it
        if (self.sparse_code_cache_K>0 and self._cache_filled==0
                and self.h_cache.abs().sum()>0):
            self._cache_filled=int((self.h_cache.abs().sum(-1)>0).sum().item())
        if (self.episodic_rule_n>0 and self._rule_cache_n==0
                and self.rule_K.abs().sum()>0):
            self._rule_cache_n=int((self.rule_K.abs().sum(-1)>0).sum().item())

        # v5.9.8 R1.A: Adaptive LISTA depth
        u_prev_f=float(self._prev_U_meta)
        N_max=getattr(self,'N_iter_override',None) or self.N_iter  # v5.9.9 DCG+ scratchpad
        N_min=max(2,int(N_max*self.lista_min_ratio))
        N_adaptive=int(min(max(N_min+int((N_max-N_min)*u_prev_f),N_min),N_max))
        conv_thr=self.delta_stuck*self.lista_convergence_ratio   # early-exit threshold

        # v6.0.9: rule_util per-token decay (prevents unbounded accumulation)
        n_r_cur=self._rule_cache_n
        if n_r_cur>0:
            _u_temp=float(getattr(self,'_last_u_temporal',0.0))
            _decay=0.999999*(1.0-0.0001*_u_temp)
            self.rule_util[:n_r_cur].mul_(_decay).clamp_(max=100.0)
        # §1.31 R2 + NR-2/NR-3: dual-key ARC retrieval
        if self.episodic_rule_n>0 and self._rule_cache_n>0:
            n_r=self._rule_cache_n
            K_r=self.rule_K[:n_r]                           # (n_r, 2*d_c)
            V_r=self.rule_V[:n_r]                           # (n_r, d_c)
            x_query=(x_c.mean(0)@self.U1.conj().T.detach()) # (d_c,) concept query
            K_concept=K_r[:,:d_c]; K_rel=K_r[:,d_c:]        # split dual key
            # Concept similarity (original)
            sim_con=(x_query@K_concept.conj().T).real/(
                x_query.norm().clamp(1e-8)*K_concept.norm(dim=-1).clamp(1e-8)+1e-8)
            # Relational similarity (§1.31): use cached phi_rel if available
            if _phi_rel is not None:
                phi_flat = _phi_rel.to(x_query.device)
                # k_rel_query = phi_rel @ psi_all[:k_l] approximated by phi_rel mean
                k_rel_query = phi_flat.mean(0) if phi_flat.dim() > 1 else phi_flat
                # Expand to d_c if needed (phi_rel is (k_l,) real; promote to (d_c,) complex)
                if k_rel_query.shape[0] != d_c:
                    k_rel_query = torch.zeros(d_c, dtype=torch.cfloat, device=dev)
                sim_rel=(k_rel_query@K_rel.conj().T).real/(
                    k_rel_query.norm().clamp(1e-8)*K_rel.norm(dim=-1).clamp(1e-8)+1e-8)
            else:
                sim_rel = sim_con  # fallback: use concept sim when no phi_rel
            alpha_arc = torch.sigmoid(self.log_alpha_arc)
            sims = alpha_arc * sim_con + (1.0 - alpha_arc) * sim_rel
            k_ret=min(3,n_r)
            top_idx=torch.topk(sims,k_ret).indices
            w_k=torch.softmax(sims[top_idx]/0.5,dim=0)
            v_blend=(w_k.to(torch.cfloat).unsqueeze(-1)*V_r[top_idx]).sum(0)
            g_rule=torch.sigmoid(self.log_gate_rule+(self.W_gate_rule*x_query.real).sum())
            h=h+g_rule*v_blend.unsqueeze(0).expand(B,-1).detach()

        # v5.9.8 R1.B: Sparse code cache retrieval
        if self.sparse_code_cache_K>0 and self._cache_filled>0:
            filled=self._cache_filled
            entries=self.h_cache[:filled]              # (filled,d_c)
            xq=x_c.mean(0)                             # (d_c,)
            sims_c=(xq@entries.conj().T).real/(d_c**0.5)           # content sims
            a_c=torch.softmax(sims_c,dim=0)
            recency=torch.linspace(0,1,filled,device=dev)
            a_r=torch.softmax(recency,dim=0)
            w_cache=0.7*a_c+0.3*a_r                   # content+recency blend
            h_ret=(w_cache.to(torch.cfloat).unsqueeze(-1)*entries).sum(0)  # (d_c,)
            gate=torch.sigmoid((self.W_cache_gate.conj()*xq).real.sum()+self.log_cache_gate_bias)  # v6.0.2 M6
            h=h+gate*h_ret.unsqueeze(0).expand(B,-1).detach()

        h_pre_escape=None    # v5.9.8 R3.B: track for rule cache
        _escaped=False; deltas=[]
        for k in range(N_adaptive):
            z  =x_proj+torch.einsum('ij,bj->bi',S_eff,h)
            # §1.40 W1: STELA smooth thresholding (replaces hard complex_soft_threshold)
            tau_k = self._tau_k(k).unsqueeze(0)                       # (1, d_c)
            tau_smooth_c = self.tau_smooth.clamp(min=1e-3)
            h_n = z * torch.sigmoid((z.abs() - tau_k) / tau_smooth_c)
            dk_val=(((h_n-h).abs().norm(dim=-1)/(h.abs().norm(dim=-1)+1e-8)).mean())
            deltas.append(dk_val)
            # v5.9.8 R1.A: early exit if converged (skip during/after escape phase)
            if not _escaped and k>=N_min and float(dk_val)<conv_thr: break
            if hopfield is not None and bank is not None and k>0 and k%N_hop==0:
                x_cur=h_n@self.U2.conj().T; x_comp=hopfield(x_cur,bank.mu_c_l[:bank.n_l])
                h_n=x_comp@self.U1.conj().T
            if (escape and not _escaped and k>=2
                    and float(dk_val)>self.delta_stuck
                    and h_n.abs().norm(dim=-1).mean().item()>self.delta_min):
                h_pre_escape=h_n.detach().clone()  # v5.9.8 R3.B: save pre-escape state
                raw_sig=float(self.epsilon_esc*h_n.abs().norm(dim=-1).mean())
                sig=max(min(raw_sig,float(SIGMA_DATA)*5.0),float(SIGMA_DATA)*0.01)
                noise=sig*torch.complex(torch.randn_like(h_n.real)/math.sqrt(2),
                                         torch.randn_like(h_n.imag)/math.sqrt(2))
                x_ns=(h_n+noise)@self.U2.conj().T; st=torch.full((B,),sig,device=dev)
                x_esc=edm_precondition_complex(x_ns,st,self); h_n=x_esc@self.U1.conj().T; _escaped=True
            h=h_n
        h_N = h   # converged sparse code (B, d_c)

        # §1.45 SE-3: reservoir-augmented reconstruction (uses bank._prev_sel_l)
        # x_c_recon = U2 @ h_N + W_dec_res @ rho_sel (no new parameters)
        if bank is not None and bank._prev_sel_l is not None:
            _sel_se3 = bank._prev_sel_l
            rho_sel_se3 = bank.rho_l[_sel_se3].mean(0)  # (d_r_node,)
            # Store for reconstruction in train_step; also usable by IterativeRefinement
            self._last_rho_sel = rho_sel_se3.detach()
        else:
            self._last_rho_sel = None

        # §1.47 TS-1: beam search during think mode (adaptive width via §1.66)
        in_think = getattr(self, '_in_thinking_mode', False)
        u_prev_for_beam = float(getattr(self, '_prev_U_meta', 0.0))
        _B_max = int(getattr(self, 'beam_B_max', 2))
        B_eff = max(1, round(1 + u_prev_for_beam * (_B_max - 1))) if in_think else 1
        if B_eff > 1:
            noise = (torch.randn_like(h_N.real) + 1j*torch.randn_like(h_N.imag)).to(torch.cfloat)
            r_lista_b2 = _r_src + self.eps_beam_scale.abs() * (self.W_ri @ noise.mean(0))
            r_lista_B2 = r_lista_b2.unsqueeze(0).expand(B, -1).detach()
            warm2 = r_lista_B2 @ self.W_rs.conj().T
            h_b2 = beta_rs * warm2
            x_proj2 = x_c @ self.U1.conj().T
            for _k2 in range(N_adaptive):
                z2 = x_proj2 + torch.einsum('ij,bj->bi', S_eff, h_b2)
                h_b2 = z2 * torch.sigmoid((z2.abs() - self._tau_k(_k2).unsqueeze(0)) / tau_smooth_c)
            Q_b1 = compute_Q_beam(h_N.mean(0), _r_src, r_lista_goal,
                                  self._goal_stack, x_c,
                                  log_w_beam=self.log_w_beam,
                                  phi_rel=_phi_rel,
                                  E_min_raw=E_min_raw, H_route_raw=H_route_raw)
            Q_b2 = compute_Q_beam(h_b2.mean(0), r_lista_b2, r_lista_goal,
                                  self._goal_stack, x_c,
                                  log_w_beam=self.log_w_beam,
                                  phi_rel=_phi_rel,
                                  E_min_raw=E_min_raw, H_route_raw=H_route_raw)
            w_beam = torch.softmax(torch.stack([Q_b1, Q_b2]), dim=0)
            # §1.47: diversity computed before blending; r_lista_b2 carries grad via eps_beam_scale
            _beam_div = (_r_src - r_lista_b2).norm()
            h_N = w_beam[0] * h_N + w_beam[1] * h_b2
            with torch.no_grad():
                _r_src = (w_beam[0] * _r_src + w_beam[1] * r_lista_b2).detach()
            self._last_Q_BEAM_score = float(Q_b1.item())  # §1.65 C3: for adaptive SSP merge
            self._last_beam_diversity = _beam_div  # §1.47: Tensor so train_step L_diversity gradient flows

        x_ref = h_N @ self.U2.conj().T

        # v5.9.8 R1.B: Update sparse code cache — skip during thinking (v6.0 CTP)
        if self.sparse_code_cache_K>0 and not self._in_thinking_mode:
            with torch.no_grad():
                Kcache=self.sparse_code_cache_K; h_mean=h_N.mean(0).detach()
                if self._cache_filled<Kcache:
                    self.h_cache[self._cache_filled]=h_mean
                    self._cache_filled+=1
                else:
                    self.h_cache[:-1]=self.h_cache[1:].clone()
                    self.h_cache[-1]=h_mean

        # UPDATE session reservoir — fully detached (no BPTT through r_lista)
        with torch.no_grad():
            e_lista = h_N.mean(0).detach()
            self.r_lista = (self.lambda_lista * self.r_lista + self.W_ri @ e_lista)

        df    =deltas[-1].detach() if isinstance(deltas[-1],torch.Tensor) else torch.tensor(float(deltas[-1]) if deltas else 0.0,device=dev)
        U_conv=1.0-torch.exp(torch.tensor(-5.0,device=dev)*df)
        if compute_meta:
            st_eval=torch.full((B,),0.1,device=dev)
            xp=edm_precondition_complex(x_ref.detach(),st_eval,self)
            residual=((xp-x_ref.detach()).conj()*(xp-x_ref.detach())).real.sum(-1).mean().sqrt()
            U_repr=1.0-torch.exp(torch.tensor(-2.0,device=dev)*residual)
        else: U_repr=torch.tensor(0.0,device=dev)
        w=torch.sigmoid(self.w_conv); _U_meta_base=w*U_conv+(1-w)*U_repr  # noqa: F841 — used via U_repr in precision

        u_hop = float(getattr(hopfield,'_last_confidence',0.0)) if hopfield is not None else 0.0
        u_epi = float(getattr(bank,'_u_epistemic_last',0.0)) if bank is not None else 0.0

        # §1.34 R3+: U_hypo fifth signal
        u_hypo_f = float(u_hypo)
        if (bank is not None and getattr(bank,'_in_hypo_mode',False)
                and bank._r_lista_hypo is not None):
            diff_sq = float((bank._r_lista_hypo - self.r_lista).norm()**2)
            u_hypo_f = float(torch.sigmoid(torch.tensor(diff_sq / max(self.d_r_lista, 1))).item())

        # §1.58: Precision-weighted U_meta (replaces log_w_meta softmax)
        u_temp_f = float(getattr(self,'_last_u_temporal', u_temporal))
        raw_signals = [
            float(U_repr.item()) if isinstance(U_repr, torch.Tensor) else float(U_repr),
            u_epi,
            u_hop,
            u_temp_f,
            u_hypo_f,
        ]
        # Update precision from signal variance (EMA)
        for _s, val in enumerate(raw_signals):
            if abs(val) > 1e-6:
                self._precision_active[_s] = True
            if not self._precision_active[_s]:
                continue
            self.sigma_sq_buffer[_s] = 0.95 * self.sigma_sq_buffer[_s] + 0.05 * val**2
            lp = -0.5 * math.log(self.sigma_sq_buffer[_s] + 1e-6)
            self.log_precision[_s] = max(-3.0, min(3.0, lp))
        prec = torch.exp(torch.tensor(self.log_precision, dtype=torch.float32, device=dev))
        signals_t = torch.tensor(raw_signals, dtype=torch.float32, device=dev)
        U_meta = (prec * signals_t).sum() / (prec.sum() + 1e-8)

        # §1.52 PLAN-B: Lyapunov timeout per SSP depth
        if (in_think and len(self._goal_stack) > 0 and r_lista_goal is not None
                and r_lista_goal.norm() > 1e-4):
            n_stuck = 12  # ssp_stuck_threshold (from cfg when available)
            if len(self._stuck_count) < len(self._goal_stack):
                self._stuck_count.extend([0]*(len(self._goal_stack)-len(self._stuck_count)))
                self._v_prev.extend([1e9]*(len(self._goal_stack)-len(self._v_prev)))
            V_curr = float((self.r_lista - r_lista_goal).norm()**2)
            if V_curr >= self._v_prev[-1]:
                self._stuck_count[-1] += 1
            else:
                self._stuck_count[-1] = 0
            self._v_prev[-1] = V_curr
            if self._stuck_count[-1] >= n_stuck and self._goal_stack:
                parent = self._goal_stack.pop()
                self._stuck_count.pop(); self._v_prev.pop()
                with torch.no_grad(): self.r_lista.copy_(parent)

        # §1.31 R2 + NR-1: dual-key ARC write — SUPPRESSED during HYPO (§1.32)
        _in_hypo = bank is not None and getattr(bank, '_in_hypo_mode', False)
        if self.episodic_rule_n>0 and not self._in_thinking_mode and not _in_hypo:
            U_meta_f = float(U_meta.item()) if isinstance(U_meta, torch.Tensor) else float(U_meta)
            U_epi_f  = float(getattr(bank,'_last_u_epi',0.0) if bank is not None else 0.0)
            trig_A = h_pre_escape is not None and U_meta_f<0.3
            trig_B = U_epi_f>0.6 and U_meta_f<0.4
            if trig_A or trig_B:
                with torch.no_grad():
                    K_concept_new = (x_c.mean(0)@self.U1.conj().T.detach()).detach()
                    # §1.31: relational key from phi_rel cache
                    if _phi_rel is not None:
                        phi_q = _phi_rel
                        k_rel_new = (phi_q.mean(0) if phi_q.dim()>1 else phi_q)
                        if k_rel_new.shape[0] != d_c:
                            k_rel_new = torch.zeros(d_c, dtype=torch.cfloat, device=dev)
                    else:
                        k_rel_new = K_concept_new
                    K_new = torch.cat([K_concept_new, k_rel_new], dim=0)  # (2*d_c,)
                    V_new = h_N.mean(0).detach()
                    n_r   = self._rule_cache_n
                    if n_r>0:
                        K_r_con = self.rule_K[:n_r,:d_c]
                        sims_w = (K_concept_new@K_r_con.conj().T).real/(
                            K_concept_new.norm().clamp(1e-8)*K_r_con.norm(dim=-1).clamp(1e-8)+1e-8)
                        best_sim,best_i = sims_w.max(0)
                        if float(best_sim.item())>0.7:
                            self.rule_K[best_i]=0.7*self.rule_K[best_i]+0.3*K_new
                            self.rule_V[best_i]=0.7*self.rule_V[best_i]+0.3*V_new
                            self.rule_util[best_i]+=0.5
                        else:
                            ptr=(int(self.rule_util[:n_r].argmin().item()) if n_r>=self.episodic_rule_n else n_r)
                            self.rule_K[ptr]=K_new; self.rule_V[ptr]=V_new
                            self.rule_util[ptr]=0.0
                            self._rule_cache_n=min(self.episodic_rule_n,n_r+1)
                    else:
                        self.rule_K[0]=K_new; self.rule_V[0]=V_new
                        self.rule_util[0]=0.0; self._rule_cache_n=1
                    self.rule_ptr.add_(1)

        self._last_warm_norm = float(warm.norm(dim=-1).mean().item())
        self._last_u_temporal = u_temporal
        self._prev_U_meta = float(U_meta.item()) if isinstance(U_meta,torch.Tensor) else float(U_meta)
        return x_ref, h_N, {'U_conv':U_conv,'U_repr':U_repr,'U_meta':U_meta,
                              'escaped':_escaped,'delta_k':deltas,
                              'warm_start_norm':self._last_warm_norm}

    def forward(self,x_c,c_noise):
        sig=torch.exp(4*c_noise.float().clamp(-10,10))
        four=continuous_noise_conditioning(sig,self.n_fourier); ns=torch.sigmoid(self.noise_proj(four))
        tau=torch.exp(self.log_thresh).unsqueeze(0)*sig.unsqueeze(-1)
        z=x_c@self.U1.conj().T
        return complex_soft_threshold(z,(ns*tau).clamp(0))@self.U2.conj().T


class DiffusionAuxiliaryModule(nn.Module):
    """Path C. v5.9.6: passes all RC params including rho_fast/mid/slow to CUN."""
    def __init__(self,d_c,T_diff=1000,n_fourier=32,lambda_diff_init=0.1,
                 lambda_diff_max=0.5,lambda_loss_max=100.0,N_iter=8,
                 delta_stuck=0.1,delta_min=0.01,epsilon_esc=0.05,
                 d_r_lista=None,rho_lista=0.99,
                 sparse_code_cache_K=32,episodic_rule_n=64,       # v5.9.9: expanded 16→64
                 lista_min_ratio=0.25,lista_convergence_ratio=0.5):
        super().__init__()
        self.cun=ComplexUnitaryDenoisingNet(d_c,n_fourier,N_iter=N_iter,
                                             delta_stuck=delta_stuck,delta_min=delta_min,
                                             epsilon_esc=epsilon_esc,
                                             d_r_lista=d_r_lista,rho_lista=rho_lista,
                                             sparse_code_cache_K=sparse_code_cache_K,   # v5.9.8
                                             episodic_rule_n=episodic_rule_n,
                                             lista_min_ratio=lista_min_ratio,
                                             lista_convergence_ratio=lista_convergence_ratio)
        ab,_=cosine_schedule(T_diff); self.register_buffer('alpha_bar',ab)
        self.log_lambda_diff=nn.Parameter(torch.log(torch.tensor(lambda_diff_init)))
        self.lambda_diff_max=lambda_diff_max; self.lambda_loss_max=lambda_loss_max; self._enabled=False

    def enable(self): self._enabled=True

    def forward(self,x_c,training=True):
        if not self._enabled or not training: return torch.tensor(0.0,device=x_c.device)
        B,_=x_c.shape; t=torch.randint(1,self.alpha_bar.shape[0],(B,),device=x_c.device)
        x_t,_=q_sample(x_c,t,self.alpha_bar.to(x_c.device))
        sigma_t=t_to_sigma(t,self.alpha_bar.to(x_c.device))
        x_pred=edm_precondition_complex(x_t,sigma_t,self.cun)
        lam=edm_loss_weight(sigma_t,lmax=self.lambda_loss_max)
        diff=x_pred-x_c.detach()
        loss=(lam*((diff.conj()*diff).real.sum(-1))).mean()
        return torch.exp(self.log_lambda_diff).clamp(max=self.lambda_diff_max)*loss
