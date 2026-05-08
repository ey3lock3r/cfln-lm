import math
import torch
import torch.nn as nn

from cfln.utils import complex_layer_norm, init_unitary

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
            self.register_buffer('rule_K',torch.zeros(episodic_rule_n,d_c,dtype=torch.cfloat))
            self.register_buffer('rule_V',torch.zeros(episodic_rule_n,d_c,dtype=torch.cfloat))
            self.register_buffer('rule_ptr',torch.zeros(1,dtype=torch.long))
            self.register_buffer('rule_util',torch.zeros(episodic_rule_n))  # v6.0.6 ARC: utility score
            self.register_buffer('rule_n',  torch.zeros(1,dtype=torch.long))     # filled count
            self._rule_cache_n: int = 0
        # v5.9.8 R2.A+R2.B: Composite U_meta_v2 weights [representation, epistemic, hopfield]
        # v6.0.7 MC-3: extended to R^4 (added U_temporal); init -2.0 for temporal
        self.log_w_meta=nn.Parameter(torch.tensor([1.0,-1.0,-1.0,-2.0]))  # softmax([1,-1,-1])≈[0.79,0.11,0.11]
        # v5.9.8 R1.A: Adaptive LISTA depth parameters
        self.lista_min_ratio=lista_min_ratio
        self.lista_convergence_ratio=lista_convergence_ratio
        self._in_thinking_mode=False
        self._x_c_prev=None          # v6.0.7 MC-3: U_temporal prev representation
        # Note: bank._u_epi_mu/_u_epi_var deliberately NOT reset — they are global
        # calibration stats that warm-start gracefully across domain changes (v6.0.9 design)
        self._ema_delta=0.0          # v6.0.7 MC-3: running mean of δ_t
        self._log_w_rec=[0.0,0.0,0.0,0.0]  # v6.0.7 MC-2: session recency weights   # v6.0 CTP: gates h_cache and rule_cache writes

    def reset_lista_reservoir(self):
        """Reset session reservoir. Called at begin_document and reset_for_inference."""
        with torch.no_grad(): self.r_lista.zero_()
        self._prev_U_meta = 0.0   # v5.9.6 I1
        self._seq_mode = True      # v5.9.7 H2
        self._log_w_rec=[0.0,0.0,0.0,0.0]  # v6.0.7 MC-2: session recency weights (on CUN, not CFLNModel)
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
        self._x_c_prev=None          # v6.0.7 MC-3: U_temporal prev representation
        # Note: bank._u_epi_mu/_u_epi_var deliberately NOT reset — they are global
        # calibration stats that warm-start gracefully across domain changes (v6.0.9 design)
        self._ema_delta=0.0          # v6.0.7 MC-3: running mean of δ_t
        self._log_w_rec=[0.0,0.0,0.0,0.0]  # v6.0.7 MC-2: session recency weights   # v6.0 CTP: always reset to normal mode

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
                       escape=True,compute_meta: bool=True,u_temporal: float=0.0):
        B,d_c=x_c.shape; dev=x_c.device
        S_eff=self._S_effective(); x_proj=x_c@self.U1.conj().T

        # WARM START (v5.9.6): per-sequence r_lista + U_meta gate (I1+I7)
        # I7: expand r_lista to (B, d_r_lista) for per-sequence warm start
        r_lista_B = self.r_lista.unsqueeze(0).expand(B, -1).detach()   # (B, d_r)
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
        if n_r_cur>0: self.rule_util[:n_r_cur].mul_(0.999999).clamp_(max=100.0)  # v6.0.9: calibrated for 1M-token max (0.999999^1M≈0.37)
        # v6.0.7 NR-2/NR-3: ARC rule cache retrieval — top-K=3 + learned gate
        if self.episodic_rule_n>0 and self._rule_cache_n>0:
            n_r=self._rule_cache_n
            K_r=self.rule_K[:n_r]; V_r=self.rule_V[:n_r]   # (n_r,d_c)
            x_query=(x_c.mean(0)@self.U1.conj().T.detach())   # (d_c,) → LISTA space
            # True cosine similarity (normalised)
            sims=(x_query@K_r.conj().T).real/(x_query.norm().clamp(1e-8)*K_r.norm(dim=-1).clamp(1e-8)+1e-8)  # (n_r,)
            # NR-2: top-K=3 softmax-weighted retrieval (T=0.5)
            k_ret=min(3,n_r)
            top_idx=torch.topk(sims,k_ret).indices           # (k_ret,)
            w_k=torch.softmax(sims[top_idx]/0.5,dim=0)       # (k_ret,) temperature-scaled
            v_blend=(w_k.to(torch.cfloat).unsqueeze(-1)*V_r[top_idx]).sum(0)  # (d_c,) blended
            # NR-3: learned gate (replaces fixed 0.3 threshold)
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
            h_n=complex_soft_threshold(z,self._tau_k(k).unsqueeze(0))
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
        x_ref=h@self.U2.conj().T

        # v5.9.8 R1.B: Update sparse code cache — skip during thinking (v6.0 CTP)
        if self.sparse_code_cache_K>0 and not self._in_thinking_mode:
            with torch.no_grad():
                K=self.sparse_code_cache_K; h_mean=h.mean(0).detach()
                if self._cache_filled<K:
                    self.h_cache[self._cache_filled]=h_mean
                    self._cache_filled+=1
                else:
                    self.h_cache[:-1]=self.h_cache[1:].clone()   # shift left
                    self.h_cache[-1]=h_mean

        # v5.9.8 R3.B: Update episodic rule cache on successful escape
        if self.episodic_rule_n>0 and h_pre_escape is not None:
            pass   # U_meta computed below; write handled after U_meta_v2 computation

        # UPDATE session reservoir — fully detached (no BPTT through r_lista)
        # Gradient to W_ri: comes from next step's loss via next h_0 computation
        with torch.no_grad():
            e_lista=h.mean(0).detach()                           # (d_c,) batch mean
            self.r_lista=(self.lambda_lista*self.r_lista+self.W_ri@e_lista)
            # W_ri is a fixed buffer (v5.9.5): no gradient needed, no .detach() required

        df    =deltas[-1].detach() if isinstance(deltas[-1],torch.Tensor) else torch.tensor(float(deltas[-1]) if deltas else 0.0,device=dev)
        U_conv=1.0-torch.exp(torch.tensor(-5.0,device=dev)*df)
        if compute_meta:
            st_eval=torch.full((B,),0.1,device=dev)
            xp=edm_precondition_complex(x_ref.detach(),st_eval,self)
            residual=((xp-x_ref.detach()).conj()*(xp-x_ref.detach())).real.sum(-1).mean().sqrt()
            U_repr=1.0-torch.exp(torch.tensor(-2.0,device=dev)*residual)
        else: U_repr=torch.tensor(0.0,device=dev)
        w=torch.sigmoid(self.w_conv); U_meta=w*U_conv+(1-w)*U_repr
        # v5.9.8 R2.B: U_hopfield from last Hopfield retrieval confidence
        u_hop=float(getattr(hopfield,'_last_confidence',0.0)) if hopfield is not None else 0.0
        # v5.9.8 R2.A: U_epistemic from routing (stored on bank by CFL5Layer)
        u_epi=float(getattr(bank,'_u_epistemic_last',0.0)) if bank is not None else 0.0
        # v5.9.8 R2.A+R2.B: composite U_meta_v2 with learned weights
        w_v2=torch.softmax(self.log_w_meta,dim=0)
        U_meta_v2=w_v2[0]*U_meta+w_v2[1]*u_epi+w_v2[2]*u_hop
        U_meta=U_meta_v2   # replace U_meta with composite (backward compat: default weights degrade gracefully)
        # v5.9.8 R3.B: Write to rule cache — skip during thinking (v6.0 CTP)
        # v6.0.7 NR-1: Dual-trigger write + v6.0.6 ARC merge — skip during thinking
        if self.episodic_rule_n>0 and not self._in_thinking_mode:
            U_meta_f=float(U_meta_v2.item()) if isinstance(U_meta_v2,torch.Tensor) else float(U_meta_v2)
            U_epi_f =float(getattr(bank,'_last_u_epi',0.0) if bank is not None else 0.0)  # v6.0.9: bank not self
            # Trigger A (escape resolved): h_pre_escape set + U_meta<0.3
            trig_A = h_pre_escape is not None and U_meta_f<0.3
            # Trigger B (novelty resolved): uncertain input but good resolution
            trig_B = U_epi_f>0.6 and U_meta_f<0.4
            if trig_A or trig_B:
                with torch.no_grad():
                    K_new=(x_c.mean(0)@self.U1.conj().T.detach()).detach()
                    V_new=h.mean(0).detach()
                    n_r=self._rule_cache_n
                    if n_r>0:
                        K_r=self.rule_K[:n_r]
                        sims_w=(K_new@K_r.conj().T).real/(K_new.norm().clamp(1e-8)*K_r.norm(dim=-1).clamp(1e-8)+1e-8)
                        best_sim,best_i=sims_w.max(0)
                        if float(best_sim.item())>0.7:   # ARC merge into existing rule
                            self.rule_K[best_i]=0.7*self.rule_K[best_i]+0.3*K_new
                            self.rule_V[best_i]=0.7*self.rule_V[best_i]+0.3*V_new
                            self.rule_util[best_i]+=0.5
                        else:                             # write new rule (QWR eviction)
                            ptr=(int(self.rule_util[:n_r].argmin().item()) if n_r>=self.episodic_rule_n
                                 else n_r)
                            self.rule_K[ptr]=K_new; self.rule_V[ptr]=V_new
                            self.rule_util[ptr]=0.0
                            self._rule_cache_n=min(self.episodic_rule_n,n_r+1)
                    else:
                        self.rule_K[0]=K_new; self.rule_V[0]=V_new
                        self.rule_util[0]=0.0; self._rule_cache_n=1
                    self.rule_ptr.add_(1)
        self._last_warm_norm=float(warm.norm(dim=-1).mean().item())   # v5.9.6: mean over B
        self._last_u_temporal=u_temporal   # v6.0.7 MC-3: cached for MC-2 signal
        self._prev_U_meta=float(U_meta.item()) if isinstance(U_meta,torch.Tensor) else float(U_meta)  # v5.9.6 I1
        # v6.0.7 MC-2: session-adaptive log_w_rec update (uses U_hopfield as CE proxy)
        # U_hopfield high → content hard to recall → high difficulty proxy
        if hasattr(self,'_log_w_rec') and len(self._log_w_rec)==4:
            u_hop_f=float(u_hop)
            u_epi_cal_f=float(getattr(bank,'_last_u_epi',0.5) if bank else 0.5)
            ce_proxy=float(self._prev_U_meta) if hasattr(self,'_prev_U_meta') else 0.5
            # v6.0.9: prev U_meta as difficulty proxy (avoids self-reinforcing bias
            # that occurs when ce_proxy==U_hopfield → signal[hopfield]=1.0 always)
            u_signals=[float(U_repr) if not isinstance(U_repr,float) else U_repr,
                       u_epi_cal_f,
                       u_hop_f,
                       float(getattr(self,'_last_u_temporal',0.0))]
            for k in range(4):
                signal_k=1.0-abs(u_signals[k]-ce_proxy)    # agreement between U[k] and difficulty
                self._log_w_rec[k]=0.95*self._log_w_rec[k]+0.05*signal_k
        return x_ref,h,{'U_conv':U_conv,'U_repr':U_repr,'U_meta':U_meta,
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
