import torch
import torch.nn as nn

from cfln.utils import complex_layer_norm, to_real
from cfln.utils import complex_rope_multiplicative
from cfln.modules.embedding import ComplexEmbedding
from cfln.modules.encoder import ComplexHierarchicalOCNEncoder
from cfln.modules.bank import CFBank
from cfln.modules.cfl5 import CFL5Layer
from cfln.modules.sti_head import ComplexSTIHead
from cfln.modules.uncertainty import ComplexUncertaintyModule, PerLayerLamPSchedule
from cfln.modules.highway import ComplexMHCHighway
from cfln.modules.telescoping import TelescopingMemory
from cfln.modules.surprise import SurpriseArchive
from cfln.modules.diffusion import DiffusionAuxiliaryModule
from cfln.modules.hopfield import HopfieldRetrieval
from cfln.modules.si import SynapticIntelligence, ExemplarDormancyBuffer, DomainTransitionHandler
from cfln.modules.dynamic_bank import DynamicLocalBank
from cfln.modules.uncertainty import CFLNPathologyMonitor
from cfln.modules.monitoring import SlowDriftDetector
from cfln.modules.v9_ops import consolidate_arc_to_cnep, micro_consolidate_arc  # noqa: F401 — used in reset_for_inference/_update_telescoping
from cfln.modules.telescoping import vq_telescope_update, vq_telescope_retrieve  # noqa: F401 — added in Step 6


class CFLNModel(nn.Module):
    """CFLN v5.9.4. All R1-R7 + gap fixes + v5.9.3 fixes + v5.9.4 RC integration."""
    def __init__(self,cfg):
        super().__init__()
        d_c=cfg['d_c']; self.d_c=d_c; K=cfg.get('K_stats',8); self.K_stats=K
        self._pos_offset=0

        self.embed   =ComplexEmbedding(cfg['vocab_size'],d_c)
        self.encoder =ComplexHierarchicalOCNEncoder(
            embed=self.embed,d_c=d_c,d_ssm_fast=cfg.get('d_ssm_fast',32),S_f=cfg.get('S_f',32),
            C_chunk=cfg.get('C_chunk',32),use_crope=cfg.get('use_crope',True),
            eta_titans=cfg.get('eta_titans',0.01),theta_decay_init=cfg.get('theta_decay_init',0.99),
            null_threshold_init=cfg.get('null_threshold_init',0.95),k_null=cfg.get('k_null',50.0),
            beta_null_aux=cfg.get('beta_null_aux',0.01),domain_alpha=cfg.get('domain_alpha',0.90),
            domain_mag_alpha=cfg.get('domain_mag_alpha',0.99),
            domain_threshold_init=cfg.get('domain_threshold_init',3.0),
            surprise_warmup_chunks=cfg.get('surprise_warmup_chunks',32),
            rope_L_train=cfg.get('rope_L_train',2048),rope_L_target=cfg.get('rope_L_target',1_048_576),
            per_sequence_memory=cfg.get('per_sequence_memory',True))
        # CFBank with RC params (v5.9.4)
        self.bank=CFBank(
            cfg.get('n_l',2048),cfg.get('n_p',256),d_c,  # v6.0.8: n_g removed
            cfg.get('d_e_l',32),cfg.get('d_e_p',64),  # v6.0.8: d_e_g removed
            cfg.get('D_g',8),cfg.get('K_hebb',16),
            d_r_node=cfg.get('d_r_node',8),
            rho_node=cfg.get('rho_node',0.95),
            n_heads_gat=cfg.get('n_heads_gat',4),
            K_L1=cfg.get('K_L1',128), K_L2=cfg.get('K_L2',32), K_L3=cfg.get('K_L3',32),
            C_chunk=cfg.get('C_chunk',32), n_roles=cfg.get('n_roles',8),
            # v5.9.6 I5: multi-scale rho kwargs
            rho_fast=cfg.get('rho_fast',0.70),
            rho_mid=cfg.get('rho_mid',0.90),
            rho_slow=cfg.get('rho_slow',0.99))
        # §1.64 C2 / §1.70 C7: per-bank hyperparams set from cfg
        self.bank.k_l_min = cfg.get('k_l_min', 10)
        self.bank.k_l_max = cfg.get('k_l_max', 40)
        self.bank.tau_proto_min = cfg.get('tau_proto_min', 0.4)
        L=cfg.get('L',6); self.lam_p_schedule=PerLayerLamPSchedule(L=L)
        self.cfl_layers=nn.ModuleList([
            CFL5Layer(self.bank,l,self.lam_p_schedule)
            for l in range(L)])
        self.sti_head   =ComplexSTIHead(d_c,cfg.get('S_f',32),cfg.get('D_g',8),
                                         cfg['vocab_size'],cfg.get('beta_U',0.3),cfg.get('D_bptt',8))
        self.unc_module =ComplexUncertaintyModule(d_c)
        self.highway    =ComplexMHCHighway(d_c=d_c,L=L)
        self.field_stats_proj=nn.Linear(2*K+1,2*d_c)
        self.telescoping_mem=TelescopingMemory(
            d_c=d_c,K_L1=cfg.get('K_L1',128),K_L2=cfg.get('K_L2',32),
            K_L3=cfg.get('K_L3',32),C_chunk=cfg.get('C_chunk',32),
            beta=cfg.get('beta_telescoping',1.0))
        # §1.59 VQ-Telescope: W_compress_L1/L2/L3 removed; replaced by VQ routing weight vectors on bank
        self.surprise_archive=SurpriseArchive(
            d_c=d_c,N_archive=cfg.get('N_archive',256),N_tau=cfg.get('surprise_N_tau',100),
            W_warmup=cfg.get('surprise_warmup_chunks',32),tau_percentile=cfg.get('surprise_threshold_pct',0.80))
        self.W_gate_mem  =nn.Parameter(torch.zeros(4,2*d_c))
        self.w_outer_gate=nn.Parameter(torch.zeros(2*d_c))
        self._L_compress_accum=None

        # §1.50: W_rc_bridge changed from register_buffer to nn.Parameter (trained via L_bridge)
        d_r_node_=cfg.get('d_r_node',8); d_r_lista_=cfg.get('d_r_lista',None) or d_c//2
        W_bridge_init=((torch.randn(d_r_lista_,d_r_node_)+1j*torch.randn(d_r_lista_,d_r_node_)
                       ).to(torch.cfloat)/d_r_node_**0.5)
        self.W_rc_bridge = nn.Parameter(W_bridge_init)              # (d_r_lista, d_r_node) TRAINED

        # DiffusionAuxiliaryModule with RC params (v5.9.4)
        self.diff_aux=DiffusionAuxiliaryModule(
            d_c,cfg.get('T_diff',1000),cfg.get('n_fourier',32),
            cfg.get('lambda_diff_init',0.1),cfg.get('lambda_diff_max',0.5),
            cfg.get('lambda_loss_max',100.0),N_iter=cfg.get('N_iter_refine',8),
            delta_stuck=cfg.get('delta_stuck',0.1),delta_min=cfg.get('delta_min',0.01),
            epsilon_esc=cfg.get('epsilon_esc',0.05),
            d_r_lista=cfg.get('d_r_lista',None),
            rho_lista=cfg.get('rho_lista',0.99),
            sparse_code_cache_K=cfg.get('sparse_code_cache_K',32),   # v5.9.8
            episodic_rule_n=cfg.get('episodic_rule_cache_n', cfg.get('N_rules', 256)),
            lista_min_ratio=cfg.get('lista_min_ratio',0.25),
            lista_convergence_ratio=cfg.get('lista_convergence_ratio',0.5))
        self.refine=IterativeRefinementModule(
            cun=self.diff_aux.cun,cfl_layers=self.cfl_layers,bank=self.bank,d_c=d_c,
            N_iter=cfg.get('N_iter_refine',8),N_hop=cfg.get('N_hop_refine',4),
            n_pre_layers=cfg.get('n_layers_diff',2),
            use_hopfield_coupling=cfg.get('use_hopfield_refine',True),
            use_escape=cfg.get('use_escape_refine',True))
        self.diff_aux.cun.init_S_from_unitaries()
        self.diff_aux.cun.beam_B_max = cfg.get('beam_B_max', 3)  # §1.66: adaptive beam width

        self.si          =SynapticIntelligence(cfg.get('c_SI',0.5),cfg.get('rho_SI',0.999),cfg.get('beta_SI',3.0))
        self.dormancy_buf=ExemplarDormancyBuffer(
            d_c,d_e_l=cfg.get('d_e_l',32),D_g=cfg.get('D_g',8),capacity=cfg.get('N_dormant',512))
        self.domain_handler=DomainTransitionHandler(max_history=20)
        self.dyn         =DynamicLocalBank(self.bank)
        self.monitor     =CFLNPathologyMonitor(L=L)
        self.slow_drift_detector=SlowDriftDetector(
            window=cfg.get('slow_drift_window',500),threshold=cfg.get('slow_drift_threshold',0.5),
            N_check=cfg.get('slow_drift_check_freq',200))
        self.lam_p_corrections=torch.ones(L); self._last_domain_step=-9999
        self._last_proactive_snapshot=-9999   # v5.9.8 CL.A: proactive SI trigger
        self._optimizers_built=False   # v6.0.2 H3: ordering guard for expand_vocabulary
        # v5.9.9 DCG+: calibration weights for commitment score
        self.w_commit=nn.Parameter(torch.ones(3))   # [w_epistemic, w_routing, w_hopfield]
        # v6.0 CTP: thinking token IDs and mode flag
        # IDs are set by expand_vocabulary(); -1 = not yet initialised (no thinking tokens)
        # v6.0.4 C1: register_buffer so THINK IDs survive state_dict save/load
        # Properties THINK_START_ID / THINK_END_ID provide backward-compatible access
        self.register_buffer('_think_start_id', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('_think_end_id',   torch.tensor(-1, dtype=torch.long))
        self._in_thinking_mode: bool = False   # gates Titans/Telescoping/SurpriseArchive/CL.A
        for l,layer in enumerate(self.cfl_layers): layer._lam_p_correction=1.0
        _=self.si._get_named_params(self)
        self.cfg = cfg  # keep for v9.0 helpers (micro_consolidate_arc, etc.)
        self._archive_loaded = False  # §1.38 Y3: guard for cross-session archive load

    def _apply(self, fn):
        super()._apply(fn)
        self.telescoping_mem._apply(fn)
        self.surprise_archive._apply(fn)
        return self

    def setup_device(self,device: torch.device) -> 'CFLNModel':
        """Move non-Module components to device. CALL AFTER model.to(device)."""
        self.bank.coact_register.to(device)
        self.telescoping_mem.to(device)
        self.surprise_archive.to(device)
        return self

    def expand_vocabulary(self, n_new: int=6) -> None:
        """v9.0: Expand vocabulary by n_new tokens (default 6).
        base+0: <think>, base+1: </think>, base+2: <hypo>, base+3: </hypo>,
        base+4: <push_goal>, base+5: </push_goal>  (§1.32 R3 + §1.39 Z)
        Extends embed_real, embed_imag, W_vocab.weight, W_vocab.bias.
        Raises ValueError if called more than once.
        """
        # v6.0.1 M8: guard against double-expansion
        if self.THINK_START_ID >= 0:
            raise ValueError(
                f'expand_vocabulary() already called (THINK_START_ID={self.THINK_START_ID}). '
                'Call only once.')
        # v6.0.2 H3: guard against wrong ordering (must be BEFORE build_optimizers)
        if getattr(self, '_optimizers_built', False):
            raise ValueError(
                'expand_vocabulary() must be called BEFORE build_optimizers_v600(). '
                'New embedding rows will not receive gradient if optimizers already built.')
        d_c=self.d_c
        # v6.0.1 H1: capture old_vocab BEFORE expansion (not after with shape-n_new trick)
        old_vocab=self.encoder.embed.embed_real.weight.shape[0]
        with torch.no_grad():
            old_r=self.encoder.embed.embed_real.weight.data.clone()
            new_r=torch.zeros(n_new,d_c,device=old_r.device,dtype=old_r.dtype)
            nn.init.normal_(new_r,std=0.02)
            self.encoder.embed.embed_real.weight=nn.Parameter(torch.cat([old_r,new_r],dim=0))
            old_i=self.encoder.embed.embed_imag.weight.data.clone()
            new_i=torch.zeros(n_new,d_c,device=old_i.device,dtype=old_i.dtype)
            nn.init.normal_(new_i,std=0.02)
            self.encoder.embed.embed_imag.weight=nn.Parameter(torch.cat([old_i,new_i],dim=0))
            wv=self.sti_head.W_vocab
            if wv is not None:
                # Expand weight (out_features, in_features) — rows = output vocab size
                old_w=wv.weight.data.clone()
                new_w=torch.zeros(n_new,old_w.shape[1],device=old_w.device,dtype=old_w.dtype)
                nn.init.normal_(new_w,std=0.02)
                self.sti_head.W_vocab.weight=nn.Parameter(torch.cat([old_w,new_w],dim=0))
                # v6.0.1 C1: also expand bias — nn.Linear has both weight AND bias
                if wv.bias is not None:
                    old_b=wv.bias.data.clone()                  # (V,)
                    new_b=torch.zeros(n_new,device=old_b.device,dtype=old_b.dtype)
                    self.sti_head.W_vocab.bias=nn.Parameter(torch.cat([old_b,new_b],dim=0))
        # old_vocab captured BEFORE expansion block — safe and unambiguous
        self.THINK_START_ID=old_vocab       # <think>
        self.THINK_END_ID  =old_vocab+1     # </think>
        # §1.32 R3: HYPO tokens
        self.register_buffer('_hypo_start_id', torch.tensor(old_vocab+2, dtype=torch.long))
        self.register_buffer('_hypo_end_id',   torch.tensor(old_vocab+3, dtype=torch.long))
        # §1.39 Z: SSP tokens
        self.register_buffer('_push_goal_id',  torch.tensor(old_vocab+4, dtype=torch.long))
        self.register_buffer('_pop_goal_id',   torch.tensor(old_vocab+5, dtype=torch.long))

    # ── v6.0.4 C1: properties for backward-compatible THINK_ID access ─────────
    @property
    def THINK_START_ID(self) -> int:
        return int(self._think_start_id.item())

    @THINK_START_ID.setter
    def THINK_START_ID(self, v: int):
        self._think_start_id.fill_(v)

    @property
    def THINK_END_ID(self) -> int:
        return int(self._think_end_id.item())

    @THINK_END_ID.setter
    def THINK_END_ID(self, v: int):
        self._think_end_id.fill_(v)

    def reset_for_inference(self) -> None:
        """Reset all session state including both reservoirs."""
        # §1.37 Y2: consolidate ARC rules → μ_c_l BEFORE clearing session state
        consolidate_arc_to_cnep(self.bank, self.diff_aux.cun,
                                tau_consol=self.cfg.get('tau_consol', 3.0),
                                alpha_consol=self.cfg.get('alpha_consol', 0.001))
        # §1.38 Y3: persist SurpriseArchive (optional)
        if self.cfg.get('persist_archive', False):
            self.surprise_archive.save_state(self.cfg.get('archive_path', 'archive.pt'))
        if self.cfg.get('persist_archive', False) and not self._archive_loaded:
            self.surprise_archive.load_state(self.cfg.get('archive_path', 'archive.pt'))
            self._archive_loaded = True

        self.encoder.reset_for_inference()
        self.telescoping_mem.reset()
        self.surprise_archive.reset()
        self.sti_head.reset()
        self._pos_offset=0
        self.bank.reset_reservoir()                          # node reservoir (also clears g_c and HYPO state)
        self.bank._last_salience=1.0
        self.diff_aux.cun.reset_lista_reservoir()            # LISTA reservoir (clears SSP stack, precision, etc.)
        self._in_thinking_mode=False
        self._x_c_prev=None
        # Note: bank._u_epi_mu/_u_epi_var deliberately NOT reset — global calibration stats
        self._ema_delta=0.0          # v6.0.7 MC-3: running mean of δ_t
        if hasattr(self.encoder,'titans'): self.encoder.titans._in_thinking_mode=False

    def _compute_field_stats(self,info_t_list,K,device):
        T_=len(info_t_list); B=info_t_list[0].get('B',1) if info_t_list else 1; out=[]
        for t in range(T_):
            info=info_t_list[t]; s_l=info.get('s_l'); E_l=info.get('E_l'); alp=info.get('alp_l')
            if s_l is not None and E_l is not None:
                tKs=torch.topk(s_l,min(K,s_l.shape[-1]),dim=-1)[0]
                if tKs.shape[-1]<K: tKs=torch.cat([tKs,torch.zeros(B,K-tKs.shape[-1],device=device)],dim=-1)
                tKe=torch.topk(-E_l,min(K,E_l.shape[-1]),dim=-1)[0]; tKe=-tKe
                if tKe.shape[-1]<K: tKe=torch.cat([tKe,torch.zeros(B,K-tKe.shape[-1],device=device)],dim=-1)
                am=(alp[(s_l.mean(0)>1.0/s_l.shape[-1])].mean().unsqueeze(0).expand(B,1)
                    if alp is not None else torch.zeros(B,1,device=device))
                out.append(torch.cat([tKs,tKe,am],dim=-1))
            else: out.append(torch.zeros(B,2*K+1,device=device))
        return torch.stack(out,dim=1)

    def _retrieve_all_memory(self, x_c_query):
        x_rm = to_real(x_c_query.mean(0))
        gates = torch.sigmoid(self.W_gate_mem @ x_rm)
        # §1.59 VQ-Telescope: retrieve via routing weight space if bank has VQ buffers
        bank = self.bank
        if hasattr(bank, 'buf_L1_w_full') and bank._L1_ptr > 0:
            # §1.59 OI-7: use cached last routing weight vector as query (was always zeros)
            _cached = getattr(bank, '_last_s_l_full', None)
            if _cached is not None:
                s_l_full_q = _cached.to(x_c_query.device)
            else:
                s_l_full_q = torch.zeros(bank.N_max_l, dtype=torch.float32, device=x_c_query.device)
            r_L1, r_L2, r_L3 = vq_telescope_retrieve(s_l_full_q, bank)
        else:
            r_L1, r_L2, r_L3 = self.telescoping_mem.retrieve_all(x_c_query)
        r_arch = self.surprise_archive.retrieve(x_c_query)
        r_comb = gates[0]*r_L1 + gates[1]*r_L2 + gates[2]*r_L3 + gates[3]*r_arch
        return torch.sigmoid(self.w_outer_gate @ x_rm) * r_comb

    def _update_telescoping(self, x_c_final_chunk: torch.Tensor, s_t: float=0.0,
                             s_l_full: 'torch.Tensor|None'=None,
                             chunk_token_ids: 'torch.Tensor|None'=None,
                             E_min_raw: float=0.0,
                             sel_l: 'torch.Tensor|None'=None,
                             training: bool=True) -> torch.Tensor:
        # §1.51: chunk_mean WITHOUT .detach() so gradient flows to CFL5Layer
        chunk_mean = x_c_final_chunk.mean(dim=(0, 1))  # (d_c,) — gradient-connected

        # §1.59 VQ-Telescope: store routing weight vectors instead of compressed embeddings
        bank = self.bank
        L_vq = torch.tensor(0.0, device=chunk_mean.device)
        if s_l_full is not None and hasattr(bank, 'buf_L1_w_full'):
            L_vq = vq_telescope_update(
                chunk_mean, s_l_full, E_min_raw,
                chunk_token_ids if chunk_token_ids is not None else torch.zeros(bank.C_chunk, dtype=torch.int32, device=chunk_mean.device),
                bank, sel_l if sel_l is not None else torch.zeros(0, dtype=torch.long, device=chunk_mean.device),
                self.cfg
            )
            bank._last_L_vq = L_vq
        else:
            # Fallback: old TelescopingMemory path (pre-VQ)
            self.telescoping_mem.add_L1(chunk_mean.detach())

        # L_bridge: §1.50 — train W_rc_bridge via local loss
        with torch.no_grad():
            last_info = getattr(self, '_last_bridge_info', None)
        if last_info is not None:
            sel_b = last_info.get('sel_l'); s_b = last_info.get('s_l')
            if sel_b is not None and s_b is not None:
                s_w = s_b.mean(0)[sel_b].to(torch.cfloat)
                rho_sel = bank.rho_l[sel_b]
                rho_weighted = (s_w.unsqueeze(-1) * rho_sel).sum(0)
                r_seed = self.W_rc_bridge @ rho_weighted
                _cun = self.diff_aux.cun
                r_seed_target = (_cun.U1.conj() @ chunk_mean.detach())[:_cun.d_r_lista].detach()
                L_bridge = ((r_seed - r_seed_target).conj() * (r_seed - r_seed_target)).real.sum()
                self._last_L_bridge = L_bridge
            else:
                self._last_L_bridge = torch.tensor(0.0, device=chunk_mean.device)
        else:
            self._last_L_bridge = torch.tensor(0.0, device=chunk_mean.device)

        # Welford E_min stats for adaptive spawn threshold (§1.68 C6 / §1.69) — training only
        bank._last_E_min_raw = E_min_raw  # §1.69: cached for memory_update_v605 spawn gate
        if training:
            bank._Emin_n += 1
            n = bank._Emin_n
            delta = E_min_raw - bank._Emin_mean
            bank._Emin_mean += delta / n
            bank._Emin_var += delta * (E_min_raw - bank._Emin_mean)
        else:
            n = bank._Emin_n

        # Surprise archive: threshold driven by Welford mean (§1.68 C6)
        surprise_thresh = self.cfg.get('surprise_threshold', 0.5)
        if n > 10:
            welford_std = (bank._Emin_var / max(n - 1, 1)) ** 0.5
            surprise_thresh = max(bank._Emin_mean + 2.5 * welford_std, 0.1)
        if E_min_raw > surprise_thresh:
            # pass slot index (% K_L1) so archive stores a valid circular-buffer index
            self.surprise_archive.add_vq((bank._L1_ptr - 1) % bank.K_L1, E_min_raw)

        # §1.54 KA-MC: micro-consolidation per chunk
        micro_consolidate_arc(bank, self.diff_aux.cun, self.cfg)

        return L_vq

    def forward(self,input_ids: torch.Tensor,training: bool=True,use_refinement: bool=False) -> tuple:
        B,T=input_ids.shape
        assert T>0,"v6.0.4 M3: T must be ≥1; T=0 causes NaN in complex_layer_norm"; d_c=self.d_c; dev=input_ids.device; C=self.encoder.C_chunk
        rope_base=self.encoder.rope_base
        pos_offset=getattr(self,'_pos_offset',0)
        x_c=self.encoder(input_ids,pos_offset=pos_offset)
        if not training: self._pos_offset=pos_offset+T
        for l,layer in enumerate(self.cfl_layers): layer._lam_p_correction=float(self.lam_p_corrections[l].item())
        lam_p_vec=self.lam_p_schedule(torch.zeros(1,device=dev))
        all_infos=[]; x_cur=x_c
        x_fast_hw,x_slow_hw=self.highway.init_streams(B,dev)
        self._L_compress_accum=None
        for l,layer in enumerate(self.cfl_layers):
            x_nxt=torch.zeros_like(x_cur); inf_t=[]
            for t in range(T):
                x_in=x_cur[:,t,:]; xn=complex_layer_norm(x_in,[d_c]) if l>0 else x_in
                xn_aug=xn+self.highway.inject(x_fast_hw,x_slow_hw,l)
                xn_aug=xn_aug+self._retrieve_all_memory(xn_aug)
                # §1.32 R3 + §1.39 Z: HYPO/SSP token dispatch (last CFL layer only)
                if l == len(self.cfl_layers) - 1:
                    tok = input_ids[:, t]
                    cun = self.diff_aux.cun
                    _hypo_start = int(getattr(self, '_hypo_start_id', torch.tensor(-1)).item())
                    _hypo_end   = int(getattr(self, '_hypo_end_id',   torch.tensor(-1)).item())
                    _push_goal  = int(getattr(self, '_push_goal_id',  torch.tensor(-1)).item())
                    _pop_goal   = int(getattr(self, '_pop_goal_id',   torch.tensor(-1)).item())
                    if _hypo_start >= 0 and (tok == _hypo_start).any():
                        self.bank._r_lista_hypo = cun.r_lista.clone()
                        self.bank._in_hypo_mode = True
                    elif _hypo_end >= 0 and (tok == _hypo_end).any():
                        if self.bank._r_lista_hypo is not None:
                            diff_sq = float((self.bank._r_lista_hypo - cun.r_lista).norm()**2)
                            self.bank._u_hypo = float(torch.sigmoid(torch.tensor(diff_sq / max(cun.d_r_lista, 1))).item())
                        self.bank._in_hypo_mode = False
                        self.bank._r_lista_hypo = None
                    elif _push_goal >= 0 and (tok == _push_goal).any():
                        if len(cun._goal_stack) < self.cfg.get('ssp_max_depth', 4):
                            cun._goal_stack.append(cun.r_lista.clone())
                            cun._stuck_count.append(0)
                            cun._v_prev.append(1e9)
                            self.bank._goal_frozen = True  # §1.39: freeze g_c during goal pursuit
                    elif _pop_goal >= 0 and (tok == _pop_goal).any():
                        if cun._goal_stack:
                            parent = cun._goal_stack.pop()
                            if cun._stuck_count: cun._stuck_count.pop()
                            if cun._v_prev: cun._v_prev.pop()
                            # §1.65 C3: Q_BEAM-weighted merge
                            merge_w = torch.sigmoid(torch.tensor(cun._last_Q_BEAM_score))
                            with torch.no_grad():
                                cun.r_lista = (1.0 - merge_w) * parent + merge_w * cun.r_lista
                            if not cun._goal_stack:
                                self.bank._goal_frozen = False  # §1.39: unfreeze when stack empty
                _upd_res=(l==len(self.cfl_layers)-1)  # v5.9.5 B6: only last layer updates reservoir
                xo,Z,U,info=layer(xn_aug,training=training,lam_p=float(lam_p_vec[l].item()),update_res=_upd_res)
                ar=torch.exp(layer.log_alpha_res)
                abs_pos=pos_offset+t
                xo_pos=(complex_rope_multiplicative(xo,abs_pos,d_c,rope_base)
                         if self.encoder.use_crope else xo)
                x_nxt[:,t,:]=x_in+ar*xo_pos; inf_t.append(info)
                if l==len(self.cfl_layers)-1 and t>0 and t%C==0:
                    prev_chunk=x_nxt[:,t-C:t,:]
                    s_t=self.encoder.titans.get_surprise(x_c[:,t-C:t,:].mean(dim=(0,1)).detach())
                    # v6.0 CTP: skip telescoping+archive updates during thinking tokens
                    if not self._in_thinking_mode:
                        # §1.59 OI-8: pass routing info so VQ write is activated
                        _info_prev = inf_t[t - C] if t - C >= 0 else inf_t[0]
                        _sel_c = _info_prev.get('sel_l')
                        _sl_c  = _info_prev.get('s_l')
                        _el_c  = _info_prev.get('E_l')
                        _s_l_full_c = None; _e_min_raw_c = 0.0
                        if _sel_c is not None and _sl_c is not None:
                            _s_l_full_c = torch.zeros(self.bank.N_max_l, dtype=torch.float32, device=dev)
                            _s_l_full_c[_sel_c] = _sl_c.mean(0)[_sel_c].float()
                            if _el_c is not None:
                                _e_min_raw_c = float(_el_c.min().item())
                        L_c=self._update_telescoping(prev_chunk, s_t,
                                                      s_l_full=_s_l_full_c, sel_l=_sel_c,
                                                      E_min_raw=_e_min_raw_c, training=training)
                        self._L_compress_accum=(L_c if self._L_compress_accum is None else self._L_compress_accum+L_c)
            x_fast_hw,x_slow_hw=self.highway.update(x_fast_hw,x_slow_hw,x_nxt.mean(1),l)
            x_cur=x_nxt; all_infos.append(inf_t)
        last_start=(T//C)*C
        if last_start<T:
            s_f=self.encoder.titans.get_surprise(x_c[:,last_start:,:].mean(dim=(0,1)).detach())
            # §1.59 OI-8: pass routing info from last token of last CFL layer
            _last_info_tail = all_infos[-1][-1]
            _sel_tail = _last_info_tail.get('sel_l')
            _sl_tail  = _last_info_tail.get('s_l')
            _el_tail  = _last_info_tail.get('E_l')
            _s_l_full_tail = None; _e_min_raw_tail = 0.0
            if _sel_tail is not None and _sl_tail is not None:
                _s_l_full_tail = torch.zeros(self.bank.N_max_l, dtype=torch.float32, device=dev)
                _s_l_full_tail[_sel_tail] = _sl_tail.mean(0)[_sel_tail].float()
                if _el_tail is not None:
                    _e_min_raw_tail = float(_el_tail.min().item())
            L_c=self._update_telescoping(x_cur[:,last_start:,:], s_f,
                                          s_l_full=_s_l_full_tail, sel_l=_sel_tail,
                                          E_min_raw=_e_min_raw_tail, training=training)
            self._L_compress_accum=(L_c if self._L_compress_accum is None else self._L_compress_accum+L_c)
        x_fin=complex_layer_norm(x_cur,[d_c]); meta_refine={}

        # v5.9.8 R2.A: aggregate U_epistemic from last CFL layer (all T positions)
        u_epi_vals=[all_infos[-1][t].get('U_epistemic',0.0) for t in range(len(all_infos[-1]))]
        if u_epi_vals:
            self.bank._u_epistemic_last=float(sum(u_epi_vals)/len(u_epi_vals))

        # ── RC BRIDGE (v5.9.6 I4): seed r_lista from routing-weighted node reservoir ──
        # After all L CFL layers, before IterativeRefinement. 'Which units fired' conditions
        # 'what reasoning context to start from'. Makes two-scale RC coherent.
        # RC bridge active at all times (training and inference)
        # §1.50: RC bridge — W_rc_bridge now nn.Parameter, gradient flows for L_bridge
        last_info=all_infos[-1][-1]   # last CFL layer, last token position
        self._last_bridge_info = last_info  # cache for _update_telescoping L_bridge
        sel_bridge=last_info.get('sel_l',None)
        s_bridge=last_info.get('s_l',None)
        if sel_bridge is not None and s_bridge is not None:
            with torch.no_grad():
                s_w=s_bridge.mean(0)[sel_bridge].to(torch.cfloat)
                rho_sel=self.bank.rho_l[sel_bridge]
                rho_weighted=(s_w.unsqueeze(-1)*rho_sel).sum(0)
                r_seed=self.W_rc_bridge@rho_weighted
                # §1.73 C10: adaptive blend alpha
                blend_alpha = float(torch.exp(self.diff_aux.cun.log_blend_alpha).clamp(0.5, 0.95).item())
                self.diff_aux.cun.r_lista = ((1.0 - blend_alpha) * r_seed.detach()
                                              + blend_alpha * self.diff_aux.cun.r_lista)
        if use_refinement and not training: x_fin,meta_refine=self.refine_for_inference(x_fin)
        fstats=self._compute_field_stats(all_infos[-1],self.K_stats,dev)
        fe=self.field_stats_proj(fstats); fstats_emb=torch.complex(fe[...,:d_c],fe[...,d_c:])
        Z_L=torch.zeros(B,T,device=dev)
        for t in range(T):
            s_l_t=all_infos[-1][t].get('s_l',None)
            if s_l_t is not None: Z_L[:,t]=s_l_t.sum(-1)
        x_ch,U_fin=self.unc_module(x_fin.detach(),Z_L); x_ch_aug=x_ch+fstats_emb.detach()
        logits_l,unc_w=[],[]
        # v6.0.9: during inference (or T=1), include all T positions so single-token
        # forward works; during training use T-1 (teacher-forcing, last has no target)
        n_logit_pos = T if (not training or T == 1) else T-1
        for t in range(n_logit_pos):
            lg,uw=self.sti_head.step_and_predict(x_ch_aug[:,t,:],U_fin[:,t])
            logits_l.append(lg); unc_w.append(uw)
        logits=torch.stack(logits_l,dim=1); unc_wts=torch.stack(unc_w,dim=1)
        aux={'all_infos':all_infos,'Z_L':Z_L,'U_final':U_fin,'unc_wts':unc_wts,
              'x_c_final':x_fin,'x_fast_hw':x_fast_hw,'x_slow_hw':x_slow_hw,
              'meta_refine':meta_refine,
              'logits':logits,          # v5.9.9 DCG+: (B,T,V) for block sampling
              'U_hopfield_per_pos':     # v5.9.9 DCG+: scalar Hopfield confidence (broadcast)
                  [float(getattr(self.diff_aux.cun,'_last_confidence',0.0))]*logits.shape[1]}
        return logits,U_fin,aux

    def forward_single_position(self,x_c):
        assert not x_c.requires_grad; x_out,_=self.refine(x_c,training=True); return x_out

    def refine_for_inference(self,x_c_final):
        B,T,d_c=x_c_final.shape; outputs=[]; metas=[]
        for t in range(T):
            xr,meta=self.refine(x_c_final[:,t,:],training=False)
            outputs.append(xr); metas.append(meta)
        return torch.stack(outputs,dim=1),metas[-1] if metas else {}


class IterativeRefinementModule(nn.Module):
    def __init__(self,cun,cfl_layers,bank,d_c,N_iter=8,N_hop=4,
                 n_pre_layers=2,use_hopfield_coupling=True,use_escape=True):
        super().__init__()
        self.cun=cun; self.layers=cfl_layers; self.bank=bank; self.d_c=d_c
        self.N_iter=N_iter; self.N_hop=N_hop; self.n_pre=n_pre_layers
        self.use_hop=use_hopfield_coupling; self.use_esc=use_escape
        self.hopfield=HopfieldRetrieval(beta=1.0)
        self.log_blend=nn.Parameter(torch.tensor(-2.0))

    def forward(self,x_c,training=True):
        x=x_c
        for l in range(min(self.n_pre,len(self.layers))):
            layer=self.layers[l]; xi=complex_layer_norm(x,[self.d_c]) if l>0 else x
            xo,_,_,_=layer(xi,training=False,local_only=True,update_res=False); x=x+torch.exp(layer.log_alpha_res)*xo  # v5.9.5 B7
        x=complex_layer_norm(x,[self.d_c])
        hop=self.hopfield if self.use_hop else None
        bank=self.bank if self.use_hop else None
        # Spec bug: u_temporal_val undefined in this scope — default to 0.0 (safe)
        x_ref,h,meta=self.cun.lista_forward(x,hopfield=hop,bank=bank,N_hop=self.N_hop,
                                              escape=self.use_esc and not training,
                                              compute_meta=not training,
                                              u_temporal=0.0)
        blend=torch.sigmoid(self.log_blend)
        return (1-blend)*x+blend*x_ref,meta

    def compute_lista_loss(self,x_raw,x_refined):
        diff=x_refined-x_raw.detach()
        L_recon=((diff.conj()*diff).real.sum(-1)).mean()
        L_sparse=(x_refined@self.cun.U1.conj().T).abs().mean()*0.01
        with torch.no_grad():
            v=torch.randn(self.d_c,dtype=torch.cfloat,device=x_raw.device)
            v=v/v.norm().clamp(1e-8)
            for _ in range(5): v=self.cun.S@v; v=v/v.norm().clamp(1e-8)
            sv=(self.cun.S@v).norm()
        L_snorm=torch.relu(sv-self.cun.rho_max).pow(2)*10.0
        return L_recon+L_sparse+L_snorm
