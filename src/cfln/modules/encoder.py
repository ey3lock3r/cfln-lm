import torch
import torch.nn as nn

from cfln.utils import complex_layer_norm, compute_ntk_rope_base
from cfln.modules.lru import ComplexLRU
from cfln.modules.titans import TitansComplexMemory


class ComplexHierarchicalOCNEncoder(nn.Module):
    """
    LRU (fast) + TitansComplexMemory (slow). v5.9.3.
    CRoPE REMOVED from encoder output — encoder returns position-agnostic x_e.
    CRoPE applied at: CFL-5 residual (CFLNModel.forward) and titans_query.
    """
    def __init__(self, embed, d_c, d_ssm_fast=32, S_f=32, C_chunk=32,
                 use_crope=True, eta_titans=0.01, theta_decay_init=0.99,
                 null_threshold_init=0.95, k_null=50.0, beta_null_aux=0.01,
                 domain_alpha=0.90, domain_mag_alpha=0.99,
                 domain_threshold_init=3.0, surprise_warmup_chunks=32,
                 rope_L_train=2048, rope_L_target=1_048_576,
                 per_sequence_memory=True):
        super().__init__()
        self.d_c=d_c; self.C_chunk=C_chunk; self.use_crope=use_crope
        self.per_seq=per_sequence_memory; self.embed=embed
        self.fast_lru=ComplexLRU(d_c,d_ssm_fast,S_f)
        self.titans=TitansComplexMemory(
            d_c=d_c,C_chunk=C_chunk,eta_init=eta_titans,
            theta_decay_init=theta_decay_init,
            null_threshold_init=null_threshold_init,k_null=k_null,
            beta_null_aux=beta_null_aux,domain_alpha=domain_alpha,
            domain_mag_alpha=domain_mag_alpha,
            domain_threshold_init=domain_threshold_init,
            surprise_warmup_chunks=surprise_warmup_chunks)
        self.W_c_proj  =nn.Parameter((torch.randn(d_c,d_c)+1j*torch.randn(d_c,d_c)).to(torch.cfloat)/d_c**0.5)
        self.B_dec_fast=nn.Parameter((torch.randn(d_c,S_f)+1j*torch.randn(d_c,S_f)).to(torch.cfloat)/S_f**0.5)
        self.register_buffer('_r_titans_cache',torch.zeros(d_c,dtype=torch.cfloat))
        self.rope_base=(compute_ntk_rope_base(d_c,rope_L_train,rope_L_target)
                         if rope_L_target>rope_L_train else 10000.0)
        # Share CRoPE params with Titans for position-aware query (v5.9.3)
        self.titans.set_crope_params(use_crope, self.rope_base)

    def forward(self, token_ids: torch.Tensor, pos_offset: int=0,
                 embedding_override: 'torch.Tensor|None'=None) -> torch.Tensor:
        """
        v5.9.3: pos_offset = absolute position of first token (for Titans query CRoPE).
        Output x_e is position-AGNOSTIC (CRoPE removed from encoder output).
        v6.0.5: embedding_override (B,T,d_c) cfloat — RPP injects pre-computed embeddings
        directly, bypassing embed lookup for differentiable gradient through e_think.
        """
        B,T=token_ids.shape; outputs=[]
        for t in range(T):
            # v6.0.5: RPP embedding override — bypass lookup for differentiable e_think
            if embedding_override is not None:
                e_c_t=embedding_override[:,t]       # (B,d_c) cfloat — injected directly
            else:
                e_c_t=self.embed(token_ids[:,t])
            # v6.0.2 H2: gate LRU update during CTP thinking tokens
            _thinking=getattr(self.titans,'_in_thinking_mode',False)
            if not _thinking:
                if self.per_seq:
                    Z_fast=self.fast_lru.step_per_sequence(e_c_t)
                    e_for_titans=e_c_t.mean(0)
                else:
                    e_mean=e_c_t.mean(0)
                    Z_fast=self.fast_lru.step(e_mean).unsqueeze(0).expand(B,-1)
                    e_for_titans=e_mean
            else:
                # Thinking: reuse current LRU output without advancing its state
                e_for_titans=e_c_t.mean(0)
                if self.per_seq:
                    h_cur=(self.fast_lru._h_batch
                           if self.fast_lru._h_batch is not None
                           else torch.zeros(B,self.fast_lru.C_c.shape[0],
                                            dtype=torch.cfloat,device=e_c_t.device))
                    Z_fast=(h_cur.detach()@self.fast_lru.C_c.conj().T)
                else:
                    Z_fast=(self.fast_lru.h.detach().unsqueeze(0).expand(B,-1)
                            @self.fast_lru.C_c.conj().T)
            self.titans.accumulate(e_c_t)
            r_new,s_t,cs,uw,stepped=self.titans.maybe_step()
            if stepped and r_new is not None:
                with torch.no_grad(): self._r_titans_cache.copy_(r_new.detach())
            abs_pos=pos_offset+t
            r_titans=self.titans.titans_query(e_for_titans,pos=abs_pos)  # CRoPE on Q_t
            proj  =e_c_t@self.W_c_proj.conj().T
            fast_c=Z_fast@self.B_dec_fast.conj().T
            # NO CRoPE here — position-agnostic output for CNEP
            x_e=complex_layer_norm(proj+fast_c+r_titans.unsqueeze(0).expand(B,-1),[self.d_c])
            outputs.append(x_e)
        return torch.stack(outputs,dim=1)

    def reset_for_inference(self):
        self.fast_lru.reset()
        self.titans.reset_to_neutral()
        with torch.no_grad(): self._r_titans_cache.zero_()
