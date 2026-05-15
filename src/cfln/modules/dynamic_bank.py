import torch

from cfln.utils import init_stiefel


class DynamicLocalBank:
    """spawn/prune/split/merge. v5.9.4: prune remaps rho_l and log_scale_l.
    v6.0.6: log_decode_scale added to all lifecycle operations."""
    N_max=16384
    def __init__(self,bank): self.bank=bank; self.n_active=bank.n_l

    def spawn(self,x_c):
        if self.n_active>=self.N_max: return -1
        idx=self.n_active; bk=self.bank
        with torch.no_grad():
            bk.mu_c_l.data[idx]=x_c.detach().mean(0)
            bk.W_l.data[idx]=init_stiefel(bk.d_e_l,bk.d_c).to(bk.W_l.device)
            bk.log_alp_l.data[idx]=bk.log_alpha_rq_l.data[idx]=bk.log_ell_l.data[idx]=0.0
            bk.H_c_l[idx].zero_(); bk.h_c_l[idx].zero_()
            bk.rho_l[idx].zero_()                                  # NEW v5.9.4: reset reservoir
            bk.log_scale_l.data[idx]=-3.0                          # NEW v5.9.4: reset scale
            if hasattr(bk,'log_decode_scale'):                      # v6.0.6: reset frequency filter
                bk.log_decode_scale.data[idx].zero_()              # init 0 → uniform weighting
            bk.active_mask_l[idx]=True; bk.is_sensory_l[idx]=False; bk.activation_freq_l[idx]=0.0
            # §1.43 SE-1: init k-shot centroid tracking
            if hasattr(bk, '_proto_count'):
                bk._proto_count[idx] = 1
                bk._proto_sum.data[idx] = bk.mu_c_l.data[idx].clone()
            # §1.63 C1: reset fisher unit
            if hasattr(bk, 'fisher_unit'):
                bk.fisher_unit[idx] = 0.0
        self.n_active+=1; bk.n_l=self.n_active; return idx

    def prune(self,keep_idx: torch.Tensor,dormancy=None,si=None):
        bk=self.bank; n=bk.n_l; dev=bk.is_sensory_l.device
        sens=bk.is_sensory_l[:n]; si_idx=sens.nonzero(as_tuple=True)[0]
        keep_idx=keep_idx.to(dev)
        all_keep=torch.unique(torch.cat([keep_idx,si_idx]))
        pruned=torch.ones(n,dtype=torch.bool,device=dev); pruned[all_keep]=False
        if dormancy is not None:
            for idx in (pruned&~sens).nonzero(as_tuple=True)[0].tolist():
                dormancy.add_from_history(bk,idx)
        k=len(all_keep)
        with torch.no_grad():
            # Parameters (nn.Parameter.data remapping)
            for attr in ['mu_c_l','W_l','log_alp_l','log_alpha_rq_l','log_ell_l',
                         'is_sensory_l','activation_freq_l','sensory_domain_id',
                         'log_scale_l','log_decode_scale']:        # v6.0.6: log_decode_scale
                if hasattr(bk,attr): getattr(bk,attr).data[:k]=getattr(bk,attr).data[all_keep]
            # Buffers (direct tensor remapping)
            _dev=bk.rho_l.device   # v5.9.5 D3: use buffer device, not CPU
            bk.H_c_l[:k]=bk.H_c_l[all_keep.to(_dev)]
            bk.h_c_l[:k]=bk.h_c_l[all_keep.to(_dev)]
            bk.rho_l[:k]=bk.rho_l[all_keep.to(_dev)]
            # §1.43 SE-1: remap k-shot centroid buffers
            if hasattr(bk, '_proto_count'):
                bk._proto_count[:k]=bk._proto_count[all_keep.to(_dev)]
                bk._proto_sum[:k]=bk._proto_sum[all_keep.to(_dev)]
            # §1.63 C1: remap fisher unit
            if hasattr(bk, 'fisher_unit'):
                bk.fisher_unit[:k]=bk.fisher_unit[all_keep.to(_dev)]
            bk.active_mask_l[:k]=True; bk.active_mask_l[k:]=False
        if si is not None: si.remap_after_prune(all_keep)
        bk.coact_register.remap_after_prune(all_keep)
        self.n_active=k; bk.n_l=k

    def split(self,idx):
        bk=self.bank
        if self.n_active>=self.N_max or bk.is_sensory_l[idx]: return -1
        ni=self.spawn(bk.mu_c_l[idx:idx+1])
        if ni<0: return -1
        with torch.no_grad():
            noise=(torch.randn_like(bk.mu_c_l[idx].real)+1j*torch.randn_like(bk.mu_c_l[idx].real))*0.05
            bk.mu_c_l.data[ni]=bk.mu_c_l.data[idx]+noise; bk.W_l.data[ni]=bk.W_l.data[idx].clone()
            # Inherit parent's reservoir state (split unit starts with parent's temporal context)
            bk.rho_l[ni]=bk.rho_l[idx].clone()                    # NEW v5.9.4: inherit reservoir
            if hasattr(bk,'log_decode_scale'):                      # v6.0.6: inherit + perturb scale
                bk.log_decode_scale.data[ni]=bk.log_decode_scale.data[idx]+torch.randn_like(bk.log_decode_scale.data[idx])*0.01
        return ni

    def merge(self,idx_a: int,idx_b: int,si=None):
        bk=self.bank
        if bk.is_sensory_l[idx_a] or bk.is_sensory_l[idx_b]: return
        with torch.no_grad():
            bk.mu_c_l.data[idx_a]=(bk.mu_c_l.data[idx_a]+bk.mu_c_l.data[idx_b])/2
            # Average reservoir states of merged units
            bk.rho_l[idx_a]=(bk.rho_l[idx_a]+bk.rho_l[idx_b])/2  # NEW v5.9.4
            if hasattr(bk,'log_decode_scale'):                      # v6.0.6: average decode scale
                bk.log_decode_scale.data[idx_a]=(bk.log_decode_scale.data[idx_a]+bk.log_decode_scale.data[idx_b])/2
        dev=bk.mu_c_l.device
        keep=torch.cat([torch.arange(0,idx_b,device=dev),torch.arange(idx_b+1,bk.n_l,device=dev)])
        self.prune(keep,si=si)
