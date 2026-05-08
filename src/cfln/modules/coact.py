import torch


class CoactivationRegister:
    """Hebbian co-activation. v5.9.3: scalar float addition (no CPU tensor creation)."""
    def __init__(self, N_max_l, K_hebb=16):
        self.K_hebb = K_hebb
        self.coact_reg  = torch.full((N_max_l, K_hebb), -1, dtype=torch.long)
        self.coact_cnt  = torch.zeros(N_max_l, K_hebb, dtype=torch.float16)
        self._write_ptr = torch.zeros(N_max_l, dtype=torch.long)
        self.decay = 0.995

    def to(self, device):
        self.coact_reg  = self.coact_reg.to(device)
        self.coact_cnt  = self.coact_cnt.to(device)
        self._write_ptr = self._write_ptr.to(device)
        return self

    @torch.no_grad()
    def update(self, s_l, threshold=1e-3, increment=0.01):
        n_l = s_l.shape[1]; self.coact_cnt[:n_l].mul_(self.decay)
        active_idx = (s_l.mean(0) > threshold).nonzero(as_tuple=True)[0]
        if len(active_idx) < 2: return
        k = len(active_idx)
        i_idx = active_idx.unsqueeze(1).expand(-1, k).reshape(-1)
        j_idx = active_idx.unsqueeze(0).expand(k, -1).reshape(-1)
        mask = i_idx != j_idx; i_idx = i_idx[mask]; j_idx = j_idx[mask]
        if len(i_idx) == 0: return
        wp = self._write_ptr[i_idx] % self.K_hebb
        self.coact_reg[i_idx, wp] = j_idx
        self.coact_cnt[i_idx, wp] = (self.coact_cnt[i_idx, wp].float() + increment).half()
        u_idx = torch.unique(i_idx)
        self._write_ptr[u_idx] = (self._write_ptr[u_idx] + 1) % self.K_hebb

    def get_hebbian_matrix(self, active_idx):
        reg_a = self.coact_reg[active_idx]; cnt_a = self.coact_cnt[active_idx].float()
        match = (reg_a.unsqueeze(-1) == active_idx.unsqueeze(0).unsqueeze(0))
        return (cnt_a.unsqueeze(-1).expand(-1, -1, len(active_idx)) * match.float()).sum(dim=1)

    def remap_after_prune(self, keep_idx: torch.Tensor) -> None:
        k = len(keep_idx); dev = self.coact_reg.device
        old_to_new = torch.full((self.coact_reg.shape[0],), -1, dtype=torch.long, device=dev)
        old_to_new[keep_idx] = torch.arange(k, dtype=torch.long, device=dev)
        new_reg = torch.full_like(self.coact_reg, -1)
        new_cnt = torch.zeros_like(self.coact_cnt)
        new_ptr = torch.zeros_like(self._write_ptr)
        new_reg[:k] = self.coact_reg[keep_idx]
        new_cnt[:k] = self.coact_cnt[keep_idx]
        new_ptr[:k] = self._write_ptr[keep_idx]
        col = new_reg[:k].clone(); valid = col >= 0
        remapped = torch.where(valid, old_to_new[col.clamp(0)], torch.full_like(col, -1))
        new_reg[:k] = remapped; new_cnt[:k][valid & (remapped < 0)] = 0.0
        self.coact_reg = new_reg; self.coact_cnt = new_cnt; self._write_ptr = new_ptr
