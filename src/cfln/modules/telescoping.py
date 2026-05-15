import torch


def vq_telescope_update(chunk_mean, s_l_full, E_min_raw, chunk_token_ids,
                        bank, sel_l, cfg):
    """§1.59 VQ-Telescope: store full routing weight vector per L1 chunk.

    Replaces W_compress; unifies reasoning space (CNEP centroids) and memory space.
    Returns L_vq (encoder commitment loss — gradient to chunk_mean, not mu_c_l).

    s_l_full: (N_max_l,) float32 full routing weight vector
    """
    ptr = bank._L1_ptr % bank.K_L1

    # Store VQ code: full routing weight vector
    with torch.no_grad():
        bank.buf_L1_w_full[ptr] = s_l_full.detach().float()

    # Verbatim token IDs (§1.36 Y1)
    if chunk_token_ids is not None and chunk_token_ids.numel() > 0:
        with torch.no_grad():
            n_ids = min(chunk_token_ids.numel(), bank.C_chunk)
            bank.buf_L1_ids[ptr, :n_ids] = chunk_token_ids[:n_ids].int()

    # L_vq: VQ encoder commitment — gradient flows to chunk_mean (encoder), not centroids
    if sel_l is not None and sel_l.numel() > 0 and len(bank.mu_c_l) > 0:
        nearest_centroids = bank.mu_c_l[sel_l].detach().mean(0)  # detach codebook
        L_vq = ((chunk_mean - nearest_centroids).conj() * (chunk_mean - nearest_centroids)).real.sum()
    else:
        L_vq = torch.tensor(0.0, device=chunk_mean.device)

    # §1.59 OI-7: cache routing weights so retrieval query can use last s_l_full
    bank._last_s_l_full = s_l_full.detach()

    bank._L1_ptr += 1

    # L2 update: average last C_L2=32 L1 routing weight vectors
    C_L2 = cfg.get('K_L2', 32)
    if bank._L1_ptr % C_L2 == 0:
        l2_ptr = (bank._L1_ptr // C_L2 - 1) % bank.K_L2
        start = (bank._L1_ptr - C_L2) % bank.K_L1
        with torch.no_grad():
            bank.buf_L2_w_full[l2_ptr] = bank.buf_L1_w_full[
                torch.arange(start, start + C_L2) % bank.K_L1].mean(0)

        # L3 update: average all L2 entries every C_L2*K_L3 chunks
        C_L3 = cfg.get('K_L3', 32)
        if bank._L1_ptr % (C_L2 * C_L3) == 0:
            l3_ptr = (bank._L1_ptr // (C_L2 * C_L3) - 1) % bank.K_L3
            with torch.no_grad():
                bank.buf_L3_w_full[l3_ptr] = bank.buf_L2_w_full.mean(0)

    return L_vq  # noqa: F821 - idxs computed but unused (L2 ptr computed inline)


def vq_telescope_retrieve(s_l_full_query, bank, return_ids=False):
    """§1.59 VQ-Telescope: retrieve via routing weight space similarity.

    s_l_full_query: (N_max_l,) float32 full routing weight vector for current token
    Returns: (r_L1, r_L2, r_L3) — reconstructed embeddings as weighted centroid means
    """
    dev = s_l_full_query.device
    n_l = bank.n_l
    zeros = torch.zeros(bank.d_c, dtype=torch.cfloat, device=dev)

    n_l1 = min(bank._L1_ptr, bank.K_L1)
    if n_l1 == 0:
        r_L1 = zeros.clone()
    else:
        sim_L1 = bank.buf_L1_w_full[:n_l1].to(dev) @ s_l_full_query  # (n_l1,)
        top_L1 = int(sim_L1.argmax().item())
        w_L1 = bank.buf_L1_w_full[top_L1, :n_l].unsqueeze(-1).to(torch.cfloat).to(dev)
        r_L1 = (w_L1 * bank.mu_c_l[:n_l]).sum(0)

    n_l2 = min(bank._L1_ptr // max(bank.K_L2, 1), bank.K_L2)
    if n_l2 == 0:
        r_L2 = zeros.clone()
    else:
        sim_L2 = bank.buf_L2_w_full[:n_l2].to(dev) @ s_l_full_query
        top_L2 = int(sim_L2.argmax().item())
        w_L2 = bank.buf_L2_w_full[top_L2, :n_l].unsqueeze(-1).to(torch.cfloat).to(dev)
        r_L2 = (w_L2 * bank.mu_c_l[:n_l]).sum(0)

    n_l3 = min(bank._L1_ptr // max(bank.K_L2 * bank.K_L3, 1), bank.K_L3)
    if n_l3 == 0:
        r_L3 = zeros.clone()
    else:
        sim_L3 = bank.buf_L3_w_full[:n_l3].to(dev) @ s_l_full_query
        top_L3 = int(sim_L3.argmax().item())
        w_L3 = bank.buf_L3_w_full[top_L3, :n_l].unsqueeze(-1).to(torch.cfloat).to(dev)
        r_L3 = (w_L3 * bank.mu_c_l[:n_l]).sum(0)

    if return_ids:
        top_ids = bank.buf_L1_ids[top_L1] if n_l1 > 0 else None  # noqa: F821
        return r_L1, r_L2, r_L3, top_ids
    return r_L1, r_L2, r_L3


class TelescopingMemory:
    """3-level hierarchical FIFO. .mH fix (v5.9.3)."""
    def __init__(self, d_c, K_L1=128, K_L2=32, K_L3=32, C_chunk=32, beta=1.0):
        self.d_c = d_c; self.K_L1 = K_L1; self.K_L2 = K_L2; self.K_L3 = K_L3
        self.C_chunk = C_chunk; self.beta = beta
        self.buf_L1 = torch.zeros(d_c, K_L1, dtype=torch.cfloat)
        self.buf_L2 = torch.zeros(d_c, K_L2, dtype=torch.cfloat)
        self.buf_L3 = torch.zeros(d_c, K_L3, dtype=torch.cfloat)
        self._ptr_L1 = self._ptr_L2 = self._ptr_L3 = 0
        self._fill_L1 = self._fill_L2 = self._fill_L3 = 0
        self._accum_L2 = []; self._accum_L3 = []
        self._pending_L2 = None; self._pending_L3 = None

    def _apply(self, fn):
        self.buf_L1 = fn(self.buf_L1)
        self.buf_L2 = fn(self.buf_L2)
        self.buf_L3 = fn(self.buf_L3)
        if self._pending_L2 is not None: self._pending_L2 = fn(self._pending_L2)
        if self._pending_L3 is not None: self._pending_L3 = fn(self._pending_L3)
        return self

    def to(self, device):
        self.buf_L1 = self.buf_L1.to(device)
        self.buf_L2 = self.buf_L2.to(device)
        self.buf_L3 = self.buf_L3.to(device)
        if self._pending_L2 is not None: self._pending_L2 = self._pending_L2.to(device)
        if self._pending_L3 is not None: self._pending_L3 = self._pending_L3.to(device)
        return self

    @torch.no_grad()
    def add_L1(self, c1):
        self.buf_L1[:, self._ptr_L1] = c1
        self._ptr_L1 = (self._ptr_L1 + 1) % self.K_L1
        self._fill_L1 = min(self._fill_L1 + 1, self.K_L1)
        self._accum_L2.append(c1.clone())
        if len(self._accum_L2) >= self.K_L2:
            self._pending_L2 = torch.stack(self._accum_L2).mean(0); self._accum_L2 = []
        else:
            self._pending_L2 = None

    @torch.no_grad()
    def add_L2(self, c2):
        self.buf_L2[:, self._ptr_L2] = c2
        self._ptr_L2 = (self._ptr_L2 + 1) % self.K_L2
        self._fill_L2 = min(self._fill_L2 + 1, self.K_L2)
        self._accum_L3.append(c2.clone())
        if len(self._accum_L3) >= self.K_L3:
            self._pending_L3 = torch.stack(self._accum_L3).mean(0); self._accum_L3 = []
        else:
            self._pending_L3 = None

    @torch.no_grad()
    def add_L3(self, c3):
        self.buf_L3[:, self._ptr_L3] = c3
        self._ptr_L3 = (self._ptr_L3 + 1) % self.K_L3
        self._fill_L3 = min(self._fill_L3 + 1, self.K_L3)

    def retrieve_all(self, x_c_query, beta=None):
        beta = beta or self.beta; device = x_c_query.device

        def hop(buf, n):
            if n == 0: return torch.zeros_like(x_c_query)
            Xi = buf[:, :n].to(device)
            w = torch.softmax((x_c_query @ Xi.conj()).real * beta, dim=-1).to(torch.cfloat)
            return w @ Xi.mH
        return hop(self.buf_L1, self._fill_L1), hop(self.buf_L2, self._fill_L2), hop(self.buf_L3, self._fill_L3)

    def reset(self):
        with torch.no_grad():
            self.buf_L1.zero_(); self.buf_L2.zero_(); self.buf_L3.zero_()
        self._ptr_L1 = self._ptr_L2 = self._ptr_L3 = 0
        self._fill_L1 = self._fill_L2 = self._fill_L3 = 0
        self._accum_L2 = []; self._accum_L3 = []
        self._pending_L2 = None; self._pending_L3 = None

    @property
    def coverage_tokens(self):
        C = self.C_chunk
        return self._fill_L1 * C + self._fill_L2 * self.K_L2 * C + self._fill_L3 * self.K_L3 * self.K_L2 * C
