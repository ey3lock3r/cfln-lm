import torch


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

    def to(self, device):
        self.buf_L1 = self.buf_L1.to(device)
        self.buf_L2 = self.buf_L2.to(device)
        self.buf_L3 = self.buf_L3.to(device)
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
