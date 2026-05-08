import torch
import torch.nn as nn

from cfln.utils import hippo_legs_init


class ComplexLRU(nn.Module):
    """HiPPO-LegS init. |λ_j|<1 → stable, no BPTT. ~200 token soft context."""
    def __init__(self, d_c, d_ssm=32, S_f=32):
        super().__init__()
        lam = hippo_legs_init(d_ssm)
        self.log_nu = nn.Parameter(torch.log(lam.abs()))
        self.theta  = nn.Parameter(lam.angle())
        self.B_c    = nn.Parameter((torch.randn(d_ssm, d_c) + 1j*torch.randn(d_ssm, d_c)).to(torch.cfloat) / d_c**0.5)
        self.C_c    = nn.Parameter((torch.randn(S_f, d_ssm) + 1j*torch.randn(S_f, d_ssm)).to(torch.cfloat) / d_ssm**0.5)
        # v6.0.6: selective gating — input-dependent λ perturbation
        self.W_select = nn.Parameter(torch.zeros(d_ssm, d_c))
        self.register_buffer('h', torch.zeros(d_ssm, dtype=torch.cfloat))
        self._h_batch = None

    @property
    def lambda_(self):
        return torch.exp(torch.complex(self.log_nu.clamp(max=-0.01), self.theta))

    def step(self, e_c):
        """Batch-mean mode (legacy). e_c: (d_c,) → (S_f,). v6.0.6: selective λ."""
        lam_eff = self.lambda_ * (1.0 + 0.1*torch.sigmoid(self.W_select @ e_c.real))
        h_new = lam_eff * self.h + self.B_c @ e_c
        out = self.C_c @ h_new
        self.h = h_new.detach()
        return out

    def step_per_sequence(self, e_c):
        """Per-sequence mode. e_c: (B,d_c) → (B,S_f). No cross-sequence contamination."""
        B = e_c.shape[0]
        if self._h_batch is None or self._h_batch.shape[0] != B:
            self._h_batch = torch.zeros(B, self.B_c.shape[0], dtype=torch.cfloat, device=e_c.device)
        # v6.0.6: selective gating per-sequence
        lam_eff = self.lambda_.unsqueeze(0) * (1.0 + 0.1*torch.sigmoid(e_c.real @ self.W_select.T))  # (B,d_ssm)
        h_new = lam_eff * self._h_batch + e_c @ self.B_c.conj().T
        out = h_new @ self.C_c.conj().T
        self._h_batch = h_new.detach()
        return out

    def reset(self):
        with torch.no_grad():
            self.h.zero_()
            if self._h_batch is not None:
                self._h_batch.zero_()

    def enforce_stability(self):
        with torch.no_grad():
            self.log_nu.clamp_(max=-0.01)
