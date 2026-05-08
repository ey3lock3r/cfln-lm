import math
import torch
import torch.nn as nn

from cfln.utils import complex_rope_multiplicative


class TitansComplexMemory(nn.Module):
    """
    Titans gradient-based complex memory. v5.9.3.

    v5.9.3 change: titans_query applies CRoPE to Q_t for position-aware retrieval.
    CRoPE was previously in the encoder, contaminating CNEP energies. Now Q_t
    is position-encoded only inside titans_query.
    set_crope_params() called by encoder during construction.
    """
    def __init__(self, d_c, C_chunk=32,
                 eta_init=0.01, theta_decay_init=0.99,
                 null_threshold_init=0.95, k_null=50.0, beta_null_aux=0.01,
                 domain_alpha=0.90, domain_mag_alpha=0.99,
                 domain_threshold_init=3.0, surprise_warmup_chunks=32):
        super().__init__()
        self.d_c = d_c; self.C = C_chunk; self.k_null = k_null
        self.beta_null_aux = beta_null_aux
        self.domain_alpha = domain_alpha; self.domain_mag_alpha = domain_mag_alpha
        self._domain_warmup = surprise_warmup_chunks
        self._use_crope = False; self._rope_base = 10000.0

        for n in ['W_K', 'W_V', 'W_Q']:
            setattr(self, n, nn.Parameter(
                (torch.randn(d_c, d_c) + 1j*torch.randn(d_c, d_c)).to(torch.cfloat) / d_c**0.5))
        self.log_eta = nn.Parameter(torch.log(torch.tensor(eta_init)))
        self.w_theta = nn.Parameter(
            torch.zeros(d_c) + math.log(theta_decay_init / (1.0 - theta_decay_init)))
        self.log_null_threshold = nn.Parameter(
            torch.tensor(math.log(null_threshold_init / (1.0 - null_threshold_init))))
        self.log_domain_threshold = nn.Parameter(torch.log(torch.tensor(domain_threshold_init)))

        self.register_buffer('M', torch.zeros(d_c, d_c, dtype=torch.cfloat))
        self.register_buffer('_prev_e_c', torch.zeros(d_c, dtype=torch.cfloat))
        self.register_buffer('_s_mag_ema', torch.tensor(1.0))
        self.register_buffer('_s_domain_ema', torch.tensor(0.0))
        self._has_prev = False; self.domain_shift_detected = False
        self._chunk_count = 0; self._chunk_accum = []; self._null_aux_loss = None
        self._s_norm_last = 1.0
        self._in_thinking_mode = False

    def set_crope_params(self, use_crope: bool, rope_base: float):
        self._use_crope = use_crope; self._rope_base = rope_base

    @property
    def null_threshold(self): return torch.sigmoid(self.log_null_threshold)
    @property
    def domain_threshold(self): return torch.exp(self.log_domain_threshold).clamp(1.1, 20.0)

    def _update_domain_detector(self, s_t):
        with torch.no_grad():
            self._s_mag_ema = (self.domain_mag_alpha * self._s_mag_ema
                               + (1.0 - self.domain_mag_alpha) * s_t)
            if self._chunk_count <= self._domain_warmup: return
            s_norm = s_t / (float(self._s_mag_ema.item()) + 1e-8)
            self._s_norm_last = s_norm
            self._s_domain_ema = (self.domain_alpha * self._s_domain_ema
                                  + (1.0 - self.domain_alpha) * s_norm)
            self.domain_shift_detected = (float(self._s_domain_ema.item())
                                          > float(self.domain_threshold.detach()))

    def step_chunk(self, e_c_mean):
        self._chunk_count += 1
        eta = torch.exp(self.log_eta).clamp(1e-4, 0.1)
        K_t = self.W_K @ e_c_mean; V_t = self.W_V @ e_c_mean; Q_t = self.W_Q @ e_c_mean
        theta_t = torch.sigmoid((self.w_theta * e_c_mean.real).sum()).clamp(0.01, 0.9999)
        y_hat = self.M @ K_t; e_t = y_hat - V_t
        s_t = float((e_t.conj() * e_t).real.sum().item())
        self._update_domain_detector(s_t)
        if self._has_prev:
            e_n = e_c_mean / e_c_mean.norm().clamp(1e-8)
            p_n = self._prev_e_c / self._prev_e_c.norm().clamp(1e-8)
            cos = (e_n.conj() * p_n).real.sum()
            uw = 1.0 - torch.sigmoid(self.k_null * (cos - self.null_threshold))
        else:
            cos = torch.tensor(0.0); uw = torch.tensor(1.0)
        if self._in_thinking_mode:
            M_new = self.M
        else:
            delta_M = uw * eta * torch.outer(e_t, K_t.conj()); M_new = theta_t * self.M - delta_M
        r_t = M_new @ Q_t
        if self.beta_null_aux > 0:
            r_n = r_t / r_t.norm().clamp(1e-8); V_n = V_t / V_t.norm().clamp(1e-8)
            self._null_aux_loss = (self.beta_null_aux * (1.0 - uw) * (1.0 - (r_n.conj() * V_n).real.sum()))
        else:
            self._null_aux_loss = torch.tensor(0.0)
        self.M = M_new.detach()
        with torch.no_grad(): self._prev_e_c.copy_(e_c_mean.detach())
        self._has_prev = True
        return r_t, s_t, float(cos.item()), float(uw.item())

    def accumulate(self, e_c): self._chunk_accum.append(e_c.detach().mean(0))

    def maybe_step(self):
        if len(self._chunk_accum) >= self.C:
            e_mean = torch.stack(self._chunk_accum).mean(0)
            r, s, cs, uw = self.step_chunk(e_mean)
            self._chunk_accum = []; return r, s, cs, uw, True
        return None, 0.0, 0.0, 1.0, False

    def titans_query(self, e_c: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """v5.9.3: CRoPE applied to Q_t for position-aware retrieval."""
        q = self.W_Q @ e_c
        if self._use_crope and pos > 0:
            q = complex_rope_multiplicative(q.unsqueeze(0), pos, self.d_c, self._rope_base).squeeze(0)
        return self.M @ q

    def reset_to_neutral(self):
        with torch.no_grad():
            self.M.zero_(); self._prev_e_c.zero_()
            self._s_mag_ema.fill_(1.0); self._s_domain_ema.fill_(0.0)
        self._chunk_accum = []; self._has_prev = False
        self.domain_shift_detected = False; self._chunk_count = 0
        self._s_norm_last = 1.0

    def get_surprise(self, e_c_mean):
        with torch.no_grad():
            K_t = self.W_K @ e_c_mean; V_t = self.W_V @ e_c_mean
            return float(((self.M @ K_t - V_t).conj() * (self.M @ K_t - V_t)).real.sum().item())
