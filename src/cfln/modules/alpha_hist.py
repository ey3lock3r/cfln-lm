import math
import torch


class AlphaHistogram:
    """v5.9.3: log-alpha bins. v5.9.2 used linear bins → spike at bin 0."""
    N_BINS = 16; LOG_ALPHA_MIN = -6.0; LOG_ALPHA_MAX = 0.0

    def __init__(self, N_max_l=16384):
        self.counts = torch.zeros(self.N_BINS, dtype=torch.long)
        self.n_units = 0

    @torch.no_grad()
    def update(self, alpha_l: torch.Tensor):
        self.n_units = len(alpha_l)
        log_alpha = torch.log(alpha_l.clamp(1e-6, 1.0))
        bins = ((log_alpha - self.LOG_ALPHA_MIN) / (self.LOG_ALPHA_MAX - self.LOG_ALPHA_MIN)
                * self.N_BINS).long().clamp(0, self.N_BINS - 1)
        self.counts = self.counts.to(bins.device)
        self.counts.zero_(); self.counts.scatter_add_(0, bins, torch.ones_like(bins, dtype=torch.long))

    def percentile(self, pct):
        if self.n_units == 0: return 0.7
        target = int(pct * self.n_units); cumsum = 0
        for k in range(self.N_BINS):
            cumsum += int(self.counts[k].item())
            if cumsum >= target:
                log_thresh = (self.LOG_ALPHA_MIN + (k+1)/self.N_BINS * (self.LOG_ALPHA_MAX - self.LOG_ALPHA_MIN))
                return float(math.exp(log_thresh))
        return 1.0

    def get_alpha_freeze(self, sensory_fraction=0.15):
        return self.percentile(1.0 - sensory_fraction)
