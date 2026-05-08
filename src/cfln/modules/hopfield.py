import math
import torch
import torch.nn as nn


class HopfieldRetrieval(nn.Module):
    """Parametric Hopfield. k_max capacity limit (v5.9.3). Owned by IterativeRefinementModule."""
    def __init__(self, beta=1.0, max_steps=3, eps=1e-3):
        super().__init__()
        self.max_steps = max_steps
        self.eps = eps
        self._last_confidence: float = 0.0
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))

    @staticmethod
    def capacity_k_max(n_l: int, d_c: int) -> int:
        return max(4, int(0.10 * n_l * n_l / d_c))

    def forward(self, x_c: torch.Tensor, mu_c_l: torch.Tensor) -> torch.Tensor:
        beta = torch.exp(self.log_beta).clamp(0.1, 20.0); n, d_c = mu_c_l.shape[0], x_c.shape[-1]
        k_max = self.capacity_k_max(n, d_c)
        if n > k_max:
            sims = (x_c @ mu_c_l.conj().T).real; top_idx = torch.topk(sims.mean(0), k_max).indices
            mu_subset = mu_c_l[top_idx]
        else:
            mu_subset = mu_c_l
        Xi = mu_subset.T; x = x_c.clone()
        for _ in range(self.max_steps):
            w = torch.softmax((x @ Xi.conj()).real * beta, dim=-1)
            x_new = w.to(torch.cfloat) @ mu_subset
            vel = float(((x_new - x).conj() * (x_new - x)).real.sum(-1).sqrt()
                        .div((x.conj() * x).real.sum(-1).sqrt().clamp(1e-8)).max())
            x = x_new
            if vel < self.eps: break
        w_entropy = -(w * (w + 1e-10).log()).sum(dim=-1).mean()
        self._last_confidence = float((w_entropy / (math.log(mu_subset.shape[0] + 1) + 1e-8)).item())
        return x

    def retrieve_with_field(self, x_c, bank):
        return self.forward(x_c, bank.mu_c_l[:bank.n_l])
