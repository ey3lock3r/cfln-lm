import torch
import torch.nn as nn


class PSCLoss(nn.Module):
    """v6.0.5: Predictive State Compression loss for CTP reasoning pre-training.
    Three self-supervised components — no task labels required.
    W_pred is a TRAINING SCAFFOLD: trained here, not used at inference.
    """
    def __init__(self, d_c: int, d_r_lista: int, n_future: int=3,
                 margin: float=0.1, alpha: float=1.0,
                 beta_max: float=0.1, gamma: float=0.5):
        super().__init__()
        # Training scaffold: predicts future h_N from r_lista^K
        # d_c × d_r_lista = 128×32 = 4096 complex = 8K real params
        self.W_pred=nn.Parameter(
            (torch.randn(d_c,d_r_lista)+1j*torch.randn(d_c,d_r_lista)).to(torch.cfloat)
            /d_r_lista**0.5)
        self.margin=margin; self.alpha=alpha
        self.beta_max=beta_max; self.gamma=gamma; self.n_future=n_future

    def forward(self,
                ce_baseline: torch.Tensor,   # scalar, detached — from Pass1 (free)
                ce_thinking: torch.Tensor,   # scalar, differentiable
                r_lista_K:   torch.Tensor,   # (d_r_lista,) complex — after K think steps
                r_lista_0:   torch.Tensor,   # (d_r_lista,) complex — before thinking
                u_epi_now:   float,          # U_epistemic for current token
                future_h_N:  torch.Tensor,   # (n_future, d_c) complex — h_N at t+3..t+5
                future_u_epi:torch.Tensor    # (n_future,) float — U_epi at future positions
               ) -> torch.Tensor:
        # L_improve: soft hinge — thinking must beat no-thinking
        delta_ce=ce_baseline.detach()-ce_thinking   # positive = improvement
        L_improve=-torch.log(torch.sigmoid(delta_ce+self.margin))

        # L_economy: minimal-change — weighted by (1-U_epi)
        r_delta=r_lista_K-r_lista_0
        L_economy=((r_delta.conj()*r_delta).real.sum())
        beta_eff=self.beta_max*(1.0-float(u_epi_now))

        # L_predictive: predict future hard-token LISTA states (scaffold, detached)
        h_pred=self.W_pred@r_lista_K.detach()            # (d_c,) — W_pred is the only trained param here
        fut_targets=future_h_N.detach()                  # (n_future,d_c)
        pred_errs=((h_pred.unsqueeze(0)-fut_targets).conj()
                   *(h_pred.unsqueeze(0)-fut_targets)).real.sum(-1)  # (n_future,)
        L_predictive=(future_u_epi.to(pred_errs.device)*pred_errs).mean()

        return self.alpha*L_improve + beta_eff*L_economy + self.gamma*L_predictive
