import torch
import torch.nn as nn


class ComplexGATLayer(nn.Module):
    """v6.0.6 CS-GAT: Chebyshev Spectral Graph Convolution on Hermitian adjacency.
    Replaces 4-scale dot-product attention with K_cheby=3 polynomial hops.
    """
    K_CHEBY = 3

    def __init__(self, d_c, n_heads=4, dropout=0.0):
        super().__init__()
        self.d_c = d_c
        self.W_in = nn.Parameter(
            (torch.randn(d_c, d_c) + 1j*torch.randn(d_c, d_c)).to(torch.cfloat) / d_c**0.5)
        self.theta_cheby = nn.ParameterList([
            nn.Parameter(torch.ones(d_c, dtype=torch.cfloat) / (self.K_CHEBY + 1))
            for _ in range(self.K_CHEBY + 1)])
        self.W_final = nn.Parameter(
            (torch.randn(d_c, d_c) + 1j*torch.randn(d_c, d_c)).to(torch.cfloat) / d_c**0.5)

    def forward(self, psi_all, theta_all, W_full):
        """
        psi_all: (k_l, d_c) complex — node features
        theta_all: (k_l,) real — unit phases (unused in CS-GAT; kept for API compat)
        W_full: (k_l, k_l) real — PSD overlap adjacency (Hermitian symmetrised)
        Returns: (k_l, d_c) complex — spectral-filtered node features
        """
        h = psi_all @ self.W_in.conj().T
        Adj = W_full.to(torch.cfloat)
        A_herm = (Adj + Adj.conj().T) * 0.5
        d_inv = 1.0 / (A_herm.real.sum(-1).clamp(min=1e-6)**0.5)
        A_norm = d_inv.unsqueeze(-1) * A_herm * d_inv.unsqueeze(0)
        T = [h, A_norm @ h]
        for _ in range(2, self.K_CHEBY + 1):
            T.append(2 * A_norm @ T[-1] - T[-2])
        out = sum(T[k] * self.theta_cheby[k] for k in range(self.K_CHEBY + 1))
        return out @ self.W_final.conj().T
