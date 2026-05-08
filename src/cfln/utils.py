import math
import heapq  # noqa: F401 — required by spec imports
import torch
import torch.nn as nn  # noqa: F401 — required by spec imports
import torch.nn.functional as F  # noqa: F401 — required by spec imports
from collections import deque  # noqa: F401 — required by spec imports


def to_real(x_c: torch.Tensor) -> torch.Tensor:
    """(B, d_c) complex → (B, 2*d_c) float32. Output boundary only."""
    return torch.view_as_real(x_c).reshape(*x_c.shape[:-1], x_c.shape[-1]*2)


def complex_layer_norm(x_c: torch.Tensor, dims: list,
                        eps: float=1e-5) -> torch.Tensor:
    """
    Phase-preserving complex normalisation using RMS. v5.9.3.

    Uses RMS normalisation (NOT zero-mean layer norm).

    v5.9.2 used F.layer_norm on magnitudes. F.layer_norm computes
    (mag - mean(mag))/std(mag), which produces negative values for features
    with below-average magnitude. Negative scale → phase flip by π.
    That corrupted CRoPE, GAT phase injection, and CNEP overlap structure.

    RMS norm: scale = 1/RMS(|z|) > 0 always → arg(z) preserved exactly.
    Output: mean(|z_k|²) = 1 across the feature dimension.
    """
    mag_sq = x_c.real.pow(2) + x_c.imag.pow(2)                    # (B, d_c) ≥ 0
    rms    = mag_sq.mean(dim=-1, keepdim=True).add(eps).sqrt()     # (B, 1) > 0 always
    return x_c / rms                                               # scale always positive

layer_norm_c = complex_layer_norm   # backward-compatible alias


def tanh_c(z: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


# silu_c removed v5.9.5 — unused
def init_stiefel(d_e: int, d_c: int) -> torch.Tensor:
    Z = (torch.randn(d_e,d_c)+1j*torch.randn(d_e,d_c)) / math.sqrt(2)
    Q, _ = torch.linalg.qr(Z.conj().T)
    return Q.conj().T


def init_unitary(d: int) -> torch.Tensor:
    Z = (torch.randn(d,d)+1j*torch.randn(d,d)) / math.sqrt(2)
    Q, R = torch.linalg.qr(Z)
    phase = torch.exp(-1j * torch.angle(torch.diag(R)))
    return Q * phase.unsqueeze(0)


def verify_stiefel(W: torch.Tensor, tol: float=1e-5) -> bool:
    I = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)  # noqa: E741
    err = (W @ W.conj().T - I).abs().max()
    return err.item() < tol


def normalize_complex_center(mu: torch.Tensor) -> torch.Tensor:
    norms = mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return mu / norms


def compute_ntk_rope_base(d_c: int, L_train: int=2048,
                           L_target: int=1_048_576,
                           base_orig: float=10000.0) -> float:
    """NTK-aware RoPE base. R7."""
    assert d_c > 2, f'v6.0.3 M1: d_c must be > 2 for NTK RoPE (d_c/(d_c-2) undefined at d_c=2), got d_c={d_c}'
    return base_orig * ((L_target / L_train) ** (d_c / (d_c - 2)))


def complex_rope_multiplicative(x_c: torch.Tensor, t: int, d_c: int,
                                 rope_base: float=10000.0) -> torch.Tensor:
    """Multiplicative CRoPE. |exp(iθ)|=1 — magnitude preserved. R7."""
    k     = torch.arange(d_c, dtype=torch.float32, device=x_c.device)
    theta = 1.0 / (rope_base ** (2.0 * k / d_c))
    return x_c * torch.exp(1j * t * theta)


def entmax15_fast(z: torch.Tensor, dim: int=-1,
                   max_iter: int=50, tol: float=1e-6) -> torch.Tensor:
    """entmax-1.5 with early termination. Converges in ~15 iterations typically."""
    z_s = z - z.max(dim=dim, keepdim=True).values
    lo  = z_s.max(dim=dim, keepdim=True).values.clone()   # tau too large: p_sum < 1
    hi  = z_s.min(dim=dim, keepdim=True).values - 1.0     # tau too small: p_sum >= 1
    for _ in range(max_iter):
        mid   = (lo + hi) / 2
        p     = (z_s - mid).clamp(min=0).pow(2)
        p_sum = p.sum(dim=dim, keepdim=True)
        lo    = torch.where(p_sum < 1, mid, lo)
        hi    = torch.where(p_sum >= 1, mid, hi)
        if (p_sum - 1.0).abs().max().item() < tol:
            break
    tau = (lo + hi) / 2
    return (z_s - tau).clamp(min=0).pow(2)

entmax15 = entmax15_fast   # alias

def entmax15_with_floor(z: torch.Tensor, eps: float, dim: int=-1) -> torch.Tensor:
    return entmax15_fast(z, dim=dim).clamp(min=eps)


# sparsemax() removed v6.0.8 (global tier removed; no longer used)


def rq_routing(E, log_alpha, log_ell):
    alpha  = torch.exp(log_alpha).clamp(0.1, 10.0)
    ell_sq = torch.exp(2*log_ell).clamp(1e-4)
    return (1.0 + E / ell_sq.unsqueeze(0)) ** (-alpha.unsqueeze(0))


def compute_energies(x_c, W_bank, mu_bank):
    delta = x_c.unsqueeze(1) - mu_bank.unsqueeze(0)
    z     = torch.einsum('ned,bnd->bne', W_bank, delta)
    return (z.conj() * z).real.sum(-1)


def compute_direction_angles_complex(mu_c):
    return torch.angle(mu_c.mean(dim=-1))


# dirichlet_energy_v53 removed v5.9.5 — unused
def apply_psd_to_weight_matrix(W, eps=1e-6):
    W_sym = (W+W.T)/2
    ev,evec = torch.linalg.eigh(W_sym.float())
    return (evec * ev.clamp(eps).unsqueeze(0)) @ evec.T


def batched_apply_psd(W_list: list, eps: float=1e-6) -> list:
    """v6.0.7 PF-2: batch eigh across L layer W_ll matrices for 4-6× GPU speedup.
    W_list: list of L (k_l,k_l) float tensors.
    Returns: list of L PSD-projected tensors (same shapes)."""
    if not W_list:
        return W_list
    W_stack = torch.stack([w.float() for w in W_list])           # (L, k_l, k_l)
    W_sym   = (W_stack + W_stack.transpose(-1,-2)) * 0.5         # enforce symmetry
    ev, evec = torch.linalg.eigh(W_sym)                          # (L, k_l), (L, k_l, k_l)
    W_psd   = (evec * ev.clamp(eps).unsqueeze(-2)) @ evec.transpose(-1,-2)  # (L, k_l, k_l)
    return [W_psd[i].to(W_list[i].dtype) for i in range(len(W_list))]


def batched_cayley_retraction(W, G, lr):
    GWH = G@W.conj().transpose(-1,-2)
    A = GWH-GWH.conj().transpose(-1,-2)
    n,d_e,_ = W.shape
    I = torch.eye(d_e,dtype=W.dtype,device=W.device).unsqueeze(0).expand(n,-1,-1)  # noqa: E741
    return torch.linalg.solve(I+(lr/2)*A,(I-(lr/2)*A)@W)


def batched_cayley_with_per_unit_lr(W, G, lr):
    GWH = G@W.conj().transpose(-1,-2)
    A = GWH-GWH.conj().transpose(-1,-2)
    n,d_e,_ = W.shape
    I = torch.eye(d_e,dtype=W.dtype,device=W.device).unsqueeze(0).expand(n,-1,-1)  # noqa: E741
    lr_h=(lr/2).view(n,1,1)
    return torch.linalg.solve(I+lr_h*A,(I-lr_h*A)@W)


def cayley_retraction_single(W, G, lr):
    GWH = G@W.conj().T
    A = GWH-GWH.conj().T
    I = torch.eye(W.shape[0],dtype=W.dtype,device=W.device)  # noqa: E741
    return torch.linalg.solve(I+(lr/2)*A,(I-(lr/2)*A)@W)


def stiefel_update_all_v51(bank, lr_l=0.001, lr_p=0.0001):
    if lr_l>0 and bank.W_l.grad is not None:
        bank.W_l.data.copy_(batched_cayley_retraction(bank.W_l.data,bank.W_l.grad,lr_l))
        bank.W_l.grad=None
    # W_g Stiefel update removed v6.0.8 (global tier removed)
    if lr_p>0 and bank.W_p.grad is not None:
        bank.W_p.data.copy_(batched_cayley_retraction(bank.W_p.data,bank.W_p.grad,lr_p))
        bank.W_p.grad=None


def stiefel_update_cun(diff_aux, lr):
    for W in [diff_aux.cun.U1, diff_aux.cun.U2]:
        if W.grad is not None:
            W.data.copy_(cayley_retraction_single(W.data, W.grad, lr))
            W.grad=None


def hippo_legs_init(d_ssm):
    j=torch.arange(1,d_ssm+1,dtype=torch.float32)
    return torch.exp(torch.complex(-(2*j-1)/d_ssm, 2*math.pi*j/d_ssm))


# NOTE: update_titans_neutral() REMOVED — M_neutral parameter removed from Titans.
# Titans self-corrects via Wirtinger updates; no neutral state needed.


def detect_domain_boundary(monitor, drop_threshold=0.30, window=10):
    """M4 channel: detect routing diversity drop (slow domain shift signal)."""
    hist=monitor.E_D_history
    if len(hist)<window:
        return False
    means=[]
    for _,ev in hist[-window:]:
        valid=[e for e in (ev if isinstance(ev,list) else [ev]) if e and e>0]
        means.append(sum(valid)/len(valid) if valid else 0.0)
    if len(means)<window:
        return False
    first=sum(means[:window//2])/(window//2)
    second=sum(means[window//2:])/(window//2)
    if first<1e-8:
        return False
    return (first-second)/first>drop_threshold
