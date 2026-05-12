# CFLN v9.0 — Complete Standalone Implementation Plan
## All changes from v6.0.9 to v9.0 · For AI verification and implementation

**Base:** `CFLN_v609_Master_Spec.md` (5,758 lines, 68 tests, fully runnable)
**Change spec:** `CFLN_v900_Spec.md` (§1.30–§1.77, 48 sections)

This document is self-contained. Read it fully before touching code.
Cross-reference the specs only for full function bodies not reproduced here.

---

## PART A — CRITICAL TYPE FACTS (memorise before writing any code)

### A.1 Types that are NOT what their name suggests

```
bank.alpha_freeze         → scalar float (histogram percentile threshold, e.g. 0.7)
                            NEVER subscriptable. bank.alpha_freeze[i] → TypeError.

bank.is_sensory_l         → (N_max_l,) bool Tensor — the per-unit frozen flag.
                            bank.is_sensory_l[i] == True ⟺ unit i is frozen.
                            This is what §1.57 Rule 3 and §1.63 must use.

bank.W_l                  → (N_max_l, d_e_l, d_c) cfloat — indexed per unit.
                            bank.W_l[i] = unit i's projection matrix.

bank.mu_c_l               → (N_max_l, d_c) cfloat — indexed per unit.
```

### A.2 What is trained vs fixed

```
TRAINED (nn.Parameter, in optimizer):
  All bank.log_* scalars, bank.role_vecs, bank.W_goal_detect
  bank.W_rc_bridge      ← WAS register_buffer, NOW nn.Parameter (§1.50)
  cun.log_alpha_arc, cun.log_w_meta, cun.tau_smooth
  cun.eps_beam_scale, cun.log_w_beam, cun.log_blend_alpha
  bank.log_cal_scale    (§1.72)

NEVER TRAINED (excluded from all optimizers):
  cun.U1, cun.U2        ← fixed Haar-random unitary basis, forever
  bank.role_vecs is in opt_g (AdamW), NOT Muon

Fisher dict             → accumulation buffer only, not a parameter
sigma_sq_buffer         → plain Python list [float], not a Tensor, not a param
log_precision           → plain Python list [float], not a Tensor, not a param
```

### A.3 Gradient flow rules

```
r_lista                 → always .detach() before use as input (no BPTT)
W_rc_bridge             → trained via L_bridge (local loss, not BPTT through r_lista)
mu_c_l in L_vq          → MUST be detached: mu_c_l[sel_l].mean(0).detach()
chunk_mean in L_compress → must NOT be detached (§1.51 fix — was detached, now fixed)
Fisher accumulation     → BEFORE clip_grad_norm_, AFTER loss.backward()
```

### A.4 Initialisation traps

```
sigma_sq_buffer: [1.0]*5    NOT [0.0]*5   (zero → precision explosion on step 0)
log_cal_scale:   log(0.15) ≈ -1.897       (not 0.0)
log_blend_alpha: log(0.8)  ≈ -0.223       (not 0.0)
role_vecs:       QR ortho init            (not randn)
fisher dict:     zeros_like each param    (not None — pre-allocate at build time)
```

---

## PART B — NEW VOCABULARY (6 tokens, add before any training)

```python
THINK_START_ID  = original_vocab_size + 0   # <think>
THINK_END_ID    = original_vocab_size + 1   # </think>
HYPO_START_ID   = original_vocab_size + 2   # <hypo>
HYPO_END_ID     = original_vocab_size + 3   # </hypo>
PUSH_GOAL_ID    = original_vocab_size + 4   # <push_goal>
POP_GOAL_ID     = original_vocab_size + 5   # </push_goal>  ← closing tag = POP

# vocab_size used everywhere = original_vocab_size + 6
```

---

## PART C — COMPLETE PARAMETER CHANGES

### C.1 Parameters REMOVED (delete from __init__ and all references)

| Parameter | Was in | Replaced by |
|---|---|---|
| `W_compress_L1` | TelescopingMemory | VQ routing weights (§1.59) |
| `W_compress_L2` | TelescopingMemory | Averaged routing weights |
| `W_compress_L3` | TelescopingMemory | Averaged routing weights |
| `W_decompress_L1` | TelescopingMemory | Centroid mean reconstruction |
| `log_w_meta` | CUN | `log_precision` list (§1.58) |
| `_log_w_rec` | CUN | `sigma_sq_buffer` list (§1.58) |
| ~~mask_embed~~ | ~~CFBank~~ | NOT removed — see §1.44, spec is authoritative |

### C.2 New nn.Parameter — CFBank

```python
# Binding kernels (§1.30, §1.35, §1.41, §1.55)
self.log_lam_bind        = nn.Parameter(torch.tensor(-3.0))
self.log_sigma_bind      = nn.Parameter(torch.tensor(0.693))   # log(2.0)
self.log_lam_role        = nn.Parameter(torch.tensor(-3.0))
self.log_lam_composition = nn.Parameter(torch.tensor(-3.0))    # Hadamard

# Role vectors: QR orthogonal init, AdamW (NOT Muon, NOT Stiefel)
R_roles = cfg.get('n_roles', 8)
_raw = torch.randn(R_roles, d_c, dtype=torch.cfloat)
_Q, _ = torch.linalg.qr(_raw.T)
self.role_vecs = nn.Parameter(_Q.T[:R_roles].contiguous())     # (R, d_c) cfloat

# Goal context (§1.33)
self.W_goal_detect  = nn.Parameter(torch.zeros(1, d_c))
self.log_lam_goal   = nn.Parameter(torch.tensor(-3.0))

# MC-1 calibration scale (§1.72 C9)
self.log_cal_scale  = nn.Parameter(torch.tensor(-1.897))       # log(0.15)
```

### C.3 W_rc_bridge: register_buffer → nn.Parameter (§1.50)

```python
# DELETE: self.register_buffer('W_rc_bridge', W_bridge_init)
# ADD:
d_r_lista = cfg.get('d_r_lista', 32)
d_r_node  = cfg.get('d_r_node',  8)
W_bridge_init = (torch.randn(d_r_lista, d_r_node)
                 + 1j*torch.randn(d_r_lista, d_r_node)).to(torch.cfloat)
W_bridge_init /= d_r_node**0.5
self.W_rc_bridge = nn.Parameter(W_bridge_init)  # trained via L_bridge
# stiefel_ids.add(id(self.W_rc_bridge))  ← add to AdamW exclusions from Muon
```

### C.4 New register_buffer — CFBank

```python
# Goal context (§1.33)
self.register_buffer('g_c', torch.zeros(d_c, dtype=torch.cfloat))

# Verbatim spans (§1.36)
C_chunk = cfg.get('C_chunk', 32)
K_L1    = cfg.get('K_L1', 128)
self.register_buffer('buf_L1_ids', torch.zeros(K_L1, C_chunk, dtype=torch.int32))

# k-shot centroid refinement (§1.43)
self.register_buffer('_proto_count', torch.zeros(N_max_l, dtype=torch.int32))
self.register_buffer('_proto_sum',   torch.zeros(N_max_l, d_c, dtype=torch.cfloat))

# VQ-Telescope (§1.59)
K_L2 = cfg.get('K_L2', 32)
K_L3 = cfg.get('K_L3', 32)
self.register_buffer('buf_L1_w_full', torch.zeros(K_L1, N_max_l))
self.register_buffer('buf_L2_w_full', torch.zeros(K_L2, N_max_l))
self.register_buffer('buf_L3_w_full', torch.zeros(K_L3, N_max_l))

# Fisher per-unit mean (§1.63)
self.register_buffer('fisher_unit', torch.zeros(N_max_l))

# Welford E_min statistics (§1.68/1.69) — plain Python floats, not buffers
self._Emin_mean = 0.0
self._Emin_var  = 0.0
self._Emin_n    = 0
```

### C.5 New nn.Parameter — CUN

```python
# ARC dual-key (§1.31)
self.log_alpha_arc  = nn.Parameter(torch.tensor(0.0))
N_rules = cfg.get('episodic_rule_cache_n', cfg.get('N_rules', 256))  # NOTE: both keys
self.register_buffer('rule_K', torch.zeros(N_rules, 2*d_c, dtype=torch.cfloat))

# U_meta_v4 five signals (§1.34) — extends from 4 to 5
self.log_w_meta = nn.Parameter(torch.tensor([1.0, -1.0, -1.0, -2.0, -2.0]))

# STELA smooth threshold (§1.40)
self.tau_smooth = nn.Parameter(torch.tensor(0.1))

# Beam search (§1.47)
self.eps_beam_scale = nn.Parameter(torch.tensor(0.1))
self.log_w_beam     = nn.Parameter(torch.zeros(3))   # F3, F4, F5 weights

# r_lista blend alpha (§1.73 C10)
self.log_blend_alpha = nn.Parameter(torch.tensor(-0.223))  # log(0.8)
```

### C.6 New session-state attributes — CUN (reset per session, NOT parameters)

```python
# U_meta precision (§1.58) — plain Python lists
self.log_precision      = [0.0, 0.0, 0.0, 0.0, 0.0]
self.sigma_sq_buffer    = [1.0, 1.0, 1.0, 1.0, 1.0]   # MUST be 1.0, NOT 0.0
self._precision_active  = [False, False, False, False, False]

# SSP goal stack (§1.39)
self._goal_stack    = []
self._stuck_count   = []   # Lyapunov monitoring per stack depth
self._v_prev        = []

# HYPO mode (§1.32)
self._in_hypo_mode  = False
self._r_lista_hypo  = None
self._u_hypo        = 0.0

# Beam / Q_BEAM state (§1.47, §1.65)
self._last_Q_BEAM_score = 0.0
self._phi_rel_cache     = None
self._phi_rel_step      = 0

# MC-2 extended to 5 signals (§1.34) — replaces old _log_w_rec [0]*4
self._log_w_rec     = [0.0, 0.0, 0.0, 0.0, 0.0]   # was [0.0]*4 in v6.0.9
```

---

## PART D — OPTIMIZER GROUPS

```python
stiefel_ids = {
    id(bank.W_l),              # Muon (Cayley retraction on Stiefel)
    id(bank.W_p),              # Muon (Cayley retraction on Stiefel)
    id(cun.U1),                # NEVER trained — fixed unitary basis
    id(cun.U2),                # NEVER trained — fixed unitary basis
    id(bank.role_vecs),        # AdamW (shape R×d_c, not square — not Stiefel)
    id(model.W_rc_bridge),     # AdamW (ESN conservative, not Stiefel)
}
# Everything NOT in stiefel_ids → AdamW (opt_g)
# fisher dict → no optimizer (manual buffer)
# sigma_sq_buffer, log_precision → no optimizer (Python lists, self-calibrating)
```

---

## PART E — CONFIG KEY CHANGES

### E.1 Keys REMOVED (now emergent — computed from signals)

```python
# Delete these from your config. If passed, they are silently ignored.
# 'surprise_threshold'    → §1.68 Welford E_min replaces fixed 0.5
# 'ssp_stuck_threshold'   → §1.71 think-budget is the natural timeout
# 'ssp_merge_alpha'       → §1.65 Q_BEAM quality-weighted merge
# 'spawn_threshold'       → §1.69 Welford E_min replaces fixed 3.0
# 'tau_proto'             → §1.70 U_epi-gated (use tau_proto_min instead)
# 'lambda_recon'          → §1.42 superseded by §1.59 VQ-Telescope (use lambda_vq)
```

### E.2 Keys ADDED

```python
# Adaptive routing
'k_l_min':            10,      # §1.64 adaptive k_l lower bound
'k_l_max':            40,      # §1.64 adaptive k_l upper bound (was hard-coded)
# Beam
'beam_B_max':          3,      # §1.66 max beam width
# k-shot
'K_proto_max':        10,      # §1.43 exposures before crystallise
'tau_proto_min':       0.4,    # §1.70 similarity floor (with U_epi gate)
# CL
'beta_KL':             0.5,    # §1.57 Fisher-KL weight (AdamW params)
'beta_SI_stiefel':     0.25,   # §1.57 SI weight (Stiefel params W_l/W_p)
'beta_KL_warmup':      500,    # §1.57 steps to anneal beta_KL from 0 → full
# Training losses
'lambda_bridge':       0.1,    # §1.50 L_bridge (W_rc_bridge predictive coding)
'lambda_vq':           0.01,   # §1.59 L_vq encoder commitment
'lambda_diversity':    0.01,   # §1.47 beam anti-collapse
'lambda_prec':         0.001,  # §1.58 precision entropy regulariser
'lambda_lipschitz':    0.001,  # §1.56 routing sharpness
'lambda_sigma_reg':    0.001,  # §1.56 phase kernel width
'alpha_micro':         0.0001, # §1.54 per-chunk micro-consolidation rate
# Misc
'alpha_young':         0.1,    # §1.43/§1.70 young unit threshold
```

### E.3 Keys CHANGED

```python
'episodic_rule_cache_n': 256,  # was 64 — use both keys: get('episodic_rule_cache_n', get('N_rules', 256))
```

---

## PART F — IMPLEMENTATION BY FILE

### F.1 bank.py / CFBank

**In `__init__`:** Add all parameters from §C.2, §C.3, §C.4.

**In `update_sensory_mask`:** After the existing histogram freeze, add Fisher-magnitude parallel trigger. See §G.2 for the complete `update_fisher_magnitude_freeze` function which calls into `is_sensory_l` the same way.

**In `spawn(idx, x_c)`:** Add at end:
```python
with torch.no_grad():
    self._proto_count[idx] = 1
    self._proto_sum[idx]   = x_c.detach().mean(0) if x_c.dim() > 1 else x_c.detach()
    self.is_sensory_l[idx] = False          # explicitly young
    self.activation_freq_l[idx] = 0.0
```

**k-shot accumulation** (called from CFL5Layer.forward per routed young unit):
```python
def _maybe_refine_centroid(bank, unit_idx, x_c_mean, cfg):
    """§1.43: k-shot centroid refinement for young units."""
    count = int(bank._proto_count[unit_idx].item())
    if count >= cfg.get('K_proto_max', 10):
        return
    freq = float(bank.activation_freq_l[unit_idx].item())
    if freq >= cfg.get('alpha_young', 0.1):
        return   # not young
    # U_epi gate (§1.70): only accumulate when routing is confident
    u_epi = float(getattr(bank, '_last_u_epi_cal', 0.5))
    if u_epi >= 0.4:
        return   # uncertain routing — don't contaminate prototype
    cos_sim = torch.nn.functional.cosine_similarity(
        x_c_mean.real.unsqueeze(0),
        bank.mu_c_l[unit_idx].real.unsqueeze(0)).item()
    if cos_sim < cfg.get('tau_proto_min', 0.4):
        return
    with torch.no_grad():
        bank._proto_count[unit_idx] += 1
        bank._proto_sum[unit_idx]   += x_c_mean.detach()
        bank.mu_c_l.data[unit_idx]   = (bank._proto_sum[unit_idx] /
                                         float(bank._proto_count[unit_idx]))
        if int(bank._proto_count[unit_idx].item()) >= cfg.get('K_proto_max', 10):
            bank.is_sensory_l[unit_idx] = True  # crystallise
```

### F.2 cfl5layer.py / CFL5Layer

**In `forward`**, apply changes in this order:

```python
def forward(self, x_c, bank, cun, cfg, training=True):
    n_l = bank.n_l
    x_c_mean = x_c.mean(0)

    # 1. GOAL CONTEXT §1.33
    if not getattr(bank, '_goal_frozen', False):
        g_t = torch.sigmoid(bank.W_goal_detect @ x_c_mean.real)
        with torch.no_grad():
            bank.g_c = (g_t * x_c_mean + (1 - g_t) * bank.g_c).detach()
    x_c_eff = x_c + torch.exp(bank.log_lam_goal) * bank.g_c.unsqueeze(0)

    # 2. CNEP ROUTING using x_c_eff
    k_l_max = cfg.get('k_l_max', 40)
    # Adaptive k_l §1.64: k_l_eff based on U_epi_cal
    u_epi_cal = float(getattr(bank, '_last_u_epi_cal', 0.5))
    k_l_min  = cfg.get('k_l_min', 10)
    k_l_eff  = k_l_min + round((k_l_max - k_l_min) * u_epi_cal)
    k_l_eff  = max(k_l_min, min(k_l_max, k_l_eff))

    E_l, sel_l, s_l = route(x_c_eff, bank, k_l=k_l_eff)
    E_min_raw   = float(E_l.min(dim=-1).values.mean().item())
    H_route_raw = float(-(s_l * (s_l + 1e-9).log()).sum(-1).mean().item())

    # 3. k-SHOT CENTROID REFINEMENT §1.43/§1.70
    for idx in sel_l:
        _maybe_refine_centroid(bank, int(idx.item()), x_c_mean, cfg)

    # 4. W_full construction with three binding terms
    W_full = existing_W_full_init(bank, sel_l, k_l_eff)

    # 4a. PHASE KERNEL B_bind §1.30/§1.41
    phi_sel = torch.angle(bank.H_c_l[sel_l].mean(dim=(-2, -1)))
    sigma_sq = torch.exp(2.0 * bank.log_sigma_bind)
    phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)
    B_bind = torch.exp(-phi_diff**2 / sigma_sq)                # (k_l, k_l) real
    W_full[:k_l_eff, :k_l_eff] += torch.exp(bank.log_lam_bind) * B_bind

    # 4b. ROLE BINDING B_role §1.35
    alpha_r = torch.softmax(
        (bank.mu_c_l[sel_l] @ bank.role_vecs.conj().T).real / (bank.mu_c_l.shape[-1]**0.5),
        dim=-1)                                                  # (k_l, R)
    B_role = alpha_r @ alpha_r.T                                # (k_l, k_l) real, PSD
    W_full[:k_l_eff, :k_l_eff] += torch.exp(bank.log_lam_role) * B_role

    # 4c. HADAMARD COMPOSITION B_comp §1.55
    B_comp = B_bind * B_role                                    # Schur: PSD ✓
    W_full[:k_l_eff, :k_l_eff] += torch.exp(bank.log_lam_composition) * B_comp

    # 5. CS-GAT aggregation (unchanged from v6.0.9)
    h_filt = cs_gat(W_full, psi_all, K_CHEBY=3)

    # 6. LYAPUNOV GOAL PROXY §1.48 (cached, recomputed only when g_c changes)
    r_lista_goal_proxy = _get_goal_proxy(self, bank, cun)

    # 7. L_bridge (W_rc_bridge predictive coding §1.50)
    L_bridge_val = None
    if training:
        rho_weighted = _get_rho_weighted(bank, sel_l, s_l)
        r_seed = self.W_rc_bridge @ rho_weighted if hasattr(self, 'W_rc_bridge') \
                 else bank.W_rc_bridge @ rho_weighted
        with torch.no_grad():
            d_r = r_seed.shape[0]
            r_seed_target = (cun.U1.conj() @ x_c_mean)[:d_r]
        L_bridge_val = (r_seed_target.detach() - r_seed).norm()**2

    # 8. lista_forward call
    h_N, meta_info = cun.lista_forward(
        x_c, hopfield=hopfield, bank=bank,
        u_temporal=u_temporal_val, u_hypo=bank._u_hypo,
        r_lista_goal=r_lista_goal_proxy,
        E_min_raw=E_min_raw, H_route_raw=H_route_raw,
        k_l_eff=k_l_eff)

    return h_N, meta_info, {'L_bridge': L_bridge_val,
                             'E_min_raw': E_min_raw,
                             'H_route_raw': H_route_raw,
                             'sel_l': sel_l, 's_l': s_l}
```

**`_get_goal_proxy` (cache to avoid recomputing every token):**
```python
def _get_goal_proxy(layer, bank, cun):
    """§1.48: r_lista goal proxy in LISTA basis. Cached."""
    if not hasattr(bank, 'g_c') or bank.g_c.norm() < 1e-6:
        return None
    prev = getattr(layer, '_g_c_prev', None)
    if prev is None or (bank.g_c - prev).norm() > 1e-4:
        layer._r_goal_proxy_cache = (cun.U1 @ bank.g_c.conj()).detach()
        layer._g_c_prev = bank.g_c.clone().detach()
    return layer._r_goal_proxy_cache
```

### F.3 cun.py / CUN.lista_forward

**New kwargs to add:**
```python
def lista_forward(self, x_c, hopfield=None, bank=None, N_hop=4,
                  escape=True, compute_meta=True, u_temporal=0.0,
                  u_hypo=0.0, r_lista_goal=None,
                  E_min_raw=None, H_route_raw=None, k_l_eff=40):
```

**phi_rel computation (§1.31, cached every C_chunk=32 tokens):**
```python
if self._phi_rel_step % 32 == 0 and bank is not None:
    H_seq_sub = bank.H_seq_mat[sel_k][:, sel_k]   # (k_l, k_l)
    try:
        eigvals, eigvecs = torch.linalg.eigh(H_seq_sub.real.float())
        self._phi_rel_cache = eigvecs[:, -1].to(torch.cfloat)   # top eigenvector
    except Exception:
        self._phi_rel_cache = None
self._phi_rel_step += 1
```

**Dual-key ARC retrieval (§1.31):**
```python
# Concept key (unchanged from v6.0.9)
k_concept = x_c.mean(0) @ cun.U1.conj().T

# Relational key (new)
if self._phi_rel_cache is not None:
    q_rel = self._phi_rel_cache @ psi_all[:k_l_eff]   # (d_c,)
else:
    q_rel = k_concept  # fallback

alpha_arc = torch.sigmoid(self.log_alpha_arc)
# sim = alpha * cos(q_concept, K_concept) + (1-alpha) * cos(q_rel, K_rel)
# K_rule shape is now (N_rules, 2*d_c): first d_c = concept, second = relational
```

**STELA smooth threshold (§1.40) — replaces hard threshold in LISTA loop:**
```python
# BEFORE:
h = sign(h) * max(abs(h) - tau, 0)
# AFTER:
h = h * torch.sigmoid((h.abs() - tau) / self.tau_smooth.clamp(min=1e-3))
```

**SE-3 reservoir augmentation (§1.45) — change LISTA reconstruction:**
```python
# BEFORE:
x_c_recon = self.U2 @ h_N
# AFTER:
if bank is not None and hasattr(bank, 'rho_l') and len(sel_l) > 0:
    rho_sel   = bank.rho_l[sel_l].mean(0)
    x_c_recon = self.U2 @ h_N + self.W_dec_res @ rho_sel
else:
    x_c_recon = self.U2 @ h_N
```

**r_lista warm-start blend (§1.73 C10):**
```python
# BEFORE: r_lista = 0.8 * r_lista + 0.2 * r_seed
# AFTER:
blend_alpha = torch.exp(self.log_blend_alpha).clamp(0.5, 0.95)
r_lista_new = blend_alpha * self.r_lista + (1 - blend_alpha) * r_seed
self.r_lista = r_lista_new.detach()
```

**Beam B=2 (§1.47) — inside THINK tokens only:**
```python
in_think = getattr(self, '_in_think_mode', False)
if in_think:
    u_meta_now = ... # compute current U_meta
    B_max = cfg.get('beam_B_max', 3)
    B_eff = max(1, round(1 + u_meta_now * (B_max - 1)))
    if B_eff >= 2:
        noise = torch.randn_like(self.r_lista) * self.eps_beam_scale.abs()
        r_lista_b2 = self.r_lista + noise
        h_b2 = self._lista_inner(x_c, r_lista_b2, N_adaptive, tau)

        Q_b1 = compute_Q_beam(h_N, self.r_lista, r_lista_goal,
                               self._goal_stack, x_c, E_min_raw, H_route_raw,
                               self._phi_rel_cache, self.log_w_beam)
        Q_b2 = compute_Q_beam(h_b2, r_lista_b2, r_lista_goal,
                               self._goal_stack, x_c, E_min_raw, H_route_raw,
                               self._phi_rel_cache, self.log_w_beam)

        w = torch.softmax(torch.stack([Q_b1, Q_b2]), dim=0)
        h_N = w[0] * h_N + w[1] * h_b2
        self.r_lista = (w[0] * self.r_lista + w[1] * r_lista_b2).detach()
        self._last_Q_BEAM_score = float(max(Q_b1.item(), Q_b2.item()))
```

**U_meta_v4 five-signal (§1.34):**
```python
# _log_w_rec is now length 5 (was 4)
U_hypo    = torch.sigmoid(torch.tensor((self.r_lista - self._r_lista_hypo).norm()**2 /
            self.r_lista.shape[0])) if self._in_hypo_mode and self._r_lista_hypo is not None \
            else torch.tensor(0.0)
signals   = [U_repr_q, U_epi_cal, U_hopfield, u_temporal, U_hypo]

# Precision-weighted (§1.58) — replaces softmax(log_w_meta)
_update_precision(self, signals, is_hypo_active=self._in_hypo_mode)
prec      = torch.exp(torch.tensor(self.log_precision))
signals_t = torch.tensor([float(s) if not isinstance(s, torch.Tensor)
                           else s.item() for s in signals])
U_meta    = (prec * signals_t).sum() / (prec.sum() + 1e-8)
```

**SSP PUSH/POP handling (§1.39, §1.52, §1.65):**
```python
# On PUSH_GOAL_ID token:
if len(self._goal_stack) < D_max:
    self._goal_stack.append(self.r_lista.clone())
    self._stuck_count.append(0)
    self._v_prev.append(float('inf'))
    bank._goal_frozen = True   # freeze g_c during subgoal

# Per think token inside PUSH (Lyapunov monitoring §1.52):
if self._goal_stack and r_lista_goal is not None and r_lista_goal.norm() > 1e-4:
    V_curr = float((self.r_lista - r_lista_goal).norm()**2)
    if V_curr < self._v_prev[-1]:
        self._stuck_count[-1] = 0    # improving
    else:
        self._stuck_count[-1] += 1   # not improving
    self._v_prev[-1] = V_curr

# On POP_GOAL_ID token OR think-budget exhausted (§1.71: N_stuck removed):
if self._goal_stack:
    parent = self._goal_stack.pop()
    self._stuck_count.pop()
    self._v_prev.pop()
    last_q = self._last_Q_BEAM_score
    merge_w = torch.sigmoid(torch.tensor(last_q)).item()   # §1.65 Q_BEAM merge
    self.r_lista = ((1 - merge_w) * parent + merge_w * self.r_lista).detach()
    if not self._goal_stack:
        bank._goal_frozen = False

# HYPO mode (§1.32):
# On HYPO_START_ID: self._r_lista_hypo = self.r_lista.clone(); self._in_hypo_mode = True
# On HYPO_END_ID:   restore, compute U_hypo, clear
```

### F.4 telescoping.py / TelescopingMemory

**Replace `maybe_update` compress logic with VQ-Telescope (§1.59):**

```python
def vq_telescope_update(self, chunk_mean, sel_l, s_l, E_min_raw,
                         chunk_token_ids, bank, cfg):
    """§1.59: Store VQ routing code instead of W_compress output."""
    k_l_eff = len(sel_l)
    ptr = bank._L1_ptr % bank.K_L1

    # Build full routing weight vector (N_max_l dimensional, sparse)
    s_l_full = torch.zeros(bank.N_max_l, device=chunk_mean.device)
    s_l_full[sel_l] = s_l[:k_l_eff].float().detach()
    bank.buf_L1_w_full[ptr] = s_l_full

    # Verbatim token IDs §1.36
    bank.buf_L1_ids[ptr] = chunk_token_ids

    # VQ encoder commitment loss §1.59 (gradient to encoder only)
    z_approx = bank.mu_c_l[sel_l].mean(0).detach()   # DETACH codebook
    L_vq = (chunk_mean - z_approx).norm()**2          # gradient to chunk_mean

    # Surprise detection §1.68 (Welford-based, no fixed threshold)
    if bank._Emin_n >= 10:
        sigma_Emin = (bank._Emin_var / bank._Emin_n + 1e-8)**0.5
        if E_min_raw > bank._Emin_mean + 2.0 * sigma_Emin:
            bank.surprise_archive.add_vq(ptr, E_min_raw)

    bank._L1_ptr += 1

    # L2 update (every C_L2 L1 chunks)
    C_L2 = cfg.get('C_L2', 32)
    if bank._L1_ptr % C_L2 == 0:
        l2_ptr = (bank._L1_ptr // C_L2) % bank.K_L2
        start  = (bank._L1_ptr - C_L2) % bank.K_L1
        idxs   = torch.arange(start, start + C_L2) % bank.K_L1
        bank.buf_L2_w_full[l2_ptr] = bank.buf_L1_w_full[idxs].mean(0)

    # L3 update (every C_L2*C_L3 L1 chunks)
    C_L3 = cfg.get('C_L3', 32)
    if bank._L1_ptr % (C_L2 * C_L3) == 0:
        l3_ptr = (bank._L1_ptr // (C_L2 * C_L3)) % bank.K_L3
        bank.buf_L3_w_full[l3_ptr] = bank.buf_L2_w_full.mean(0)

    # §1.68/1.69 Welford E_min update — MUST be here, after all buffer writes
    bank._Emin_n    += 1
    delta            = E_min_raw - bank._Emin_mean
    bank._Emin_mean += delta / bank._Emin_n
    bank._Emin_var  += delta * (E_min_raw - bank._Emin_mean)

    return L_vq


def vq_telescope_retrieve(self, s_l_full_query, bank):
    """§1.59: Retrieve past chunk by routing weight cosine similarity."""
    n_l1 = min(int(bank._L1_ptr), bank.K_L1)
    if n_l1 == 0:
        return None, None, None, None

    # Dot product in full routing weight space
    sim_L1  = bank.buf_L1_w_full[:n_l1] @ s_l_full_query
    top_L1  = int(sim_L1.argmax().item())

    # Reconstruct embedding: weighted sum of centroids
    w1   = bank.buf_L1_w_full[top_L1]
    r_L1 = (w1[:bank.n_l].unsqueeze(-1) * bank.mu_c_l[:bank.n_l].T).sum(-1)
    r_L1 = r_L1.to(torch.cfloat)

    # L2
    n_l2 = min(int(bank._L1_ptr // 32), bank.K_L2)
    if n_l2 > 0:
        sim_L2 = bank.buf_L2_w_full[:n_l2] @ s_l_full_query
        top_L2 = int(sim_L2.argmax().item())
        w2     = bank.buf_L2_w_full[top_L2]
        r_L2   = (w2[:bank.n_l].unsqueeze(-1) * bank.mu_c_l[:bank.n_l].T).sum(-1).to(torch.cfloat)
    else:
        r_L2 = torch.zeros(bank.mu_c_l.shape[-1], dtype=torch.cfloat)

    # L3
    n_l3 = min(int(bank._L1_ptr // 1024), bank.K_L3)
    if n_l3 > 0:
        sim_L3 = bank.buf_L3_w_full[:n_l3] @ s_l_full_query
        top_L3 = int(sim_L3.argmax().item())
        w3     = bank.buf_L3_w_full[top_L3]
        r_L3   = (w3[:bank.n_l].unsqueeze(-1) * bank.mu_c_l[:bank.n_l].T).sum(-1).to(torch.cfloat)
    else:
        r_L3 = torch.zeros(bank.mu_c_l.shape[-1], dtype=torch.cfloat)

    return r_L1, r_L2, r_L3, bank.buf_L1_ids[top_L1]
```

**Compression gradient fix (§1.51) — ALREADY in v6.0.9 spec:**
```python
# L_compress target must NOT be detached:
L_compress = ((chunk_mean - x_recon_1).conj() * (chunk_mean - x_recon_1)).real.sum()
# NOT: chunk_mean.detach()
# lambda_compress reduced to 0.001 (from 0.01)
```

**`should_spawn` (§1.69):**
```python
def should_spawn(bank, E_min_raw):
    """§1.69: Welford-based spawn threshold (replaces fixed 3.0)."""
    if bank._Emin_n < 10:
        return E_min_raw > 3.0              # fallback before stats stabilise
    sigma = math.sqrt(bank._Emin_var / bank._Emin_n + 1e-8)
    return E_min_raw > bank._Emin_mean + 2.5 * sigma
```

### F.5 model.py / CFLNModel

**`reset_for_inference` — order is critical:**
```python
def reset_for_inference(self):
    # 1. CONSOL-1 BEFORE clearing §1.37
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun, self.cfg)

    # 2. Persist SA §1.38 (optional)
    if self.cfg.get('persist_archive', False):
        self.bank.surprise_archive.save_state(self.cfg['archive_path'])

    # 3. SSP + Lyapunov state §1.39/§1.52
    self.diff_aux.cun._goal_stack  = []
    self.diff_aux.cun._stuck_count = []
    self.diff_aux.cun._v_prev      = []
    self.bank._goal_frozen         = False

    # 4. HYPO state §1.32
    self.bank.g_c.zero_()
    self.bank._in_hypo_mode  = False
    self.bank._r_lista_hypo  = None
    self.bank._u_hypo        = 0.0

    # 5. Precision state §1.58 (per-session reset)
    cun = self.diff_aux.cun
    cun.sigma_sq_buffer   = [1.0, 1.0, 1.0, 1.0, 1.0]
    cun._precision_active = [False, False, False, False, False]
    # log_precision NOT reset (long-term learned parameter)

    # 6. Standard resets (r_lista, etc.)
    self.diff_aux.cun.reset_lista_reservoir()
    # ... existing resets ...
```

**`consolidate_arc_to_cnep` (§1.37, §1.54, §1.67):**
```python
def consolidate_arc_to_cnep(bank, cun, cfg, micro=False):
    """§1.37 (session-end) and §1.54 (per-chunk micro).
    §1.67: consolidation rate is Fisher-scaled.
    """
    tau_consol  = cfg.get('tau_consol', 3.0)
    alpha_base  = cfg.get('alpha_micro', 0.0001) if micro else cfg.get('alpha_consol', 0.001)
    n_r = getattr(cun, '_rule_cache_n', 0)
    if n_r == 0:
        return

    utils = cun.rule_util[:n_r]

    # Micro-consolidation: only top-1 rule
    # Session-end: all rules above threshold
    candidates = [utils.argmax().item()] if micro else range(n_r)

    for idx in candidates:
        if float(utils[idx].item()) < tau_consol:
            continue
        k_rule  = cun.rule_K[idx, :bank.d_c]
        with torch.no_grad():
            dists   = (bank.mu_c_l[:bank.n_l] - k_rule).norm(dim=-1).real
            nearest = int(dists.argmin().item())

            # §1.67: Fisher-scaled learning rate
            fu      = float(bank.fisher_unit[nearest].item())
            alpha_e = alpha_base / (1.0 + fu)

            # SI proxy: mature units get smaller update
            freq    = float(bank.activation_freq_l[nearest].item())
            si_gate = max(0.0, 1.0 - freq / cfg.get('alpha_young', 0.1))

            delta   = alpha_e * si_gate * (k_rule - bank.mu_c_l[nearest])
            bank.mu_c_l.data[nearest] += delta
```

### F.6 train_step.py

**Complete ordering (§1.60):**
```python
def train_step(batch, model, opts, si, fisher, cfg, step):
    # Forward
    logits, info = model(batch)
    loss = cross_entropy(logits, batch)

    # MDLM Stage 0 §1.44 — mask_embed IS required (spec §1.44 is authoritative)
    # bank.mask_embed = nn.Parameter(torch.zeros(d_c, dtype=torch.cfloat))  ← must exist
    # The gap analysis verdict "keep deleted" was WRONG — not in spec's Removed Parameters.
    if cfg.get('stage') == 'stage0' and cfg.get('p_mask', 0) > 0:
        mask_pos = torch.bernoulli(cfg['p_mask'] * torch.ones(batch.shape)).bool()
        x_c_in   = model.embed(batch)
        x_c_in[mask_pos] = bank.mask_embed.expand_as(x_c_in[mask_pos])
        logits_m, *_ = model.forward_from_embed(x_c_in)
        L_mlm = F.cross_entropy(
            logits_m[mask_pos[..., :-1]].reshape(-1, cfg['vocab_size']),
            batch[mask_pos[..., 1:]].reshape(-1))
        loss += cfg.get('lambda_mlm', 0.3) * L_mlm

    # CL protection
    beta_kl  = min(cfg['beta_KL'], cfg['beta_KL'] * step / max(cfg['beta_KL_warmup'], 1))
    loss += beta_kl          * compute_L_KL(model, fisher, cfg)
    loss += cfg['beta_SI_stiefel'] * compute_L_SI_stiefel(model, si, cfg)

    # Auxiliary losses from forward pass
    if info.get('L_bridge'):    loss += cfg['lambda_bridge']   * info['L_bridge']
    if info.get('L_vq'):        loss += cfg['lambda_vq']       * info['L_vq']
    if info.get('L_diversity'): loss += cfg['lambda_diversity'] * info['L_diversity']

    # Regularisers
    loss += _L_lipschitz(model.bank, cfg)   # §1.56
    loss += _L_sigma_reg(model.bank, cfg)   # §1.56
    loss += _L_precision(model, cfg)        # §1.58

    # BACKWARD
    loss.backward()

    # Fisher accumulation — BEFORE clip_grad_norm_ §1.57
    accumulate_fisher(model, stiefel_ids, model.bank, fisher)

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

    # Optimizers
    for opt in [opts['opt_g'], opts['opt_u'], opts['muon']]:
        opt.step(); opt.zero_grad()

    # Stiefel retraction
    stiefel_update_all_v51(model)

    # Fisher-magnitude freeze check — every 100 steps §1.63
    if step % 100 == 0:
        update_fisher_magnitude_freeze(model.bank, fisher)

    # Micro-consolidation — per-chunk §1.54
    # Called from the chunk-boundary handler (CFL5Layer or model.py):
    #   L_vq = vq_telescope_update(...)
    #   consolidate_arc_to_cnep(bank, cun, cfg, micro=True)  # ← here

    # §1.56 ROB-L: log_alp_l regulariser (young units only)
    # §1.56 ROB-S: log_sigma_bind regulariser
```

**`_L_lipschitz` and `_L_sigma_reg` (§1.56):**
```python
def _L_lipschitz(bank, cfg):
    """§1.56 ROB-L: penalise sharp routing for young units."""
    n = bank.n_l
    if n == 0:
        return torch.tensor(0.0)
    young_mask = ~bank.is_sensory_l[:n]
    if not young_mask.any():
        return torch.tensor(0.0)
    L = bank.log_alp_l[:n][young_mask].mean()
    return cfg.get('lambda_lipschitz', 0.001) * L

def _L_sigma_reg(bank, cfg):
    """§1.56 ROB-S: encourage wider phase kernel."""
    L = torch.exp(-bank.log_sigma_bind)   # 1/sigma
    return cfg.get('lambda_sigma_reg', 0.001) * L
```

**`accumulate_fisher` (§1.57 with alpha_freeze fix):**
```python
def accumulate_fisher(model, stiefel_ids, bank, fisher):
    """§1.57: Fisher EMA for AdamW params. BEFORE clip_grad_norm_.
    Rule 3: uses bank.is_sensory_l (per-unit bool Tensor).
            bank.alpha_freeze is a scalar float — NOT subscriptable.
    """
    n_l = bank.n_l
    for name, param in model.named_parameters():
        if id(param) in stiefel_ids:    continue   # SI handles Stiefel
        if param.grad is None:           continue   # Rule 2

        # Rule 3: skip frozen units
        unit_idx = _get_unit_idx(name, n_l)
        if unit_idx is not None and bank.is_sensory_l[unit_idx]:
            continue

        if name not in fisher:
            fisher[name] = torch.zeros_like(param.data)
        fisher[name].mul_(0.99).add_(0.01 * param.grad.detach()**2)


def _get_unit_idx(param_name, n_l):
    """Extract unit index from parameter name, or None if not unit-indexed."""
    # W_l[i] is unit-specific; other params (log_lam_bind, etc.) are global
    # Implement based on your naming convention, e.g.:
    # 'bank.W_l' → shape (N_max_l, d_e_l, d_c) → not unit-specific at param level
    # The fisher dict key for W_l covers all units in one tensor
    # → unit_idx gating is applied per-slice in update_fisher_magnitude_freeze
    # For non-sliced parameters, return None (no per-unit gate needed)
    return None  # most params are not unit-specific at the nn.Parameter level
```

**`update_fisher_magnitude_freeze` (§1.63 C1):**
```python
def update_fisher_magnitude_freeze(bank, fisher, k_sigma=1.5):
    """§1.63: Fisher-magnitude parallel freeze trigger. Every 100 steps.
    Parallel to histogram-based freeze in update_sensory_mask().
    Both write to bank.is_sensory_l. bank.alpha_freeze (scalar) unchanged.
    """
    n = bank.n_l
    if n == 0:
        return

    # Find W_l fisher entry (shape N_max_l × d_e_l × d_c)
    wl_key = next((k for k in fisher if 'W_l' in k and ('bank' in k or k == 'W_l')), None)
    if wl_key is None:
        return

    fwl = fisher[wl_key]
    if fwl.shape[0] < n:
        return

    with torch.no_grad():
        # Per-unit mean Fisher
        unit_fisher = fwl[:n].abs().mean(dim=list(range(1, fwl.dim())))  # (n,)
        bank.fisher_unit[:n] = unit_fisher

        if unit_fisher.std() < 1e-8:
            return   # early training — all same, nothing to freeze

        mu        = unit_fisher.mean()
        sigma     = unit_fisher.std()
        threshold = mu + k_sigma * sigma

        new_frozen = (unit_fisher > threshold) & ~bank.is_sensory_l[:n]
        if new_frozen.any():
            bank.is_sensory_l[:n]          |= new_frozen
            bank.sensory_domain_id[:n][new_frozen] = -1
```

---

## PART G — Q_BEAM COMPOSITE (§1.46, §1.53)

```python
def compute_Q_beam(h_N, r_lista, r_goal_proxy, goal_stack, x_c,
                   E_min_raw=None, H_route_raw=None,
                   phi_rel=None, log_w_beam=None):
    """Multi-field beam quality. No MLP. Parameter-free core (3 optional scalars).
    
    Signals:
      F3: MDL sparsity  (-||h_N||₁)
      F4: Lyapunov goal (-||r_lista - r_goal_proxy||²)
      F5: CSP arc       (min cosine sim to SSP stack entries)
      D1: phi_rel       (relational context richness)
      F1: thermodynamic (-(E_min_raw × H_route_raw))  [optional]
    """
    signals = []

    # F3: MDL (always present)
    signals.append(-h_N.abs().sum())

    # F4: Lyapunov (when goal proxy available)
    if r_goal_proxy is not None:
        signals.append(-(r_lista - r_goal_proxy).norm()**2)

    # F5: CSP arc-consistency (when stack non-empty) — §1.49
    if goal_stack:
        sims = [torch.nn.functional.cosine_similarity(
                    r_lista.real.unsqueeze(0), s.real.unsqueeze(0)).item()
                for s in goal_stack]
        signals.append(min(sims))

    # D1: relational richness
    if phi_rel is not None:
        signals.append(float(phi_rel.norm().item()))

    # F1: thermodynamic (optional)
    if E_min_raw is not None and H_route_raw is not None:
        signals.append(-(E_min_raw * H_route_raw))

    if not signals:
        return torch.tensor(0.0)

    signals_t = torch.stack([s if isinstance(s, torch.Tensor)
                              else torch.tensor(float(s)) for s in signals])

    if log_w_beam is not None and len(log_w_beam) >= len(signals):
        w = torch.softmax(log_w_beam[:len(signals)], dim=0)
        return (w * signals_t).sum()
    return signals_t.mean()
```

---

## PART H — PRECISION UPDATE (§1.58, replaces MC-2)

```python
def update_precision(cun, signals, is_hypo_active=False):
    """§1.58: Self-calibrating precision. Replaces MC-2 _log_w_rec EMA.
    signals = [U_repr_q, U_epi_cal, U_hopfield, U_temporal, U_hypo]
    
    MUST: sigma_sq_buffer initialised to [1.0]*5 (not 0.0).
    """
    import math
    for s, sig in enumerate(signals):
        val = float(sig) if not isinstance(sig, torch.Tensor) else sig.item()

        # Pathway activity gates
        if s == 4 and not is_hypo_active:
            continue   # U_hypo: skip if HYPO never activated this session

        if abs(val) > 1e-6:
            cun._precision_active[s] = True
        if not cun._precision_active[s]:
            continue   # hold sigma_sq=1.0 until pathway activates

        cun.sigma_sq_buffer[s] = 0.95 * cun.sigma_sq_buffer[s] + 0.05 * val**2
        lp = -0.5 * math.log(cun.sigma_sq_buffer[s] + 1e-6)
        cun.log_precision[s] = max(-3.0, min(3.0, lp))

    prec      = torch.exp(torch.tensor(cun.log_precision))
    signals_t = torch.tensor([float(s) if not isinstance(s, torch.Tensor)
                               else s.item() for s in signals])
    return (prec * signals_t).sum() / (prec.sum() + 1e-8)
```

---

## PART I — DCG+ ADAPTIVE COMMIT (§1.74 C12)

```python
# In generate_cfln_dcg_plus(), replace:
#   if commit_score > 0.4:
# With:
u_epi_cal = float(getattr(bank, '_last_u_epi_cal', 0.5))
commit_threshold = max(0.1, 1.0 - u_epi_cal)
if commit_score > commit_threshold:
    # commit the token
```

---

## PART J — RULE UTIL ADAPTIVE DECAY (§1.75 D5)

```python
# In CUN.lista_forward, rule utility decay step, replace:
#   rule_util[k] *= 0.999999
# With:
u_temp = float(getattr(self, '_last_u_temporal', 0.0))
decay_k = 0.999999 * (1.0 - 0.0001 * u_temp)
rule_util[k] = rule_util[k] * decay_k
```

---

## PART K — VERIFICATION CHECKLIST

Run these checks before considering an implementation complete.

### K.1 Type checks (will crash at runtime if wrong)

```python
assert isinstance(bank.alpha_freeze, float), "alpha_freeze must be scalar float"
assert isinstance(bank.is_sensory_l, torch.Tensor), "is_sensory_l must be Tensor"
assert bank.is_sensory_l.dtype == torch.bool
assert isinstance(bank.W_rc_bridge, torch.nn.Parameter), "W_rc_bridge must be Parameter"
assert not isinstance(getattr(bank, 'W_rc_bridge_old', None), torch.Tensor) \
       or bank.W_rc_bridge.requires_grad, "W_rc_bridge must require grad"
```

### K.2 Initialisation checks

```python
cun = model.diff_aux.cun
assert all(abs(v - 1.0) < 1e-6 for v in cun.sigma_sq_buffer), \
    "sigma_sq_buffer must init to [1.0]*5"
assert abs(float(cun.log_blend_alpha.item()) - (-0.223)) < 0.01, \
    "log_blend_alpha must init to log(0.8)"
assert abs(float(bank.log_cal_scale.item()) - (-1.897)) < 0.01, \
    "log_cal_scale must init to log(0.15)"
```

### K.3 Optimizer exclusion checks

```python
param_ids_in_optimizers = set()
for opt in [opts['opt_g'], opts['opt_u'], opts['muon']]:
    for g in opt.param_groups:
        for p in g['params']:
            param_ids_in_optimizers.add(id(p))

assert id(cun.U1) not in param_ids_in_optimizers, "U1 must NOT be in any optimizer"
assert id(cun.U2) not in param_ids_in_optimizers, "U2 must NOT be in any optimizer"
assert id(bank.W_rc_bridge) in param_ids_in_optimizers, "W_rc_bridge must be in optimizer"
```

### K.4 Gradient flow checks

```python
# L_vq must give gradient to chunk_mean, not mu_c_l
chunk_mean = torch.randn(d_c, dtype=torch.cfloat, requires_grad=True)
sel_l_test = torch.tensor([0, 1, 2])
bank.mu_c_l.data[:3] = torch.randn(3, d_c, dtype=torch.cfloat)
z_approx = bank.mu_c_l[sel_l_test].mean(0).detach()
L_vq = (chunk_mean - z_approx).norm()**2
L_vq.backward()
assert chunk_mean.grad is not None and chunk_mean.grad.norm() > 0
if bank.mu_c_l.grad is not None:
    assert bank.mu_c_l.grad.norm() < 1e-6, "mu_c_l must NOT get gradient from L_vq"

# L_bridge must give gradient to W_rc_bridge
rho = torch.randn(d_r_node, dtype=torch.cfloat)
r_seed = bank.W_rc_bridge @ rho
r_seed_target = (cun.U1.conj() @ x_c_test.mean(0))[:r_seed.shape[0]].detach()
L_bridge = (r_seed_target - r_seed).norm()**2
L_bridge.backward()
assert bank.W_rc_bridge.grad is not None and bank.W_rc_bridge.grad.norm() > 0
```

### K.5 Fisher accumulation order

```python
# In your train_step, verify the call order by inspection:
# 1. loss.backward()         ← gradients computed
# 2. accumulate_fisher(...)  ← Fisher uses UNCLIPPED gradients
# 3. clip_grad_norm_(...)    ← THEN clip
# 4. optimizer.step()
```

### K.6 Welford E_min update

```python
# After running 20+ chunks, Welford stats must be non-trivial:
assert bank._Emin_n >= 20
assert bank._Emin_mean > 0, "Welford mean must accumulate"
assert bank._Emin_var >= 0, "Welford variance must be non-negative"
# Should_spawn must use statistics, not fixed 3.0:
sigma = (bank._Emin_var / bank._Emin_n + 1e-8)**0.5
spawn_thr = bank._Emin_mean + 2.5 * sigma
assert spawn_thr != 3.0 or bank._Emin_n < 10, "spawn threshold must be Welford-based"
```

### K.7 Reset ordering

```python
# Test that CONSOL-1 fires BEFORE r_lista reset:
import unittest.mock as mock
with mock.patch('cfln.bank.consolidate_arc_to_cnep') as mock_consol:
    model.reset_for_inference()
    assert mock_consol.called, "consolidate_arc_to_cnep must be called in reset"
    # And r_lista must be reset after (check cun.r_lista is zero after reset)
```

---

## PART L — ALL KNOWN BUGS AND THEIR FIXES

| # | Bug | Location | Fix |
|---|---|---|---|
| 1 | `alpha_freeze[i]` → TypeError | §1.57, §1.62, §1.63 | Use `is_sensory_l[i]` everywhere |
| 2 | `mask_embed` carry-forward | AI Instructions §4 | Deleted in v9.0; do not add it |
| 3 | Fisher skip uses wrong mask | §1.57 Rule 3 | `bank.is_sensory_l[unit_idx]` |
| 4 | `sigma_sq_buffer = [0.0]*5` | §1.58 init | Must be `[1.0]*5` |
| 5 | `chunk_mean.detach()` in L_compress | §1.51 | Remove `.detach()` from target |
| 6 | `L_vq` gives gradient to `mu_c_l` | §1.59 | Add `.detach()` to codebook side |
| 7 | Fisher after clip_grad_norm_ | §1.57 | Move `accumulate_fisher` before clip |
| 8 | Welford never updated | §1.68/1.69 | Add 4 lines to `vq_telescope_update` |
| 9 | `update_fisher_magnitude_freeze` missing | §1.63 | Implement + call every 100 steps |
| 10 | `N_rules` vs `episodic_rule_cache_n` | config | Use fallback: `get('episodic_rule_cache_n', get('N_rules', 256))` |
| 11 | `U_hypo` precision decay when inactive | §1.58 | Skip sigma_sq update when `not is_hypo_active` |
| 12 | Lyapunov fires before `g_c` set | §1.52 | Guard: `r_lista_goal.norm() > 1e-4` |

---

## PART M — INVARIANTS (never violate)

1. `bank.alpha_freeze` = scalar float · `bank.is_sensory_l` = per-unit bool Tensor
2. `cun.U1`, `cun.U2` not in any optimizer (Haar-random fixed basis)
3. `bank.W_rc_bridge` = nn.Parameter (not register_buffer since v9.0 §1.50)
4. `sigma_sq_buffer` initialised to `[1.0]*5` every session reset
5. `B_bind`, `B_role`, `B_comp` all PSD → `apply_psd` on `W_full` always valid
6. `W_full[:k_l,:k_l] +=` all three terms: lam_bind×B_bind + lam_role×B_role + lam_comp×B_comp
7. `role_vecs` in `stiefel_ids` exclusion → goes to AdamW, not Muon
8. `tau_smooth.clamp(min=1e-3)` in STELA — prevents sign-flip
9. Fisher accumulation: after `loss.backward()`, before `clip_grad_norm_()`
10. `L_vq`: codebook detached, encoder not detached
11. `L_compress`: `chunk_mean` NOT detached (§1.51 fix)
12. `consolidate_arc_to_cnep` called BEFORE session state reset
13. VQ buf stores full N_max_l weight vectors (not k_l index arrays)
14. `_Emin_n < 10` fallback in `should_spawn` (Welford unstable early)
15. `merge_weight = sigmoid(_last_Q_BEAM_score)` on SSP POP (not fixed 0.3)
