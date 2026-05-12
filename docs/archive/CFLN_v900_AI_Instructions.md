# CFLN v9.0 — AI Implementation Instructions
## Complete guide for implementing the v9.0 change spec on top of v6.0.9

---

## 0. CONTEXT AND FILES

You are implementing changes to the CFLN (Complex Field Latent Network) architecture.
You have two source documents:

| File | Role |
|---|---|
| `CFLN_v609_Master_Spec.md` | **Base implementation** — 5,758 lines, fully runnable Python. Your ground truth. Do not modify its logic unless a change section explicitly says to. |
| `CFLN_v900_Spec.md` | **Change spec** — 1,888 lines. Contains §1.30–§1.77. Every new feature, removal, and fix to apply on top of v6.0.9. |

**Do not use any other spec files.** All earlier versions (v7.0, v8.0, v8.0 Addendum) are superseded and fully absorbed into v9.0.

---

## 1. CRITICAL TYPE FACTS — READ BEFORE TOUCHING ANY CODE

These are the most common sources of bugs. Verify before writing any line.

### 1.1 alpha_freeze is a SCALAR FLOAT, not a Tensor

```python
bank.alpha_freeze          # float, e.g. 0.7 — the histogram percentile threshold
bank.alpha_freeze[i]       # ← TypeError: float is not subscriptable — DO NOT DO THIS
```

The per-unit frozen signal is:
```python
bank.is_sensory_l[i]       # bool Tensor (N_max_l,) — True when unit i is frozen
```

**Whenever the spec says "alpha_freeze[i]" or "frozen unit" check**, use `bank.is_sensory_l[i]`.
This includes §1.57 (Fisher skip), §1.62 (test), §1.63 (C1 Fisher-magnitude trigger).

### 1.2 r_lista is always detached

```python
self.r_lista = (...)       .detach()    # always — BPTT through r_lista is disabled by design
r_seed        = W_bridge @ rho_weighted # r_seed IS differentiable (before detach)
```

The L_bridge loss (§1.50) provides gradient to W_bridge through r_seed **before** detach.

### 1.3 U1 and U2 are NEVER TRAINED

```python
stiefel_ids = {id(W_l), id(W_p), id(U1), id(U2)}  # U1/U2 excluded from ALL optimizers
```

They are Haar-random unitary matrices, fixed at init. Do not add them to any optimizer.

### 1.4 VQ codes use full routing weight vectors, not index arrays

```python
# WRONG: store sel_l indices and compute Jaccard
# CORRECT: store full n_l-dimensional weight vector
bank.buf_L1_w_full[ptr] = s_l_full.detach()   # (N_max_l,) float32, sparse
```

Where `s_l_full` is constructed as:
```python
s_l_full = torch.zeros(bank.N_max_l)
s_l_full[sel_l] = s_l[:k_l_eff]               # scatter routing weights
```

### 1.5 sigma_sq_buffer must init to 1.0, not 0.0

```python
self.sigma_sq_buffer = [1.0, 1.0, 1.0, 1.0, 1.0]   # NOT [0.0, ...]
```

If zero: `-0.5 * log(1e-6) = 6.9`, clamped to 3.0, giving 20× weight on first token.

### 1.6 L_vq must detach the codebook, not the encoder

```python
z_approx = bank.mu_c_l[sel_l].mean(0).detach()  # codebook detached (trained by CNEP)
L_vq = (chunk_mean - z_approx).norm()**2          # gradient flows to chunk_mean (encoder)
```

Never write `chunk_mean.detach()` in L_vq — that was the old bug fixed in §1.51.

### 1.7 Fisher accumulation ordering in train_step

```
loss.backward()
→ accumulate_fisher()      ← BEFORE clip_grad_norm_ (unclipped gradients = true curvature)
→ clip_grad_norm_()
→ optimizer.step()
```

Fisher after clipping systematically underestimates importance of high-gradient parameters.

---

## 2. IMPLEMENTATION ORDER

Apply changes in this exact order. Each group is internally consistent.
Do not skip to a later group before completing an earlier one.

### GROUP 1 — Infrastructure (no dependencies)
| Section | What | Files |
|---|---|---|
| §1.36 | buf_L1_ids verbatim spans | bank.py |
| §1.38 | persist_archive flag | archive.py |
| §1.43 | k-shot centroid buffers (_proto_count, _proto_sum) | bank.py |
| §1.44 | mask_embed parameter | model.py |
| §1.50 | W_rc_bridge buffer→nn.Parameter | model.py |
| §1.57 | fisher dict buffer + accumulate_fisher() | train_step.py, bank.py |
| §1.68/69 | Welford E_min buffers (_Emin_mean, _Emin_var, _Emin_n) | bank.py |
| §1.72 | log_cal_scale parameter | bank.py |
| §1.73 | log_blend_alpha parameter | cun.py |
| §1.77 | Config key changes | cfg.py |

### GROUP 2 — Binding (depends on Group 1)
| Section | What | Files |
|---|---|---|
| §1.30/1.41 | B_bind phase kernel + log_sigma_bind | cfl5layer.py |
| §1.35 | B_role RAH + role_vecs (QR ortho init) | cfl5layer.py, bank.py |
| §1.55 | B_comp = B_bind * B_role (Hadamard, §1.77 log_lam_composition) | cfl5layer.py |
| §1.33 | Goal-anchored context g_c | bank.py, cfl5layer.py |
| §1.64 | Precision-adaptive k_l_eff | cfl5layer.py |

### GROUP 3 — Reasoning (depends on Groups 1-2)
| Section | What | Files |
|---|---|---|
| §1.31 | Two-key ARC (phi_rel, log_alpha_arc) | cun.py |
| §1.32 | HYPO mode (r_lista_hypo, HYPO_START/END tokens) | model.py, cun.py |
| §1.34 | U_meta_v4 five-signal (add U_hypo, extend _log_w_rec→5) | cun.py |
| §1.40 | STELA smooth threshold (tau_smooth, clamp min=1e-3) | cun.py |
| §1.45 | Reservoir-augmented LISTA reconstruction | cun.py |
| §1.58 | Precision-weighted U_meta (log_precision, sigma_sq_buffer) | cun.py |

### GROUP 4 — Memory (depends on Groups 1-2)
| Section | What | Files |
|---|---|---|
| §1.59 | VQ-Telescope (remove W_compress, add buf_L1_w_full) | bank.py, telescoping.py |
| §1.51 | Compression gradient fix (remove .detach() from chunk_mean in L_compress) | telescoping.py |
| §1.42 | L_recon → L_vq transition | telescoping.py, train_step.py |

### GROUP 5 — Planning/Reasoning Stack (depends on Groups 1-4)
| Section | What | Files |
|---|---|---|
| §1.39 | SSP goal stack PUSH/POP D=4 (add 6 vocab tokens) | model.py, cun.py, tokenizer.py |
| §1.47 | r_lista beam B_eff (eps_beam_scale, log_w_beam) | cun.py |
| §1.46 | Q_BEAM composite F1–F5+D1 (compute_Q_beam function) | cun.py |
| §1.48 | Lyapunov goal proxy (r_lista_goal_proxy, TS-3) | cfl5layer.py, cun.py |
| §1.49 | CSP arc-consistency via SSP stack (F5) | cun.py |
| §1.52 | Lyapunov timeout auto-POP (_stuck_count removed, use think-budget §1.71) | cun.py, model.py |
| §1.53 | phi_rel richness D1 in Q_BEAM | cun.py |
| §1.65 | Q_BEAM-weighted SSP merge (_last_Q_BEAM_score) | cun.py |
| §1.66 | U_meta-adaptive beam B_eff | cun.py |
| §1.71 | Remove N_stuck config (think-budget is timeout) | cun.py, cfg.py |

### GROUP 6 — CL / Emergence (depends on all groups)
| Section | What | Files |
|---|---|---|
| §1.37 | CONSOL-1 (consolidate_arc_to_cnep, before reset) | bank.py, model.py |
| §1.54 | Micro-consolidation per-chunk | telescoping.py |
| §1.57 | Fisher-KL (accumulate_fisher, L_KL, ordering) | train_step.py |
| §1.63 | Fisher-magnitude alpha_freeze (fisher_unit buffer → is_sensory_l) | bank.py, train_step.py |
| §1.67 | Fisher-scaled consolidation rates | bank.py |

### GROUP 7 — Training objectives (depends on all groups)
| Section | What | Files |
|---|---|---|
| §1.44 | MDLM masked token training (Stage 0, L_mlm) | train_step.py |
| §1.50 | L_bridge (W_bridge predictive coding loss) | cfl5layer.py, train_step.py |
| §1.56 | ROB-L/S regularisers (L_lipschitz, L_sigma_reg) | train_step.py |
| §1.60 | Full train_step ordering | train_step.py |

### GROUP 8 — Emergent parameter replacements (depends on all groups)
| Section | What | Old constant removed |
|---|---|---|
| §1.63 | Fisher-magnitude freeze (parallel trigger) | alpha_freeze 85th percentile only |
| §1.64 | k_l_eff = f(U_epi_cal) | k_l=40 fixed |
| §1.65 | merge_weight = sigmoid(Q_BEAM) | ssp_merge_alpha=0.7 |
| §1.66 | B_eff = f(U_meta) | B=2 fixed |
| §1.67 | alpha_eff = alpha_base/(1+fisher_unit) | alpha_consol/micro fixed |
| §1.68 | Welford E_min surprise | surprise_threshold=0.5 |
| §1.69 | Welford spawn threshold | spawn_threshold=3.0 |
| §1.70 | U_epi-gated k-shot | tau_proto=0.6 |
| §1.71 | Think-budget timeout | N_stuck=12 removed |
| §1.72 | log_cal_scale in MC-1 | 0.15 hardcoded |
| §1.73 | log_blend_alpha in r_lista | 0.8 hardcoded |
| §1.74 | U_epi-adaptive DCG+ commit | commit_threshold=0.4 |
| §1.75 | U_temporal rule_util decay | 0.999999 fixed |
| D3    | N_rules cfg change | 64 → 256 |

---

## 3. NEW VOCABULARY TOKENS (6 total)

Add to tokenizer before any training:

```python
SPECIAL_TOKENS = {
    'think_start':  '<think>',
    'think_end':    '</think>',
    'hypo_start':   '<hypo>',
    'hypo_end':     '</hypo>',
    'push_goal':    '<push_goal>',
    'pop_goal':     '</push_goal>',   # closing tag = POP signal
}
# vocab_size_extended = vocab_size + 6
```

---

## 4. NEW PARAMETERS (complete list)

Add these in `__init__` of the indicated class. All use `opt_g` (AdamW) unless noted.

### CFBank additions
```python
# Binding
self.log_lam_bind        = nn.Parameter(torch.tensor(-3.0))
self.log_sigma_bind      = nn.Parameter(torch.tensor(0.693))   # init log(2.0)
self.log_lam_role        = nn.Parameter(torch.tensor(-3.0))
self.log_lam_composition = nn.Parameter(torch.tensor(-3.0))    # Hadamard B_comp
self.log_lam_goal        = nn.Parameter(torch.tensor(-3.0))
self.W_goal_detect       = nn.Parameter(torch.zeros(1, d_c))

# Role vectors: QR ortho init, excluded from Muon (stiefel_ids.add)
_raw = torch.randn(R_roles, d_c, dtype=torch.cfloat)
_Q, _ = torch.linalg.qr(_raw.T)
self.role_vecs = nn.Parameter(_Q.T[:R_roles].contiguous())

# Metacognition calibration
self.log_cal_scale       = nn.Parameter(torch.tensor(-1.897))  # init log(0.15)

# W_bridge: was register_buffer, now nn.Parameter (trained via L_bridge §1.50)
self.W_rc_bridge         = nn.Parameter(W_bridge_init)          # NOT register_buffer

# k-shot refinement buffers (NOT parameters)
self.register_buffer('_proto_count', torch.zeros(N_max_l, dtype=torch.int32))
self.register_buffer('_proto_sum',   torch.zeros(N_max_l, d_c, dtype=torch.cfloat))

# VQ-Telescope buffers
self.register_buffer('buf_L1_w_full', torch.zeros(K_L1, N_max_l))
self.register_buffer('buf_L2_w_full', torch.zeros(K_L2, N_max_l))
self.register_buffer('buf_L3_w_full', torch.zeros(K_L3, N_max_l))
self.register_buffer('buf_L1_ids',    torch.zeros(K_L1, C_chunk, dtype=torch.int32))

# Welford E_min statistics
self._Emin_mean = 0.0
self._Emin_var  = 0.0
self._Emin_n    = 0

# Fisher per-unit mean (NOT nn.Parameter — pure buffer)
self.register_buffer('fisher_unit', torch.zeros(N_max_l))

# g_c goal register
self.register_buffer('g_c', torch.zeros(d_c, dtype=torch.cfloat))

# MDLM mask embedding (nn.Parameter so it gets gradient)
self.mask_embed = nn.Parameter(torch.zeros(d_c, dtype=torch.cfloat))
```

### CFLNModel additions
```python
# W_bridge moved here from CFBank in some implementations
# Ensure it's nn.Parameter NOT register_buffer
```

### CUN additions
```python
# ARC dual-key
self.log_alpha_arc       = nn.Parameter(torch.tensor(0.0))
self.register_buffer('rule_K', torch.zeros(N_rules, 2*d_c, dtype=torch.cfloat))

# U_meta_v4
self.log_w_meta          = nn.Parameter(torch.tensor([1.0, -1.0, -1.0, -2.0, -2.0]))
self._log_w_rec          = [0.0] * 5    # extended from 4 to 5

# Precision-weighted metacognition (REPLACES log_w_meta weighting role)
# log_precision is a plain list, NOT nn.Parameter (self-calibrates from data)
self.log_precision       = [0.0] * 5
self.sigma_sq_buffer     = [1.0, 1.0, 1.0, 1.0, 1.0]   # MUST init 1.0 not 0.0
self._precision_active   = [False] * 5

# STELA
self.tau_smooth          = nn.Parameter(torch.tensor(0.1))

# Beam search
self.eps_beam_scale      = nn.Parameter(torch.tensor(0.1))
self.log_w_beam          = nn.Parameter(torch.zeros(3))

# Reasoning state blend (replaces hardcoded 0.8)
self.log_blend_alpha     = nn.Parameter(torch.tensor(-0.223))  # init log(0.8)

# SSP state (reset per session)
self._goal_stack         = []
self._stuck_count        = []   # used only during PUSH_GOAL (Lyapunov monitoring)
self._v_prev             = []

# Q_BEAM state
self._last_Q_BEAM_score  = 0.0
self._phi_rel_cache      = None
self._phi_rel_step       = 0

# HYPO state
self._in_hypo_mode       = False
self._r_lista_hypo       = None

# Fisher verifier (removed W_verif_1/2 — no MLP, use Q_BEAM)
```

---

## 5. REMOVED PARAMETERS (delete from __init__ and all uses)

```python
# FROM TelescopingMemory / CFBank:
# W_compress_L1, W_compress_L2, W_compress_L3   ← removed, VQ replaces
# W_decompress_L1                                ← removed, centroid mean replaces

# FROM CUN:
# log_w_meta            ← replaced by log_precision (self-calibrating)
# _log_w_rec            ← replaced by sigma_sq_buffer (same role, different update)
```

**Remove from train_step:**
- `L_recon` term → replaced by `L_vq`
- `L_compress` on W_compress → VQ-Telescope handles compression

---

## 6. CONFIG KEY CHANGES

```python
# REMOVE these keys (now emergent — not heuristic):
# 'surprise_threshold'       → §1.68 Welford E_min replaces it
# 'ssp_stuck_threshold'      → §1.71 think-budget is the natural timeout
# 'ssp_merge_alpha'          → §1.65 Q_BEAM quality-weighted merge
# 'spawn_threshold'          → §1.69 Welford E_min replaces it
# 'tau_proto'                → §1.70 U_epi-gated (tau_proto_min=0.4 instead)

# ADD:
'k_l_min':           10,     # §1.64 adaptive k_l
'k_l_max':           40,     # §1.64 (was hard-coded k_l=40)
'beam_B_max':        3,      # §1.66 adaptive beam
'tau_proto_min':     0.4,    # §1.70 (softer, gated by U_epi)
'beta_KL':           0.5,    # §1.57 Fisher-KL weight (AdamW params)
'beta_SI_stiefel':   0.25,   # §1.57 SI weight (Stiefel params)
'beta_KL_warmup':    500,    # §1.57 anneal steps
'lambda_prec':       0.001,  # §1.58 precision entropy regulariser
'lambda_vq':         0.01,   # §1.59 VQ encoder commitment weight
'lambda_bridge':     0.1,    # §1.50 W_bridge L_bridge weight
'lambda_mlm':        0.3,    # §1.44 MDLM masked token weight
'lambda_diversity':  0.01,   # §1.47 beam anti-collapse
'lambda_lipschitz':  0.001,  # §1.56 routing sharpness
'lambda_sigma_reg':  0.001,  # §1.56 phase kernel width
'lambda_vq':         0.01,   # §1.59 VQ consistency
'p_mask':            0.15,   # §1.44 MDLM masking rate (Stage 0 only)
'alpha_micro':       0.0001, # §1.54 micro-consolidation rate
'K_proto_max':       10,     # §1.43 k-shot max exposures
'alpha_young':       0.1,    # §1.43 young unit threshold

# CHANGE:
'episodic_rule_cache_n': 256,  # was 64 (§1.76 D3)
```

---

## 7. OPTIMIZER GROUPS — WHAT GOES WHERE

```python
stiefel_ids = {
    id(bank.W_l),             # Muon (Cayley retraction)
    id(bank.W_p),             # Muon (Cayley retraction)
    id(cun.U1),               # NEVER trained (fixed unitary basis)
    id(cun.U2),               # NEVER trained (fixed unitary basis)
    id(bank.role_vecs),       # AdamW (not Stiefel — shape R×d_c, not square)
    id(model.W_rc_bridge),    # AdamW (ESN conservative — keep out of Muon)
}
# Everything NOT in stiefel_ids → AdamW (opt_g)
# fisher dict → NOT in any optimizer (accumulation buffer, not a parameter)
# sigma_sq_buffer → NOT in any optimizer (plain Python list)
# log_precision → NOT in any optimizer (self-calibrates, not gradient-updated)
```

---

## 8. TRAIN STEP ORDERING (§1.60)

```python
def train_step(batch, model, opts, si, fisher, cfg, step):
    # 1. Forward
    logits, info = model(batch)

    # 2. Base loss
    loss = cross_entropy(logits, batch)

    # 3. MDLM masking (Stage 0 only)
    if cfg.get('stage') == 'stage0' and cfg.get('p_mask', 0) > 0:
        loss += cfg['lambda_mlm'] * compute_L_mlm(batch, model, cfg)

    # 4. CL protection losses
    loss += cfg['beta_KL']        * compute_L_KL(model, fisher, cfg)
    loss += cfg['beta_SI_stiefel']* compute_L_SI_stiefel(model, si, cfg)

    # 5. Auxiliary losses from forward info
    if info.get('L_bridge'):   loss += cfg['lambda_bridge']   * info['L_bridge']
    if info.get('L_vq'):       loss += cfg['lambda_vq']       * info['L_vq']
    if info.get('L_diversity'):loss += cfg['lambda_diversity']* info['L_diversity']

    # 6. Regularisers
    loss += compute_L_lipschitz(model.bank, cfg)   # §1.56 ROB-L
    loss += compute_L_sigma_reg(model.bank, cfg)   # §1.56 ROB-S
    loss += compute_L_precision(model, cfg)        # §1.58 precision entropy

    # 7. BACKWARD
    loss.backward()

    # 8. Fisher accumulation — BEFORE clip_grad_norm_ (critical ordering)
    accumulate_fisher(model, stiefel_ids, model.bank, fisher)

    # 9. Gradient clipping
    clip_grad_norm_(all_params, max_norm=1.0)

    # 10. Optimizer steps
    for opt in [opts['opt_g'], opts['opt_u'], opts['muon']]:
        opt.step()
        opt.zero_grad()

    # 11. Stiefel retraction (Muon/Cayley for W_l, W_p)
    stiefel_update_all_v51(model)

    # 12. Fisher-magnitude freeze check (every 100 steps)
    if step % 100 == 0:
        update_fisher_magnitude_freeze(model.bank, fisher)
```

---

## 9. RESET_FOR_INFERENCE ORDERING (§1.39, §1.52, §1.58)

```python
def reset_for_inference(self):
    # 1. Consolidate ARC → μ_c_l BEFORE clearing (§1.37)
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun, self.cfg)

    # 2. Persist SurpriseArchive (§1.38, optional)
    if self.cfg.get('persist_archive', False):
        self.bank.surprise_archive.save_state(self.cfg['archive_path'])

    # 3. Reset SSP + Lyapunov state
    self.diff_aux.cun._goal_stack  = []
    self.diff_aux.cun._stuck_count = []
    self.diff_aux.cun._v_prev      = []

    # 4. Reset HYPO state
    self.bank.g_c.zero_()
    self.bank._in_hypo_mode = False
    self.bank._r_lista_hypo = None
    self.bank._u_hypo       = 0.0

    # 5. Reset precision (per-session calibration)
    self.diff_aux.cun.sigma_sq_buffer   = [1.0] * 5
    self.diff_aux.cun._precision_active = [False] * 5
    # NOTE: log_precision is NOT reset (long-term learned parameter)

    # 6. Standard resets (r_lista, rule cache, etc.)
    self.diff_aux.cun.reset_lista_reservoir()
```

---

## 10. ACCUMULATE_FISHER — CORRECT IMPLEMENTATION

```python
def accumulate_fisher(model, stiefel_ids, bank, fisher):
    """§1.57: Fisher EMA for AdamW params. SI unchanged for Stiefel.
    
    RULE 1: Only for AdamW params (NOT stiefel_ids).
    RULE 2: Only if param.grad is not None.
    RULE 3: Skip frozen units — use bank.is_sensory_l (per-unit bool Tensor).
            DO NOT use bank.alpha_freeze — that is a scalar float threshold.
    """
    n_l = bank.n_l

    for name, param in model.named_parameters():
        if id(param) in stiefel_ids:
            continue                          # Rule 1: SI handles Stiefel
        if param.grad is None:
            continue                          # Rule 2

        # Rule 3: skip parameters belonging to a frozen unit
        unit_idx = _get_unit_idx_from_param_name(name, n_l)
        if unit_idx is not None and bank.is_sensory_l[unit_idx]:
            continue                          # frozen unit — Fisher locked

        # EMA toward squared gradient (Fisher diagonal approximation)
        if name not in fisher:
            fisher[name] = torch.zeros_like(param.data)
        fisher[name] = 0.99 * fisher[name] + 0.01 * param.grad.detach()**2


def update_fisher_magnitude_freeze(bank, fisher, k_sigma=1.5):
    """§1.63 C1: Fisher-magnitude parallel freeze trigger.
    Sets is_sensory_l[i] = True for high-Fisher units.
    bank.alpha_freeze (scalar) is UNCHANGED — still used by histogram path.
    """
    if bank.fisher_unit.sum() == 0:
        return  # no Fisher accumulated yet

    # Compute per-unit mean Fisher
    for i in range(bank.n_l):
        unit_fisher_vals = []
        for name, f in fisher.items():
            if _param_belongs_to_unit(name, i):
                unit_fisher_vals.append(f.mean().item())
        if unit_fisher_vals:
            bank.fisher_unit[i] = sum(unit_fisher_vals) / len(unit_fisher_vals)

    # Compute threshold: μ + 1.5σ
    active = bank.fisher_unit[:bank.n_l]
    mu = active.mean()
    sigma = active.std()
    threshold = mu + k_sigma * sigma

    # Freeze high-Fisher units (parallel to histogram-based freeze)
    new_frozen = (active > threshold) & ~bank.is_sensory_l[:bank.n_l]
    bank.is_sensory_l[:bank.n_l] |= new_frozen
```

---

## 11. VQ-TELESCOPE — KEY FUNCTIONS

```python
def vq_telescope_update(chunk_mean, sel_l, s_l, E_min_raw, chunk_token_ids, bank, cfg):
    """§1.59: Store VQ routing code in telescoping buffer."""
    ptr = bank._L1_ptr % bank.K_L1

    # Build full routing weight vector
    s_l_full = torch.zeros(bank.N_max_l)
    k_l_eff  = len(sel_l)
    s_l_full[sel_l] = s_l[:k_l_eff].float()

    bank.buf_L1_w_full[ptr] = s_l_full.detach()
    bank.buf_L1_ids[ptr]    = chunk_token_ids

    # Surprise detection (Welford-based, §1.68 — no fixed threshold)
    sigma_Emin = (bank._Emin_var / (bank._Emin_n + 1e-8))**0.5
    if E_min_raw > bank._Emin_mean + 2.0 * sigma_Emin:
        bank.surprise_archive.add_vq(ptr, E_min_raw)

    # Encoder commitment loss — gradient to chunk_mean ONLY
    z_approx = bank.mu_c_l[sel_l].mean(0).detach()    # DETACH codebook
    L_vq = (chunk_mean - z_approx).norm()**2           # gradient to encoder

    bank._L1_ptr += 1

    # L2/L3 averaging (amortised)
    _update_l2_l3_if_needed(bank)

    # Update Welford E_min statistics
    bank._Emin_n   += 1
    delta           = E_min_raw - bank._Emin_mean
    bank._Emin_mean += delta / bank._Emin_n
    bank._Emin_var  += delta * (E_min_raw - bank._Emin_mean)

    return L_vq


def vq_telescope_retrieve(s_l_full_query, bank):
    """§1.59: Retrieve past chunk by routing weight similarity."""
    n_l1 = min(bank._L1_ptr, bank.K_L1)
    if n_l1 == 0:
        return None, None, None, None

    # Dot product similarity in full routing weight space
    sim_L1  = bank.buf_L1_w_full[:n_l1] @ s_l_full_query
    top_L1  = int(sim_L1.argmax().item())

    # Reconstruct embedding from routing weights × centroids
    w1 = bank.buf_L1_w_full[top_L1]                           # (N_max_l,)
    r_L1 = (w1.unsqueeze(-1) * bank.mu_c_l[:bank.n_l].T).sum(-1).to(torch.cfloat)

    # L2, L3 analogously
    r_L2 = _retrieve_level(bank.buf_L2_w_full, s_l_full_query, bank.mu_c_l, bank.n_l)
    r_L3 = _retrieve_level(bank.buf_L3_w_full, s_l_full_query, bank.mu_c_l, bank.n_l)

    return r_L1, r_L2, r_L3, bank.buf_L1_ids[top_L1]
```

---

## 12. SPAWN THRESHOLD — WELFORD-BASED (§1.69)

```python
def should_spawn(bank, E_min_raw):
    """§1.69: Data-adaptive spawn threshold (replaces fixed 3.0).
    Uses same Welford buffers as §1.68 surprise detection.
    2.5σ threshold (more selective than surprise's 2.0σ).
    """
    if bank._Emin_n < 10:
        # Fall back to fixed threshold until statistics stabilise
        return E_min_raw > 3.0
    sigma_Emin = (bank._Emin_var / bank._Emin_n)**0.5
    return E_min_raw > bank._Emin_mean + 2.5 * sigma_Emin
```

---

## 13. PRECISION UPDATE — REPLACE MC-2 (§1.58)

```python
def update_precision(cun, signals, is_hypo_active=False, is_hopfield_active=True):
    """§1.58: Self-calibrating precision. Replaces MC-2 _log_w_rec EMA.
    
    signals: [U_repr_q, U_epi_cal, U_hopfield, U_temporal, U_hypo]
    MUST init sigma_sq_buffer = [1.0]*5 NOT [0.0]*5
    """
    for s, sig in enumerate(signals):
        val = float(sig)

        # Pathway activity gates (§1.58 Issue 2 fix)
        if s == 4 and not is_hypo_active:    continue  # U_hypo inactive
        if s == 2 and not is_hopfield_active: continue  # U_hopfield disabled

        # Activate pathway on first non-zero signal
        if abs(val) > 1e-6:
            cun._precision_active[s] = True
        if not cun._precision_active[s]:
            continue  # hold sigma_sq=1.0 (equal weight) until activated

        # EMA toward squared signal
        cun.sigma_sq_buffer[s] = 0.95 * cun.sigma_sq_buffer[s] + 0.05 * val**2
        lp = -0.5 * math.log(cun.sigma_sq_buffer[s] + 1e-6)
        cun.log_precision[s] = max(-3.0, min(3.0, lp))   # clamp [-3, 3]

    # Precision-weighted U_meta
    prec     = torch.exp(torch.tensor(cun.log_precision))
    signals_t = torch.tensor(signals, dtype=torch.float32)
    U_meta    = (prec * signals_t).sum() / (prec.sum() + 1e-8)
    return float(U_meta.item())
```

---

## 14. Q_BEAM COMPOSITE (§1.46, §1.53)

```python
def compute_Q_beam(h_N, r_lista, r_goal_proxy, goal_stack, x_c,
                   W_bridge=None, E_min_raw=None, H_route_raw=None,
                   log_w_beam=None, phi_rel=None):
    """§1.46+§1.53: Multi-field beam quality score. No MLP. No learned labels.
    
    F1: Thermodynamics — -(E_min_raw × H_route_raw)   [optional]
    F2: Predictive coding — -||r_seed_target - W_bridge@rho||²  [optional, needs W_bridge trained]
    F3: MDL — -||h_N||₁ (sparse = good)
    F4: Lyapunov — -||r_lista - r_goal_proxy||²  [only if g_c active]
    F5: CSP arc — min cosine_sim(r_lista, stack_entry)  [only if stack non-empty]
    D1: phi_rel richness — phi_rel.norm()  [only if phi_rel cached]
    """
    signals = []

    # F3: MDL sparsity (always)
    signals.append(-h_N.abs().sum())

    # F4: Lyapunov goal distance (when g_c active)
    if r_goal_proxy is not None:
        signals.append(-(r_lista - r_goal_proxy).norm()**2)

    # F5: CSP arc-consistency (when stack non-empty)
    if goal_stack:
        sims = [F.cosine_similarity(r_lista.real.unsqueeze(0),
                                     s.real.unsqueeze(0)).item()
                for s in goal_stack]
        signals.append(min(sims))

    # D1: relational context richness
    if phi_rel is not None:
        signals.append(float(phi_rel.norm().item()))

    # F1: thermodynamics (optional — requires E_min_raw, H_route_raw)
    if E_min_raw is not None and H_route_raw is not None:
        signals.append(-(E_min_raw * H_route_raw))

    # F2: predictive coding (optional — requires trained W_bridge)
    if W_bridge is not None:
        r_seed_target = None  # computed from U1 @ x_c in context
        # See §1.50 for L_bridge formula
        pass

    if not signals:
        return torch.tensor(0.0)

    signals_t = torch.stack([s if isinstance(s, torch.Tensor)
                              else torch.tensor(float(s)) for s in signals])

    # Equal weighting (parameter-free) or optional learned weights
    if log_w_beam is not None and len(log_w_beam) >= 3:
        w = torch.softmax(log_w_beam[:len(signals)], dim=0)
        return (w * signals_t).sum()
    return signals_t.mean()
```

---

## 15. TESTS TO WRITE / UPDATE

All tests from §1.62 plus updates for the alpha_freeze fix:

```python
# §1.62 CORRECTED tests (alpha_freeze fix applied):

def test_W_rc_bridge_is_trained_parameter():
    """W_rc_bridge must be nn.Parameter (§1.50)."""
    assert isinstance(model.W_rc_bridge, nn.Parameter)
    assert model.W_rc_bridge.requires_grad

def test_L_bridge_has_gradient_to_W_bridge():
    """L_bridge must give non-zero gradient to W_rc_bridge (§1.50)."""
    r_seed = model.W_rc_bridge @ rho
    target = (cun.U1.conj() @ x_c.mean())[:r_seed.shape[0]]
    L_bridge = (target.detach() - r_seed).norm()**2
    L_bridge.backward()
    assert model.W_rc_bridge.grad is not None
    assert model.W_rc_bridge.grad.norm() > 0

def test_fisher_not_updated_for_frozen_units():
    """Fisher must NOT accumulate for is_sensory_l=True units (§1.57).
    NOTE: bank.alpha_freeze is a scalar float. bank.is_sensory_l is the per-unit mask.
    """
    bank.is_sensory_l[0] = True   # NOT bank.alpha_freeze[0] = 1
    # ... rest of test uses is_sensory_l

def test_sigma_sq_buffer_init_unit_variance():
    """sigma_sq_buffer must init to [1.0]*5 to prevent precision explosion (§1.58)."""
    for val in cun.sigma_sq_buffer:
        assert abs(val - 1.0) < 1e-6

def test_L_vq_gradient_only_to_encoder():
    """L_vq must NOT give gradient to mu_c_l (codebook trained by CNEP) (§1.59)."""
    z_approx = bank.mu_c_l[sel_l].mean(0).detach()  # detached
    L_vq = (chunk_mean - z_approx).norm()**2
    L_vq.backward()
    assert bank.mu_c_l.grad is None or bank.mu_c_l.grad.norm() < 1e-6

def test_compress_gradient_flows_to_cfl5layer():
    """§1.51: chunk_mean in L_compress must NOT be detached."""
    chunk_mean = torch.randn(4, dtype=torch.cfloat, requires_grad=True)
    c1_live = W_compress @ chunk_mean
    x_recon = W_compress.conj().T @ c1_live
    L_compress = ((chunk_mean - x_recon).conj() * (chunk_mean - x_recon)).real.sum()
    L_compress.backward()
    assert chunk_mean.grad is not None and chunk_mean.grad.norm() > 0
```

---

## 16. ABLATIONS (run in this priority order)

| Priority | ID | Tests | Validates |
|---|---|---|---|
| 1 | OQ-v598-2 | U_epi vs actual CE correlation r>0.4 | Metacognition A− claim |
| 2 | A97 | RAH role binding ON vs OFF | Compositionality/Analogy grade |
| 3 | A93 | 2-tier vs 3-tier CL benchmark | CL grade |
| 4 | A99 | SSP hierarchical vs flat CTP | Multi-step reasoning grade |
| 5 | OQ-VQ-1 | VQ retrieval vs Hopfield embedding | VQ-Telescope semantic quality |
| 6 | OQ-FISHER-1 | Fisher-KL vs SI forgetting curves | CL protection quality |
| 7 | A107 | W_bridge trained vs fixed buffer | r_lista seed quality |
| 8 | A114 | Fisher-KL vs SI (AdamW only) | §1.57 theoretical improvement |
| 9 | A116 | VQ-Telescope vs W_compress | Context window A→A− tradeoff |

---

## 17. KNOWN INVARIANTS — NEVER VIOLATE

1. `B_bind`, `B_role`, `B_comp` are all PSD → `apply_psd` on `W_full` is always valid
2. `W_full[:k_l,:k_l] += lam_bind×B_bind + lam_role×B_role + lam_comp×B_comp` (all three terms)
3. `role_vecs` excluded from Muon: `stiefel_ids.add(id(bank.role_vecs))`
4. `tau_smooth.clamp(min=1e-3)` in STELA formula — prevents sign-flip
5. Fisher accumulates BEFORE `clip_grad_norm_` in every training step
6. `sigma_sq_buffer` init = `[1.0]*5`, NOT `[0.0]*5`
7. `L_vq` uses `mu_c_l[sel_l].mean(0).detach()` — codebook gradient from CNEP only
8. `CONSOL-1` called BEFORE session state reset in `reset_for_inference`
9. `bank.alpha_freeze` = scalar float. `bank.is_sensory_l` = per-unit bool Tensor
10. `U1`, `U2` never in any optimizer (fixed unitary basis for LISTA)
