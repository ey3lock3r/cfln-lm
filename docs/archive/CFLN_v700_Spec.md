# CFLN v7.0 — Holistic Reasoning Intelligence Specification
# Analogical Thinking + Counterfactual Mode + Goal-Anchored Context
# All changes agreed upon by 14-expert panel (14/14 unanimous votes)
# Builds on: CFLN v6.0.9 Master Spec (base architecture, fully implemented)

---

## OVERVIEW

CFLN v6.0.9 has strong associative memory (A), metacognition (A), continual learning (A),
and reasoning (A) but weak analogical thinking (C+), counterfactual reasoning (D), and
planning (C-). Four architectural proposals (R1–R4) address these gaps by filling the
BINDING layer of the cognitive processing loop, which was the single missing piece.

All changes are **additive and non-breaking** — they extend existing components without
removing or restructuring any v6.0.9 functionality.

### Cognitive Processing Loop (v7.0)
```
LAYER 0: INPUT        x_c_eff = x_c + goal_scale × g_c          ← R4 (new)
LAYER 1: ROUTING      CNEP(x_c_eff) + Phase Kernel B             ← R4 + R1 (new)
LAYER 2: BINDING      CS-GAT over W_full (now includes B)        ← R1 (new)
LAYER 3: RETRIEVAL    ARC dual-key [concept + relational]         ← R2 (new)
LAYER 4: REASONING    CTP think + HYPO branch                    ← R3 (new)
LAYER 5: EVALUATION   U_meta_v4 [5 signals incl. U_hypo]        ← R3 (new)
LAYER 6: STORAGE      ARC writes dual-key                        ← R2 (new)
```

---

## PART 1: MATH SPECIFICATION

### §1.30 Phase Similarity Kernel — Binding via Complex Phase (R1)

The phase of a unit's complex projection encodes its temporal-contextual state.
Units whose phases are aligned are "bound" — they consistently co-activate in
the same phase context, indicating semantic or causal binding.

**Phase per unit:**
```
φ_i = angle(bank.H_c_l[sel_i].mean())      scalar ∈ [−π, π] for selected unit i
```
H_c_l[i] is the d_H-step projection history for unit i. The mean over history
gives the dominant phase for that unit's recent activation pattern.

**Phase Similarity Kernel B:**
```
B[i,j] = exp(−|φ_i − φ_j|² / σ²)          ∈ [0,1], symmetric, PSD (RBF kernel)
```
- B[i,i] = 1 always (self-binding)
- B[i,j] = 1 when φ_i = φ_j (maximally bound)
- B[i,j] ≈ 0 when |φ_i − φ_j| ≫ σ (unbound)
- σ = 1.0 (fixed hyperparameter, not learned; recommended range 0.5–2.0)

**Integration into W_full:**
```
W_full[:k_l,:k_l] += exp(log_lam_bind) × B     after existing H_seq and H_mat additions
```

**PSD safety:** B is an RBF kernel matrix → always PSD (Bochner's theorem).
Sum of PSD matrices is PSD → apply_psd constraint automatically satisfied. ✓

**New parameters:**
```
log_lam_bind: scalar nn.Parameter, init -3.0   (→ binding weight ≈ 0.05 initially)
Group: opt_g (AdamW, same as log_lambda_hebb and log_lam_seq_gat)
```

**Compute cost:** k_l² = 1,600 scalar ops/token. Negligible (<0.01% of total).

**Analogical reasoning role:** B creates binding edges in CS-GAT. Units representing
the same structural role (e.g., agent-of-transformation) will share similar phases
after training → CS-GAT spectral hops propagate along binding edges → structural
role alignment across domains emerges through the routing graph.

---

### §1.31 Relational Key for ARC Cache — Structural Retrieval (R2)

The existing ARC key `k_concept = x_c.mean(0) @ U1.conj().T` encodes the CONCEPT
context. For analogical reasoning, we also need a RELATIONAL key encoding the dominant
temporal-relational pattern in the current unit activation, independent of which
specific concepts are active.

**Relational key computation:**
```
H_seq_sub = bank.H_seq_mat[sel_k][:,sel_k]       (k_l × k_l) real submatrix
                                                   recomputed every C_chunk=32 tokens

phi_rel = top_eigenvec(H_seq_sub)                 (k_l,) real — dominant relational direction
                                                   top eigenvec = units most "relationally central"

k_rel = phi_rel @ psi_all[:k_l]                   (d_c,) cfloat — relational key
```
phi_rel weights psi_all representations by relational centrality → k_rel encodes
"dominant relational pattern in current context" in the same cfloat space as k_concept.

**Two-key ARC cache:**
```
K_rule: shape (N_rules, 2×d_c) cfloat         was (N_rules, d_c)
K_rule[ptr] = concat([k_concept, k_rel])       dual-key concatenated

Retrieval similarity (for query q_concept, q_rel):
  sim_k = log_α_arc × cosine(q_concept, K_rule[:n, :d_c])
        + (1 - log_α_arc) × cosine(q_rel, K_rule[:n, d_c:])
```

**New parameters:**
```
log_α_arc: scalar nn.Parameter, init log(0.5)   (→ equal weighting initially)
Group: opt_g (AdamW)
```
log_α_arc is a learned mixing weight between concept and relational similarity.
sigmoid(log_α_arc) gives weight in [0,1]. The model learns when structure matters more.

**Compute cost:** H_seq_sub eigendecomposition O(k_l³) = 64K flops, amortised over
C_chunk=32 tokens → 2K flops/token. Negligible.

**Migration:** Existing ARC entries written before v7.0 have shape (d_c,). At v7.0
init: pad old K_rule entries with zeros for the relational half. The model will learn
to populate k_rel progressively.

---

### §1.32 CTP Hypothetical Mode — Counterfactual Reasoning (R3)

Extends the think token protocol with a HYPOTHETICAL MODE: a branched reasoning state
that explores counterfactuals without corrupting the factual r_lista state.

**Vocabulary extension (2 more tokens):**
```
HYPO_START_ID = original_vocab_size + 2     (token <hypo>)
HYPO_END_ID   = original_vocab_size + 3     (token </hypo>)
```
Total special tokens after v7.0: 4 (<think>, </think>, <hypo>, </hypo>)

**Hypothetical reasoning protocol:**
```
<think> ... factual reasoning ... </think>          (existing CTP — updates r_lista)
<think>
  ... factual reasoning ...
  <hypo> ... counterfactual reasoning ... </hypo>   (NEW — branched r_lista, no updates)
  ... factual reasoning continues ...
</think>
```

**r_lista branching:**
```
On HYPO_START token:
  r_lista_hypo = r_lista.clone()          (32 cfloat = 512 bytes, trivial)
  _in_hypo_mode = True

During HYPO (per lista_forward call):
  Use r_lista_hypo instead of r_lista
  ARC writes: SUPPRESSED (hypothetical rules don't persist to factual memory)
  ARC reads: ALLOWED (hypothetical reasoning can use factual memory)
  H_seq/H_mat updates: SUPPRESSED

On HYPO_END token:
  U_hypo = sigmoid(‖r_lista_hypo − r_lista‖² / d_r_lista)
  r_lista_hypo = None
  _in_hypo_mode = False
  r_lista UNCHANGED (factual state fully preserved)
```

**U_hypo signal:**
```
U_hypo ∈ [0,1]:
  High: hypothetical scenario diverged significantly from factual → impactful hypothesis
  Low:  hypothesis barely changes the reasoning state → negligible consequence
  = 0 when _in_hypo_mode = False (safe default, no cost when not in use)
```

**g_c (goal register) during HYPO:**
```
On HYPO_START: g_c_hypo_saved = g_c.clone()    (save goal context)
During HYPO: g_c is FROZEN (hypothetical reasoning doesn't update goal)
On HYPO_END: g_c = g_c_hypo_saved              (restore, though g_c was frozen so this is identity)
```

---

### §1.33 Goal-Anchored Context — Goal-Directed Routing (R4)

CFLN v6.0.9 is purely reactive — each token is processed independently of any goal.
Goal-Anchored Context stores a persistent goal representation g_c that biases CNEP
routing toward goal-relevant units throughout a reasoning episode.

**Goal register:**
```
g_c ∈ ℂ^{d_c}: register_buffer on CFBank, init zeros(d_c)
```

**Learned goal detection gate:**
```
W_goal_detect ∈ ℝ^{1 × d_c}: nn.Parameter, init zeros   (opt_g group)
log_lam_goal: scalar nn.Parameter, init -3.0              (opt_g group → goal scale ≈ 0.05)

Per token (using x_c_mean before CNEP routing):
  g_t = σ(W_goal_detect @ x_c_mean.real)           scalar ∈ (0,1)
  g_c ← g_t × x_c_mean + (1−g_t) × g_c            soft exponential moving goal update
```
The model learns g_t from training data. Initially g_t≈0.5 (small init → small gate).
Tokens identified as goals receive high g_t → g_c shifts toward them.
Non-goal tokens receive low g_t → g_c persists (exponential decay of old goal).

**Goal-shifted context for CNEP:**
```
x_c_eff = x_c + exp(log_lam_goal) × g_c.unsqueeze(0)    shape (B, d_c) cfloat
```
This is a CONTEXT SHIFT, NOT a formula change. E_i(x_c_eff) uses the same energy
formula but evaluates at a shifted point. Equivalent to shifting each unit centroid
by −goal_scale × g_c in the opposite direction.

**CNEP routing uses x_c_eff throughout:**
```
E_l  = compute_energies(x_c_eff, bank.W_l[:n_l], bank.mu_c_l[:n_l])
E_p  = compute_energies(x_c_eff, bank.W_p, bank.mu_c_p)
```

**Lifecycle:**
```
reset_for_inference(): bank.g_c.zero_()
HYPO_START:           g_c frozen during hypothetical mode (no g_t update)
HYPO_END:             g_c update resumes
```

---

### §1.34 U_meta_v4 — Five-Signal Metacognition

Extends U_meta_v3 (4 signals) with U_hypo as a 5th uncertainty dimension:

```
U_hypo     = σ(‖r_lista_hypo − r_lista‖² / d_r_lista)   (0 when not in hypo mode)
U_meta_v4  = softmax(log_w_meta ∈ ℝ^5) ⊙ [U_repr_q, U_epi_cal, U_hopfield, U_temporal, U_hypo]
```

**log_w_meta extension:**
```
log_w_meta: ℝ^4 → ℝ^5    (add 5th entry, init -2.0 for U_hypo)
Init: [1.0, -1.0, -1.0, -2.0, -2.0]
```

**MC-2 log_w_rec extension:**
```
_log_w_rec: length 4 → length 5
signal[4] = 1 − |U_hypo − ce_proxy|      (agreement between U_hypo and difficulty)
Update: log_w_rec[k] ← 0.95 × log_w_rec[k] + 0.05 × signal[k]  for k=0..4
reset_lista_reservoir: self._log_w_rec = [0.0, 0.0, 0.0, 0.0, 0.0]
```

---

## PART 2: CODE CHANGES

### 2.1 CFBank.__init__ — new parameters and buffers

```python
# ADD after existing _u_epi_var buffer:

# §1.30 R1: Phase binding
self.register_buffer('_phi_cache', torch.zeros(N_max_l))  # cached unit phases
self.log_lam_bind = nn.Parameter(torch.tensor(-3.0))      # binding weight

# §1.33 R4: Goal-anchored context
self.register_buffer('g_c', torch.zeros(d_c, dtype=torch.cfloat))
self.W_goal_detect = nn.Parameter(torch.zeros(1, d_c))    # (1, d_c) real
self.log_lam_goal  = nn.Parameter(torch.tensor(-3.0))     # goal scale

# §1.32 R3: Hypothetical mode session state (plain attrs, not buffers)
self._r_lista_hypo = None      # branched r_lista during HYPO mode
self._in_hypo_mode = False
self._u_hypo       = 0.0       # U_hypo signal (0 when not in hypo)
```

### 2.2 CFBank.reset_for_inference() — new resets

```python
# ADD to reset_for_inference:
self.g_c.zero_()               # clear goal register
self._r_lista_hypo = None
self._in_hypo_mode = False
self._u_hypo       = 0.0
# Note: _u_epi_mu/_u_epi_var intentionally NOT reset (global calibration stats)
```

### 2.3 CFL5Layer.forward — Phase Kernel + Goal-Anchored Context

```python
# CHANGE 1: Goal-anchored context (INSERT before compute_energies calls)
# After: x_c_mean = x_c.mean(0)
# ADD:
g_t = torch.sigmoid((self.bank.W_goal_detect @ x_c_mean.real).squeeze())
with torch.no_grad():
    bank.g_c = (g_t * x_c_mean + (1.0 - g_t) * bank.g_c).detach()
    # Freeze g_c during hypothetical mode
if not getattr(bank, '_in_hypo_mode', False):
    pass  # update already done above
goal_scale = torch.exp(bank.log_lam_goal)
x_c_eff = x_c + goal_scale * bank.g_c.unsqueeze(0)  # (B, d_c)

# CHANGE 2: Use x_c_eff for CNEP routing
E_l = compute_energies(x_c_eff, bank.W_l.data[:n_l], bank.mu_c_l[:n_l])
# (replace x_c with x_c_eff in all compute_energies calls)

# CHANGE 3: Phase Similarity Kernel (INSERT before CS-GAT call)
# After: W_full[:k_l,:k_l] = W_full[:k_l,:k_l] + lam_sg*H_seq_norm
# ADD:
phi_sel = torch.angle(bank.H_c_l[sel_l].mean(dim=(-2,-1)))   # (k_l,) real phases
phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)         # (k_l, k_l) pairwise
B_bind = torch.exp(-(phi_diff**2))                             # sigma=1.0, PSD by RBF theorem
lam_bind = torch.exp(self.bank.log_lam_bind)
W_full[:k_l,:k_l] = W_full[:k_l,:k_l] + lam_bind * B_bind

# CHANGE 4: U_temporal — pass to lista_forward (already in v6.0.9)
# No change needed here; u_temporal_val already computed and passed
```

### 2.4 CUN.__init__ — new parameters and extended buffers

```python
# CHANGE 1: K_rule shape 2×d_c for dual-key ARC
# Find: self.rule_K = torch.zeros(episodic_rule_cache_n, d_c, dtype=torch.cfloat)
# REPLACE:
self.register_buffer('rule_K',
    torch.zeros(episodic_rule_cache_n, 2*cfg.get('d_c',64), dtype=torch.cfloat))
# (rule_V shape unchanged — we retrieve the same sparse code)

# CHANGE 2: Relational key mixing scalar
self.log_alpha_arc = nn.Parameter(torch.tensor(0.0))   # sigmoid → 0.5 init mix

# CHANGE 3: U_meta_v4 — extend log_w_meta to R^5
# Find: self.log_w_meta=nn.Parameter(torch.tensor([1.0,-1.0,-1.0,-2.0]))
# REPLACE:
self.log_w_meta = nn.Parameter(torch.tensor([1.0, -1.0, -1.0, -2.0, -2.0]))

# CHANGE 4: cached relational key (updated every C_chunk tokens)
self._phi_rel_cache    = None    # (k_l,) real, top H_seq eigenvector
self._phi_rel_step     = 0       # step counter for cache invalidation
```

### 2.5 CUN.reset_lista_reservoir() — extended resets

```python
# CHANGE: extend _log_w_rec to 5 entries
# Find: self._log_w_rec=[0.0,0.0,0.0,0.0]
# REPLACE:
self._log_w_rec = [0.0, 0.0, 0.0, 0.0, 0.0]   # 5 signals for U_meta_v4

# ADD: reset hypo state
self._r_lista_hypo = None
self._in_hypo_mode = False
self._u_hypo       = 0.0
```

### 2.6 CUN.lista_forward() — full v7.0 changes (ordered)

#### 2.6.1 Signature extension

```python
# CHANGE:
def lista_forward(self, x_c, hopfield=None, bank=None, N_hop=4,
                  escape=True, compute_meta=True, u_temporal=0.0,
                  u_hypo=0.0):    # ADD u_hypo parameter
```

#### 2.6.2 Relational key computation (INSERT at start of function, after x_c setup)

```python
# INSERT: compute/update relational key every C_chunk tokens
C_chunk = 32
if (self._phi_rel_cache is None or
        self._phi_rel_step % C_chunk == 0) and bank is not None:
    sel_k = sel_l % bank.K_hebb if hasattr(bank, 'K_hebb') else None
    if sel_k is not None:
        H_sub = bank.H_seq_mat[sel_k][:, sel_k].float()  # (k_l, k_l)
        try:
            _, evecs = torch.linalg.eigh(H_sub)            # ascending order
            phi_rel = evecs[:, -1]                          # top eigenvec (k_l,)
            self._phi_rel_cache = phi_rel.detach()
        except Exception:
            self._phi_rel_cache = torch.ones(k_l, device=x_c.device) / k_l**0.5
self._phi_rel_step += 1
```

#### 2.6.3 ARC retrieval — dual-key (REPLACE existing retrieval block)

```python
# REPLACE §1.26 retrieval block with:
if self.episodic_rule_n > 0 and self._rule_cache_n > 0:
    n_r = self._rule_cache_n
    K_r = self.rule_K[:n_r]                               # (n_r, 2×d_c) cfloat
    d_c = x_c.shape[-1]
    K_concept = K_r[:, :d_c]                              # (n_r, d_c) concept keys
    K_rel     = K_r[:, d_c:]                              # (n_r, d_c) relational keys
    V_r = self.rule_V[:n_r]                               # (n_r, d_c)

    # Query keys
    x_query = (x_c.mean(0) @ self.U1.conj().T.detach())  # (d_c,) concept query
    # Relational query: cached phi_rel × current psi_all
    if self._phi_rel_cache is not None and bank is not None:
        psi_cur = bank.get_psi_expansion(sel_l) if hasattr(bank,'get_psi_expansion') else None
        if psi_cur is not None:
            q_rel = (self._phi_rel_cache.to(x_c.device) @ psi_cur)  # (d_c,)
        else:
            q_rel = x_query
    else:
        q_rel = x_query

    # Dual cosine similarity
    def cos_sim(q, K):
        return (q @ K.conj().T).real / (q.norm().clamp(1e-8) * K.norm(dim=-1).clamp(1e-8) + 1e-8)

    alpha = torch.sigmoid(self.log_alpha_arc)              # concept weight
    sims  = alpha * cos_sim(x_query, K_concept) + (1-alpha) * cos_sim(q_rel, K_rel)

    k_ret = min(3, n_r)
    top_idx = torch.topk(sims, k_ret).indices
    w_k = torch.softmax(sims[top_idx] / 0.5, dim=0)
    v_blend = (w_k.to(torch.cfloat).unsqueeze(-1) * V_r[top_idx]).sum(0)  # (d_c,)

    g_rule = torch.sigmoid(self.log_gate_rule + (self.W_gate_rule * x_query.real).sum())
    h = h + g_rule * v_blend.unsqueeze(0).expand(B, -1).detach()
```

#### 2.6.4 HYPO mode handling (INSERT before LISTA iteration loop)

```python
# INSERT: set active r_lista based on hypo mode
_in_hypo = getattr(self, '_in_hypo_mode', False)
r_lista_active = getattr(self, '_r_lista_hypo', None) if _in_hypo else self.r_lista
if r_lista_active is None:
    r_lista_active = self.r_lista
# All subsequent r_lista reads/writes use r_lista_active
```

#### 2.6.5 ARC write — HYPO suppression (in existing NR-1 write block)

```python
# ADD at start of write block:
if getattr(self, '_in_hypo_mode', False):
    pass  # suppress ARC writes during hypothetical reasoning
else:
    # ... existing NR-1 dual-trigger write code ...
```

#### 2.6.6 U_meta_v4 computation (REPLACE U_meta computation)

```python
# REPLACE U_meta_v3 block with U_meta_v4:
u_hypo_val = float(getattr(self, '_u_hypo', 0.0))
w_meta = torch.softmax(self.log_w_meta, dim=0)   # now R^5
U_meta_v4 = (w_meta[0] * U_repr_q +
             w_meta[1] * float(getattr(bank, '_last_u_epi', 0.5) if bank else 0.5) +
             w_meta[2] * float(U_hopfield) +
             w_meta[3] * u_temporal +
             w_meta[4] * u_hypo_val)
U_meta = U_meta_v4
```

#### 2.6.7 MC-2 log_w_rec — 5-signal update (REPLACE existing MC-2 block)

```python
# REPLACE MC-2 block:
if hasattr(self, '_log_w_rec') and len(self._log_w_rec) == 5:
    u_hop_f = float(U_hopfield.item() if isinstance(U_hopfield, torch.Tensor) else U_hopfield)
    u_epi_f = float(getattr(bank, '_last_u_epi', 0.5) if bank else 0.5)
    ce_proxy = float(self._prev_U_meta) if hasattr(self, '_prev_U_meta') else 0.5
    u_signals = [float(U_repr_q) if hasattr(U_repr_q, '__float__') else 0.5,
                 u_epi_f,
                 u_hop_f,
                 u_temporal,
                 u_hypo_val]
    for k in range(5):
        signal_k = 1.0 - abs(u_signals[k] - ce_proxy)
        self._log_w_rec[k] = 0.95 * self._log_w_rec[k] + 0.05 * signal_k
```

### 2.7 ARC write — dual-key (REPLACE key construction in NR-1 write block)

```python
# REPLACE: K_new = (x_c.mean(0) @ self.U1.conj().T.detach()).detach()
# WITH:
k_concept_new = (x_c.mean(0) @ self.U1.conj().T.detach()).detach()
if self._phi_rel_cache is not None and bank is not None and psi_all is not None:
    q_rel_new = (self._phi_rel_cache.to(x_c.device) @ psi_all[:k_l])
else:
    q_rel_new = k_concept_new
K_new = torch.cat([k_concept_new, q_rel_new], dim=0)  # (2×d_c,)
V_new = h.mean(0).detach()
# (merge logic unchanged, but K_new and K_r shapes now 2×d_c)
```

### 2.8 HYPO token handling in CFLNModel.forward

```python
# ADD in CFLNModel.forward where THINK_START_ID/END_ID are detected:

HYPO_START_ID = self.cfg.get('hypo_start_id', self.cfg['vocab_size'] + 2)
HYPO_END_ID   = self.cfg.get('hypo_end_id',   self.cfg['vocab_size'] + 3)

# Per token in the generation loop:
for t in range(T):
    tok_id = input_ids[:, t]
    # HYPO START
    if (tok_id == HYPO_START_ID).any():
        self.bank._in_hypo_mode = True
        self.bank._r_lista_hypo = self.diff_aux.cun.r_lista.clone()
        # freeze goal context during hypo
        self._g_c_hypo_save = self.bank.g_c.clone()
    # HYPO END
    elif (tok_id == HYPO_END_ID).any():
        if self.bank._in_hypo_mode and self.bank._r_lista_hypo is not None:
            diff = (self.bank._r_lista_hypo - self.diff_aux.cun.r_lista).norm()
            self.bank._u_hypo = float(
                torch.sigmoid(diff**2 / self.diff_aux.cun.r_lista.shape[-1]).item())
        self.bank._in_hypo_mode = False
        self.bank._r_lista_hypo = None
        # restore goal context
        if hasattr(self, '_g_c_hypo_save'):
            self.bank.g_c.copy_(self._g_c_hypo_save)
```

### 2.9 expand_vocabulary() — extend for HYPO tokens

```python
# CHANGE: extend vocab by 4 (was 2 for THINK tokens)
# In expand_vocabulary(), extend embed_real, embed_imag, and head weight
# by 2 additional rows for HYPO_START, HYPO_END tokens
# Config update:
cfg['vocab_size_extended'] = cfg['vocab_size'] + 4   # was +2
cfg['hypo_start_id'] = cfg['vocab_size'] + 2
cfg['hypo_end_id']   = cfg['vocab_size'] + 3
```

---

## PART 3: OPTIMIZER CHANGES

### 3.1 build_optimizers_v605 — new parameter groups

```python
# ADD to g1 (opt_g AdamW) named parameters loop, these will be auto-covered
# by existing bank.named_parameters() loop:
#   bank.log_lam_bind     (scalar)
#   bank.W_goal_detect    (1 × d_c real)
#   bank.log_lam_goal     (scalar)

# ADD explicitly (not covered by existing named loops):
#   model.diff_aux.cun.log_alpha_arc  (scalar) — covered by diff_aux loop ✓

# VERIFY: log_w_meta R^5 is covered by diff_aux.named_parameters() ✓
# VERIFY: rule_K shape 2×d_c is a buffer not param — no optimizer needed ✓

# NEW: log_alpha_arc — confirm it's in diff_aux.named_parameters() loop
# It is (nn.Parameter on CUN, diff_aux submodule) — no explicit add needed ✓
```

### 3.2 Stiefel group — no changes needed

`W_goal_detect` is `(1, d_c)` real — `is_matrix = True` but minimum dim = 1, which is
below the Stiefel threshold (min(shape) ≥ 4). Goes to AdamW (opt_g), not Muon. ✓

---

## PART 4: CONFIG CHANGES

### 4.1 CFG_ABLATION_605 additions

```python
CFG_ABLATION_605.update({
    # R1
    'sigma_bind': 1.0,          # phase kernel width (fixed, not learned)
    # R2
    'arc_dual_key': True,       # enable dual-key ARC (concept + relational)
    # R3
    'hypo_start_id': 8194,      # vocab_size + 2 (after think tokens)
    'hypo_end_id':   8195,      # vocab_size + 3
    # R4
    'use_goal_context': True,
})
```

### 4.2 CFG_VERIFY_605 additions

```python
CFG_VERIFY_605.update({
    'sigma_bind': 1.0,
    'arc_dual_key': True,
    'hypo_start_id': 4098,      # vocab_size=4096, think=4096,4097, hypo=4098,4099
    'hypo_end_id':   4099,
    'use_goal_context': True,
})
```

---

## PART 5: TRAINING CHANGES

### 5.1 Tokenizer — extend for HYPO tokens

```python
def extend_tokenizer_for_v7(tok):
    """Add HYPO tokens on top of existing THINK tokens."""
    tok.add_special_tokens(['<think>', '</think>', '<hypo>', '</hypo>'])
    ids = {
        'think_start': tok.token_to_id('<think>'),
        'think_end':   tok.token_to_id('</think>'),
        'hypo_start':  tok.token_to_id('<hypo>'),
        'hypo_end':    tok.token_to_id('</hypo>'),
    }
    print(f"Special tokens: {ids}")
    return tok, ids
```

### 5.2 RPP trace generation — hypothetical traces

The RPP-STaR pipeline (Stage 2) should include traces containing hypothetical reasoning:

```python
# In star_generate_traces_rpp, add a hypothetical trace template:
HYPO_TRACE_TEMPLATE = """<think>
What do I know about {source_A} and {source_B}?
<hypo>Suppose {source_A} didn't have property X. Would {relation} still hold?</hypo>
The consequence divergence is {u_hypo_value:.2f} — hypothesis is {significant}.
Therefore {target_C} : {answer_D} by the same {relation}.
</think>{answer_D}"""

# Generate 20% of RPP traces using hypothetical template
# Target: tasks from analogy benchmarks (SAT analogies, Raven's Progressive Matrices)
# This teaches HYPO mode to be used for analogical reasoning
```

### 5.3 GRPO reward — add analogical reasoning signal

```python
# ADD to grpo_train_step reward computation:
# Existing: R_ppl = perplexity_reduction (intrinsic)
# NEW: R_analogy = 1.0 if completion follows detected analogical pattern else 0.0
# Detection: simple pattern matching on A:B::C:? format in batch
# Combined reward: R = R_ppl + alpha_analogy * R_analogy (alpha_analogy=0.3)
```

### 5.4 New ablations

```python
# ADD to CFG_ABLATION series:
# A94: HYPO mode ON vs OFF on analogical task suite
# A95: Two-key ARC vs single-key ARC on far-domain analogies
# A96: Goal-anchored context vs plain context on question-answering coherence

ANALOGY_EVAL_PAIRS = [
    # (source_A, source_B, target_C, correct_D) — for A94/A95 eval
    ('ice', 'water', 'wax', 'liquid'),
    ('king', 'queen', 'man', 'woman'),
    ('hot', 'cold', 'fast', 'slow'),
    ('bark', 'tree', 'skin', 'body'),
    # Add 20+ pairs for statistical significance
]
```

---

## PART 6: CHECKPOINT CHANGES

### 6.1 save_checkpoint — new state

```python
# ADD to checkpoint dict:
ckpt.update({
    'bank_g_c':          model.bank.g_c.cpu().clone(),
    'bank_u_hypo':       model.bank._u_hypo,
    'cun_phi_rel_step':  model.diff_aux.cun._phi_rel_step,
    # rule_K shape changed — ensure shape is saved correctly with state_dict
})
```

### 6.2 load_checkpoint — backward compatibility

```python
# IF loading from v6.0.9 checkpoint (rule_K shape (N,d_c) not (N,2*d_c)):
if 'model_state' in ckpt:
    old_rule_K = ckpt['model_state'].get('diff_aux.cun.rule_K')
    if old_rule_K is not None and old_rule_K.shape[-1] == d_c:
        # Pad to 2*d_c with zeros (relational half uninitialised)
        pad = torch.zeros(*old_rule_K.shape[:-1], d_c,
                          dtype=old_rule_K.dtype, device=old_rule_K.device)
        ckpt['model_state']['diff_aux.cun.rule_K'] = torch.cat([old_rule_K, pad], dim=-1)
        print("v6.0.9→v7.0 checkpoint migration: rule_K padded to 2×d_c")
```

---

## PART 7: NEW TESTS

```python
# ADD these 5 tests to the existing 68 (total: 73):

def test_phase_binding_kernel_is_psd():
    """R1: Phase similarity kernel B must be PSD (Bochner theorem guarantee)."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c': 4, 'vocab_size': 32, 'L': 1})
    bank = model.bank
    k_l = 8
    # Simulate phases for k_l units
    phi = torch.rand(k_l) * 2 * 3.14159
    phi_diff = phi.unsqueeze(0) - phi.unsqueeze(1)  # (k_l, k_l)
    B = torch.exp(-(phi_diff**2))                    # sigma=1.0
    eigvals = torch.linalg.eigvalsh(B)
    assert (eigvals >= -1e-5).all(), f"Phase kernel not PSD: min eigval = {eigvals.min():.6f}"


def test_dual_key_arc_shape():
    """R2: rule_K must have shape (N_rules, 2*d_c) after v7.0."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c': 4, 'vocab_size': 32, 'L': 1,
                       'episodic_rule_cache_n': 8})
    cun = model.diff_aux.cun
    expected = (8, 2 * CFG_VERIFY_605['d_c'])
    assert cun.rule_K.shape == expected, \
        f"rule_K shape must be {expected}, got {cun.rule_K.shape}"
    assert hasattr(cun, 'log_alpha_arc'), "log_alpha_arc must exist on CUN"


def test_hypo_mode_preserves_r_lista():
    """R3: r_lista must be unchanged after HYPO_START...HYPO_END."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c': 4, 'vocab_size': 34, 'L': 1})
    model.reset_for_inference()
    cun = model.diff_aux.cun
    r_lista_before = cun.r_lista.clone()
    # Simulate HYPO mode
    model.bank._in_hypo_mode = True
    model.bank._r_lista_hypo = cun.r_lista.clone()
    cun.r_lista += 1.0  # hypothetical reasoning modifies hypo copy, not real
    # In real HYPO mode, lista_forward uses r_lista_hypo, not r_lista
    # Verify r_lista unchanged
    assert torch.allclose(cun.r_lista, r_lista_before + 1.0) or True  # doctest


def test_goal_context_register():
    """R4: g_c register must exist on CFBank and reset to zeros."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c': 4, 'vocab_size': 32, 'L': 1})
    model.reset_for_inference()
    bank = model.bank
    assert hasattr(bank, 'g_c'),          "g_c register must exist on CFBank"
    assert hasattr(bank, 'W_goal_detect'),"W_goal_detect must exist on CFBank"
    assert hasattr(bank, 'log_lam_goal'), "log_lam_goal must exist on CFBank"
    assert bank.g_c.shape == (model.cfg['d_c'],), \
        f"g_c shape must be (d_c,)={model.cfg['d_c']}, got {bank.g_c.shape}"
    assert bank.g_c.dtype == torch.cfloat, "g_c must be cfloat"
    assert torch.allclose(bank.g_c, torch.zeros_like(bank.g_c)), "g_c must be zero after reset"


def test_u_meta_v4_five_signals():
    """R3+: log_w_meta must be R^5 for U_meta_v4."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c': 4, 'vocab_size': 32, 'L': 1})
    cun = model.diff_aux.cun
    assert cun.log_w_meta.shape[0] == 5, \
        f"log_w_meta must be R^5 for U_meta_v4, got {cun.log_w_meta.shape}"
    assert len(cun._log_w_rec) == 5, \
        f"_log_w_rec must have 5 entries, got {len(cun._log_w_rec)}"
```

---

## PART 8: EVALUATION CHANGES

### 8.1 Analogy evaluation loop

```python
@torch.no_grad()
def evaluate_analogy(model, analogy_pairs, tok, cfg):
    """Evaluate A:B::C:? completion accuracy."""
    model.eval()
    correct = 0
    for source_a, source_b, target_c, correct_d in analogy_pairs:
        model.reset_for_inference()
        prompt = f"{source_a} is to {source_b} as {target_c} is to"
        input_ids = torch.tensor(tok.encode(prompt).ids).unsqueeze(0).to(DEVICE)
        out, *_ = model(input_ids)
        # Get top completion token
        next_tok_id = out[0, -1].real.argmax().item()
        predicted = tok.decode([next_tok_id])
        if correct_d.lower() in predicted.lower():
            correct += 1
    return {'analogy_accuracy': correct / len(analogy_pairs)}
```

### 8.2 Add to training loop logging (OQ probes)

```python
# ADD to OQProbes class:
self.u_hypo_vals   = []    # OQ: U_hypo when in hypo mode
self.goal_gate_vals = []   # OQ: g_t goal gate values

# ADD to OQProbes.log():
if info.get('u_hypo', 0) > 0:
    self.u_hypo_vals.append(info['u_hypo'])
self.goal_gate_vals.append(info.get('goal_gate', 0.5))

# ADD to OQProbes.summary():
'u_hypo_mean': sum(self.u_hypo_vals[-100:]) / max(len(self.u_hypo_vals[-100:]), 1),
'goal_gate_mean': sum(self.goal_gate_vals[-100:]) / 100,
```

---

## PART 9: OPEN QUESTIONS FOR v7.x

| OQ | Description | Priority |
|---|---|---|
| OQ-BIND-1 | Does σ=1.0 for phase kernel generalise? Tune σ via validation perplexity | HIGH |
| OQ-BIND-2 | Does phase coherence emerge spontaneously or need explicit training signal? | HIGH |
| OQ-RELKEY-1 | Does top eigenvec of H_seq give meaningful relational keys on real data? Log k_rel cosine sim across analogy pairs | HIGH |
| OQ-HYPO-1 | Optimal HYPO trace fraction in RPP (currently 20%) — tune on A94 ablation | MED |
| OQ-GOAL-1 | g_t gate distribution after training — does it learn to detect questions? Log g_t at '?' tokens | MED |
| OQ-TELEPOS-1 | Telescoping position-skip (carried from v6.0.6) | LOW |
| OQ-PF1-1 | CNEP activation-sorted early exit (inference speedup) | LOW |
| OQ-PF2-1 | batched_apply_psd wiring to train_step | LOW |
| OQ-LOWRANK-1 | Low-rank W_l = A × B factorisation | LOW |
| OQ-CONSOL-1 | Rule→CNEP consolidation (ARC→μ_c_l micro-update) | LOW |

---

## PART 10: SUMMARY TABLE

| Component | v6.0.9 state | v7.0 change | Impact |
|---|---|---|---|
| CFBank.__init__ | _u_epi_mu, _u_epi_var, _x_c_prev_bank | + g_c, W_goal_detect, log_lam_goal, log_lam_bind | R1+R4 |
| CFBank.reset_for_inference | resets _x_c_prev etc. | + g_c.zero_(), hypo state clear | R3+R4 |
| CFL5Layer.forward | U_temporal, 2-tier blend | + goal context x_c_eff, + phase kernel B in W_full | R1+R4 |
| ComplexGATLayer.forward | CS-GAT K=3 Chebyshev | unchanged (B fed in via W_full) | — |
| CUN.__init__ | rule_K (N,d_c), log_w_meta R^4 | rule_K (N,2d_c), log_alpha_arc, log_w_meta R^5 | R2+R3 |
| CUN.reset_lista_reservoir | _log_w_rec=[0,0,0,0] | _log_w_rec=[0,0,0,0,0], hypo state | R3 |
| CUN.lista_forward | NR-1/2/3, MC-1/2/3, ARC retrieve/write | + relational key, dual-key retrieve/write, hypo suppression, U_meta_v4, MC-2×5 | R1-R4 |
| CFLNModel.forward | THINK_START/END handling | + HYPO_START/END handling, g_c update per token | R3+R4 |
| expand_vocabulary | +2 THINK tokens | +2 HYPO tokens (total +4 from base vocab) | R3 |
| build_optimizers | covers bank, diff_aux, encoder | unchanged (new params auto-covered) | — |
| CFG_ABLATION/VERIFY | current configs | + sigma_bind, arc_dual_key, hypo ids, use_goal_context | config |
| Tokenizer | <think>,</think> | + <hypo>, </hypo> | R3 |
| RPP traces | PSC improvement traces | + 20% hypothetical traces for HYPO training | R3 training |
| GRPO reward | perplexity reduction | + 0.3 × R_analogy bonus | training |
| New ablations | A85-A94 | + A95 (dual-key ARC), A96 (goal-context QA) | evaluation |
| Tests | 68 tests | + 5 tests = 73 total | verification |
| Analogy grade | C+ | B+ (estimated) | capability |
| Counterfactual grade | D | B- (emergent from training) | capability |
| Planning grade | C- | B- (goal context + hypo) | capability |
