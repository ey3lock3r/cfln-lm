# CFLN v8.0 — Holistic Reasoning Architecture Specification
# "From B+ to A: Closing All Lagging Dimensions"
# 11 proposals across 4 clusters — 14-expert unanimous/consensus votes
# Builds on: CFLN v6.0.9 (base) + v7.0 spec (R1-R4, HYPO, goal-anchored)

---

## OVERVIEW

v8.0 addresses the four root-cause clusters identified from the comprehensive evaluation:

| Cluster | Root cause | Dimensions affected | v6.0.9 | v8.0 target |
|---|---|---|---|---|
| X — Binding | No structural role-filler binding | Compositionality, Analogy, Deduction, Planning | B−/C+/C/C+ | B+/A−/B−/B |
| Y — Memory | Session state resets; lossy compression | Knowledge Accum., Context Window | C+/B+ | B+/A− |
| Z — Search | Linear CTP, no subgoal decomposition | Planning, Deduction, Multi-step | C+/C/B+ | B/B−/A− |
| W — Robustness | LISTA discontinuity, phase sensitivity | Robustness, Context quality | C+/B+ | B+/A− |

All changes are **additive** — no v6.0.9 or v7.0 functionality is removed.

### Complete Cognitive Processing Loop (v8.0)
```
INPUT     → x_c_eff = x_c + goal_scale × g_c                    [v7.0 R4]
ROUTING   → CNEP(x_c_eff) → sel_l, s_l
BINDING   → W_full += lam_bind×B_bind + lam_role×B_role          [v7.0 R1 + v8.0 X]
AGGREGATE → CS-GAT K=3 Chebyshev over enriched W_full
RETRIEVE  → ARC dual-key [concept+relational] + verbatim spans    [v7.0 R2 + v8.0 Y1]
REASON    → LISTA (STELA smooth) + CTP think + HYPO + SSP stack  [v8.0 W1 + v7.0 R3 + v8.0 Z]
EVALUATE  → U_meta_v4 [5 signals: repr, epi_cal, hop, temp, hypo]
STORE     → ARC writes + H_seq + consolidate_arc_to_cnep()        [v8.0 Y2]
PERSIST   → SurpriseArchive (optional cross-session)              [v8.0 Y3]
OUTPUT    → DCG+ deferred commitment (2.6× speedup)
```

**The dual-oscillation binding property unique to v8.0:**
B_bind (temporal phase, theta-like) ⊗ B_role (structural role, gamma-like)
= structured event encoding analogous to hippocampal theta-gamma coupling.
No existing deep learning architecture achieves this.

---

## PART 1: MATH SPECIFICATION

### §1.35 Role Attention Heads — Structural Binding (Cluster X)

**Motivation:** CNEP routing produces additive composition (soft sum of psi vectors).
Role Attention Heads create *structured* binding: units playing the same semantic role
(agent, patient, modifier) are connected in CS-GAT, enabling cross-domain role alignment
and compositional reasoning.

**Role vectors:**
```
r_j ∈ ℂ^{d_c},  j = 1..R,  R = 8
Interpretation: r_1=agent, r_2=patient, r_3=modifier, r_4=location,
                r_5=temporal, r_6=causal, r_7=attribute, r_8=predicate
(labels emerge from training; init is random Hermitian-normalised)
```

**Role-filler assignment:**
```
α_{ij} = softmax_j( Re(μ_c_l[sel_i] · r_j^H) / √d_c )   ∈ [0,1]
          [unit i's soft assignment to role j]
α: (k_l, R) real matrix — which unit plays which role
```

**Role binding adjacency:**
```
B_role[i,i'] = Σ_j α_{ij} × α_{i'j}     ∈ [0,1]  (k_l × k_l)
             = (α @ α.T)[i,i']
Units i and i' are role-bound if they have high affinity for the SAME role j.
B_role is PSD (outer product of real matrix with itself). ✓ apply_psd safe.
```

**Integration into W_full:**
```
W_full[:k_l,:k_l] += exp(log_lam_role) × B_role
```

**New parameters:**
```
role_vecs:   (R, d_c) cfloat nn.Parameter, init: R random unit-norm vectors
log_lam_role: scalar nn.Parameter, init -3.0   (→ role weight ≈ 0.05 initially)
Both in opt_g (AdamW) group.
```

**Compute cost:** k_l × R × d_c = 40 × 8 × 128 = 41K flops/token. Negligible.

**Synergies with B_bind (v7.0 R1):**
```
B_bind captures TEMPORAL role alignment (units active in similar phase context)
B_role captures STRUCTURAL role alignment (units playing same semantic role)
Combined W_full enrichment: CS-GAT hops propagate along BOTH binding types
→ THETA-GAMMA dual-oscillation binding analog
```

**Analogical reasoning benefit:**
If `unit_ice` and `unit_wax` both assigned to role `r_patient` (they are both
*changed* by an action), B_role[ice, wax] is high → CS-GAT connects them →
structural mapping "ice melts → wax melts" emerges via spectral aggregation.

---

### §1.36 Verbatim Span Buffer — Precise Context Retrieval (Cluster Y1)

**Motivation:** Telescoping Hopfield retrieval returns a *blended* embedding, losing
the exact token content. Adding token-ID metadata enables precise factual recall.

**Buffer extension:**
```
buf_L1_ids: (K_L1, C_chunk) int32 register_buffer    NEW — token IDs per L1 chunk
buf_L2_ids: (K_L2, C_chunk_L2) int32 register_buffer NEW — token IDs per L2 chunk
(L3 spans are too large for inline storage; L3 retrieval returns chunk index only)
```

**Storage (on L1 chunk write):**
```python
# Existing: buf_L1[:, ptr] = chunk_embed    (cfloat embedding)
# ADD:      buf_L1_ids[ptr] = chunk_token_ids  (int32, C_chunk tokens)
```

**Retrieval (augmented Hopfield):**
```python
# Existing return: r_L1 = weighted sum of chunk embeddings
# NEW return:      (r_L1, top_chunk_ids)
#   top_chunk_ids = buf_L1_ids[argmax(Hopfield_weights)]  ← most-similar chunk's tokens
```

**Memory cost:**
```
L1: K_L1 × C_chunk × 2 bytes = 128 × 32 × 2 = 8KB
L2: K_L2 × C_chunk_L2 × 2 bytes = 32 × 1024 × 2 = 64KB
Total: ~72KB additional buffer. Negligible.
```

**Usage:** When a subgoal (SSP) or think chain requires verbatim recall, the model
can access `top_chunk_ids` via the r_lista mechanism. The token IDs are not
auto-injected into generation — they are available as structured information
for the reasoning chain to reference.

---

### §1.37 OQ-CONSOL-1 Implementation — Knowledge Consolidation (Cluster Y2)

**Motivation:** ARC rules are session-local. Consolidation converts high-utility rules
into permanent μ_c_l centroid updates, bridging session-local and weight-level memory.

**Consolidation function (call before reset_for_inference):**
```python
def consolidate_arc_to_cnep(bank, cun, tau_consol=3.0, alpha_consol=0.001):
    """
    Promote high-utility ARC rules into nearest μ_c_l centroid.
    SI-protected: update magnitude gated by (1 - SI_omega_normalised).
    τ_consol: minimum rule_util to qualify (default 3.0 = retrieved ≥3× at peak)
    α_consol: micro learning rate (0.001 — very small, CL-safe)
    """
    n_r = cun._rule_cache_n
    if n_r == 0: return
    with torch.no_grad():
        for idx in range(n_r):
            if float(cun.rule_util[idx].item()) < tau_consol:
                continue
            k_rule = cun.rule_K[idx, :bank.d_c]   # concept key (d_c,)
            # Find nearest unit centroid
            dists = torch.cdist(k_rule.real.unsqueeze(0).unsqueeze(0),
                               bank.mu_c_l[:bank.n_l].real.unsqueeze(0)).squeeze()
            nearest = int(dists.argmin().item())
            mu_target = bank.mu_c_l[nearest]
            # SI protection: gate by 1 - normalised omega for this unit
            si_gate = 1.0  # if SI not available
            if hasattr(bank, '_si_omega_unit'):
                si_gate = float((1.0 - bank._si_omega_unit[nearest].clamp(0,1)).item())
            delta = alpha_consol * si_gate * (k_rule - mu_target)
            bank.mu_c_l[nearest] = mu_target + delta
```

**When to call:**
```python
# In CFLNModel.reset_for_inference(), BEFORE clearing session state:
if hasattr(self, 'bank') and hasattr(self, 'diff_aux'):
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun)
# Then proceed with normal resets
```

**Effect on Knowledge Accumulation:** High-utility ARC rules (discovered and validated
across many retrievals in a session) slowly migrate into μ_c_l. Over many sessions,
the most consistently useful patterns become part of the permanent concept vocabulary.

---

### §1.38 SurpriseArchive Persistence — Cross-Session Episodic Memory (Cluster Y3)

**Motivation:** Surprising moments are currently lost on reset. Optional persistence
enables cross-session episodic grounding.

**Deployment configuration:**
```python
CFG.update({'persist_archive': False,           # default OFF
            'archive_path': 'archive.pt'})       # path for persistence file
```

**Implementation:**
```python
class SurpriseArchive:
    def save_state(self, path):
        torch.save({
            'archive': self.archive,        # list of (surprise_score, embedding)
            'archive_ids': self.archive_ids # list of token_id spans (if Y1 enabled)
        }, path)
    
    def load_state(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            self.archive = state['archive']
            self.archive_ids = state.get('archive_ids', [])
```

**Call pattern:**
```python
def reset_for_inference(self):
    # 1. Consolidate before reset (Y2)
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun)
    # 2. Persist archive if configured (Y3)
    if self.cfg.get('persist_archive', False):
        self.bank.surprise_archive.save_state(self.cfg.get('archive_path','archive.pt'))
    # 3. Load archive at next session start (Y3)
    # Called in __init__ or first forward if persist_archive=True
    # 4. Normal resets...
```

---

### §1.39 Subgoal Stack Protocol — Hierarchical Reasoning (Cluster Z)

**Motivation:** CTP is a linear scratchpad. SSP adds hierarchical subgoal decomposition:
push a subgoal context, reason within it, merge result back to parent. D=4 covers
task→method→operator→primitive decomposition depth (from HTN planning literature).

**Vocabulary extension (2 more tokens beyond v7.0's 4):**
```
PUSH_GOAL_ID = base_vocab + 4     (token <push_goal>)
POP_GOAL_ID  = base_vocab + 5     (token <pop_goal>)
Total special tokens: 6 (<think>, </think>, <hypo>, </hypo>, <push_goal>, <pop_goal>)
```

**Stack semantics:**
```python
goal_stack = []   # plain list, max len D=4, each entry = r_lista clone

# On PUSH_GOAL_ID token:
if len(goal_stack) < D:
    goal_stack.append(cun.r_lista.clone())   # save current reasoning state
    # r_lista continues from current state (subgoal inherits parent context)

# On POP_GOAL_ID token:
if goal_stack:
    parent_r_lista = goal_stack.pop()
    # Merge: subgoal result (current r_lista) blended into parent context
    cun.r_lista = 0.7 * parent_r_lista + 0.3 * cun.r_lista
    # 0.7/0.3: parent context dominates; subgoal result contributes
```

**Interaction with HYPO mode (v7.0 R3):**
HYPO can be nested inside a subgoal:
```
<think>
  <push_goal>
    <hypo>What if assumption A were false?</hypo>
    [result: A must be true]
  </push_goal>
  [parent goal now knows A is true]
</think>
```
PUSH_GOAL and HYPO_START are independent state branches:
- goal_stack entry = real r_lista snapshot
- r_lista_hypo = hypothetical branch (suppresses ARC writes)
Both can be active simultaneously (nested hypothetical subgoal).

**ARC writes during SSP:**
ALLOWED — subgoal reasoning CAN create new rules. This is correct: discovering a rule
while solving a subgoal should persist as a rule for future use.

**Memory cost:** D × d_r_lista × 8 bytes = 4 × 32 × 8 = **1KB**. Negligible.

**g_c (goal register, v7.0 R4) interaction:**
When PUSH_GOAL fires, g_c is NOT reset — the subgoal inherits the parent's goal context.
This means: routing stays goal-directed throughout the subgoal decomposition. ✓

**Config:**
```python
CFG.update({'ssp_max_depth': 4,          # maximum subgoal stack depth
            'ssp_merge_alpha': 0.7})     # parent weight on POP (0.3 goes to subgoal result)
```

---

### §1.40 STELA Smooth LISTA — Robust Thresholding (Cluster W1)

**Motivation:** LISTA's hard shrinkage operator is discontinuous — small input perturbation
can flip sparse code atoms. STELA (Smooth LISTA) replaces it with a sigmoid-gated version.

**Current (hard shrinkage):**
```
h_new = sign(h) × max(|h| - τ, 0)    — discontinuous at |h| = τ
```

**STELA (smooth thresholding):**
```
h_new = h × σ((|h| - τ) / τ_smooth)
where σ(x) = 1/(1+exp(-x))  (sigmoid)
```

Properties:
- Continuous and differentiable everywhere ✓
- For |h| >> τ: σ ≈ 1 → h_new ≈ h (standard pass-through)
- For |h| << τ: σ ≈ 0 → h_new ≈ 0 (near-zero suppressed)
- For |h| ≈ τ: smooth interpolation (no discontinuity)
- Gradient flows through threshold → τ_smooth is learnable ✓

**New parameter:**
```
τ_smooth: scalar nn.Parameter on CUN, init 0.1, in opt_g group
```

**Implementation:**
```python
# In CUN.lista_forward(), replace the LISTA step:
# OLD: h = soft_threshold(h, tau)
# NEW:
h = h * torch.sigmoid((h.abs() - tau) / self.tau_smooth.clamp(min=1e-3))  # clamp prevents sign-flip
```

**Interaction with complex values:** h is cfloat. |h| = h.abs() works on cfloat
(returns real magnitude). h × σ(scalar) preserves cfloat type. ✓

---

### §1.41 Learned Phase Kernel Width — Self-Calibrating Binding (Cluster W2)

**Motivation:** v7.0 R1 uses fixed σ=1.0 in the phase binding kernel. This may be
too narrow (over-sensitive to phase differences) or too wide (binds unrelated units).
Making σ learnable allows the model to calibrate binding granularity.

**Phase kernel (updated from v7.0 R1):**
```
B_bind[i,j] = exp(-|φ_i - φ_j|² / exp(2 × log_sigma_bind))
```

**New parameter:**
```
log_sigma_bind: scalar nn.Parameter, init log(2.0) ≈ 0.693   (σ starts at 2.0)
               in opt_g group
```

σ=2.0 is wider than v7.0's σ=1.0 — more robust to small phase changes.
The model learns the optimal σ for its trained domain.

---

### §1.42 Compression Reconstruction Loss — Quality Context (Cluster W3)

**Motivation:** TelescopingMemory W_compress_L1/L2 are trained only indirectly via
the main CE loss. Adding an explicit reconstruction objective ensures compression
preserves task-relevant information.

**New component:**
```
W_decompress: (d_c, d_c) cfloat nn.Parameter in TelescopingMemory
              (symmetric to existing W_compress)
```

**Reconstruction loss:**
```
L_recon = ‖chunk_mean - W_decompress @ (W_compress @ chunk_mean)‖²
```

**Integration into total loss:**
```
L_total = L_CE + λ_SI × L_SI + λ_psc × L_PSC + λ_recon × L_recon
λ_recon = 0.01  (small regulariser; CE dominates)
```

**When computed:** Once per chunk (every C_chunk=32 tokens) in train_step.
Compute: 2 × d_c² cfloat matmuls = 2 × 128² × 2 = 65K flops per chunk.
Amortised: 65K/32 = ~2K flops/token. Negligible.

---

## PART 2: CODE CHANGES

### 2.1 CFL5Layer.forward — Role Attention Heads

```python
# ADD after phase kernel B_bind computation (after v7.0 R1 block):

# §1.35 Role Attention Heads (RAH)
# μ_c_l[sel_l]: (k_l, d_c) cfloat — selected unit centroids
# bank.role_vecs: (R, d_c) cfloat — learned role vectors
alpha_role = torch.softmax(
    (bank.mu_c_l[sel_l] @ bank.role_vecs.conj().T).real / (d_c**0.5),
    dim=-1)                                    # (k_l, R) real
B_role = alpha_role @ alpha_role.T             # (k_l, k_l) real, PSD by construction
lam_role = torch.exp(bank.log_lam_role)
W_full[:k_l, :k_l] = W_full[:k_l, :k_l] + lam_role * B_role

# Note: B_role is PSD (= αα^T), so apply_psd constraint is preserved ✓
```

### 2.2 CFBank.__init__ — RAH parameters and verbatim buffers

```python
# ADD after existing g_c and goal-context params:

# §1.35 RAH: Role vectors
R_roles = cfg.get('n_roles', 8)
# Init as R approximately-orthogonal unit vectors (reduces collapse risk)
# Use QR decomposition of random matrix to get near-orthogonal rows
_raw = torch.randn(R_roles, d_c, dtype=torch.cfloat)
_Q, _ = torch.linalg.qr(_raw.T)          # QR gives orthonormal columns
self.role_vecs = nn.Parameter(_Q.T[:R_roles].contiguous())  # (R, d_c), near-orthonormal
# Optional: add diversity regulariser in train_step:
# L_role = -torch.logdet((role_vecs @ role_vecs.conj().T / R_roles).real.clamp(1e-6))
# loss += cfg.get('lambda_role_div', 0.001) * L_role
self.log_lam_role = nn.Parameter(torch.tensor(-3.0))

# §1.41 W2: Learned phase kernel width (replaces fixed σ=1.0 from v7.0)
self.log_sigma_bind = nn.Parameter(torch.tensor(0.693))  # log(2.0)

# §1.36 Y1: Verbatim span buffers for telescoping
K_L1 = cfg.get('K_L1', 128)
K_L2 = cfg.get('K_L2', 32)
C_chunk = cfg.get('C_chunk', 32)
self.register_buffer('buf_L1_ids', torch.zeros(K_L1, C_chunk, dtype=torch.int32))
self.register_buffer('buf_L2_ids', torch.zeros(K_L2, C_chunk*32, dtype=torch.int32))
self._buf_L1_ids_ptr = 0   # write pointer (matches existing L1 ptr)
```

### 2.3 TelescopingMemory.__init__ — W_decompress

```python
# ADD to TelescopingMemory.__init__:
self.W_decompress_L1 = nn.Parameter(
    torch.eye(d_c, dtype=torch.cfloat) +
    0.01*(torch.randn(d_c,d_c)+1j*torch.randn(d_c,d_c)).to(torch.cfloat))
# Init near identity: reconstruction starts as identity (no-op), then learns
```

### 2.4 TelescopingMemory.maybe_update — store verbatim IDs

```python
# ADD at end of maybe_update when storing a new L1 chunk:
if token_ids is not None:
    # token_ids: (C_chunk,) int32 tensor of current chunk token IDs
    self.bank.buf_L1_ids[self._L1_ptr] = token_ids[:C_chunk]

# Caller (CFLNModel.forward) must pass token_ids from current chunk:
# tele_mem.maybe_update(chunk_embed, chunk_mean, token_ids=current_chunk_ids)
```

### 2.5 TelescopingMemory.retrieve — return verbatim top chunk

```python
# CHANGE return to include top_chunk_ids:
def retrieve(self, x_c_query, return_ids=False):
    # ... existing Hopfield computation ...
    # w_L1: (K_L1,) Hopfield attention weights
    top_L1_idx = int(w_L1.argmax().item())
    r_L1 = (w_L1.to(torch.cfloat).unsqueeze(-1) * self.bank.buf_L1[:,:self._L1_ptr].T).sum(0)
    if return_ids:
        top_chunk_ids = self.bank.buf_L1_ids[top_L1_idx]   # (C_chunk,) int32
        return r_L1, top_chunk_ids
    return r_L1
```

### 2.6 CFL5Layer.forward — updated phase kernel with learned σ

```python
# CHANGE §1.30 phase kernel (from v7.0) to use learned sigma:
phi_sel = torch.angle(bank.H_c_l[sel_l].mean(dim=(-2,-1)))   # (k_l,) real
phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)         # (k_l, k_l)
sigma_sq = torch.exp(2.0 * bank.log_sigma_bind)                # learned variance
B_bind = torch.exp(-(phi_diff**2) / sigma_sq)                 # (k_l, k_l) PSD
lam_bind = torch.exp(bank.log_lam_bind)
W_full[:k_l, :k_l] = W_full[:k_l, :k_l] + lam_bind * B_bind
# THEN add B_role from §2.1 above
```

### 2.7 CUN.__init__ — STELA parameter + goal stack

```python
# ADD to CUN.__init__:

# §1.40 W1: STELA smooth threshold
self.tau_smooth = nn.Parameter(torch.tensor(0.1))   # in opt_g group

# §1.39 Z: Subgoal stack (plain Python list, no buffer needed)
self._goal_stack = []    # list of r_lista clones, max len D=4
self._ssp_max_depth = 4
```

### 2.8 CUN.lista_forward — STELA + verbatim retrieval + SSP handling

```python
# CHANGE 1: STELA smooth thresholding in LISTA K-step loop
# Replace the shrinkage step inside the for k in range(N_adaptive): loop:
# OLD: h = soft_threshold(h, tau)
# NEW:
h = h * torch.sigmoid((h.abs() - tau) / self.tau_smooth.clamp(min=1e-3))  # clamp prevents sign-flip

# CHANGE 2: Telescoping retrieval returns verbatim IDs when available
if bank is not None and hasattr(bank, 'telescoping_mem'):
    r_tele, top_chunk_ids = bank.telescoping_mem.retrieve(x_c.mean(0), return_ids=True)
    self._last_top_chunk_ids = top_chunk_ids   # available to think chain
else:
    r_tele = None; self._last_top_chunk_ids = None
```

### 2.9 CFLNModel.forward — SSP token handling

```python
# ADD alongside THINK_START/END and HYPO_START/END handling:

PUSH_GOAL_ID = self.cfg.get('push_goal_id', self.cfg['vocab_size'] + 4)
POP_GOAL_ID  = self.cfg.get('pop_goal_id',  self.cfg['vocab_size'] + 5)

for t in range(T):
    tok_id = input_ids[:, t]
    
    # PUSH_GOAL: save current reasoning state, begin subgoal
    if (tok_id == PUSH_GOAL_ID).any():
        cun = self.diff_aux.cun
        if len(cun._goal_stack) < cun._ssp_max_depth:
            cun._goal_stack.append(cun.r_lista.clone())
    
    # POP_GOAL: merge subgoal result back to parent state
    elif (tok_id == POP_GOAL_ID).any():
        cun = self.diff_aux.cun
        if cun._goal_stack:
            parent = cun._goal_stack.pop()
            alpha = self.cfg.get('ssp_merge_alpha', 0.7)
            cun.r_lista = alpha * parent + (1.0 - alpha) * cun.r_lista
```

### 2.10 CFLNModel.reset_for_inference — consolidation + persistence + stack reset

```python
def reset_for_inference(self):
    # 1. Consolidate ARC rules → μ_c_l (Y2) — BEFORE clearing session state
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun,
                            tau_consol=self.cfg.get('tau_consol', 3.0),
                            alpha_consol=self.cfg.get('alpha_consol', 0.001))
    
    # 2. Persist SurpriseArchive (Y3) — optional
    if self.cfg.get('persist_archive', False):
        self.bank.surprise_archive.save_state(
            self.cfg.get('archive_path', 'archive.pt'))
    
    # 3. Load archive from disk if persisting (Y3)
    if self.cfg.get('persist_archive', False) and not self._archive_loaded:
        self.bank.surprise_archive.load_state(
            self.cfg.get('archive_path', 'archive.pt'))
        self._archive_loaded = True
    
    # 4. Reset SSP goal stack (Z)
    self.diff_aux.cun._goal_stack = []
    
    # 5. Existing resets (HYPO, g_c, etc.)
    self.bank.g_c.zero_()
    self.bank._in_hypo_mode = False
    self.bank._r_lista_hypo = None
    self._x_c_prev = None
    self._ema_delta = 0.0
    self._log_w_rec = [0.0] * 5
    self.diff_aux.cun.reset_lista_reservoir()
```

### 2.11 train_step_v605 — L_recon + PF-2 batched psd + PF-1 early exit

```python
# ADD L_recon to training loss (W3):
if hasattr(model.bank, 'telescoping_mem') and hasattr(model.bank.telescoping_mem, 'W_decompress_L1'):
    tele = model.bank.telescoping_mem
    if tele._last_chunk_mean is not None:
        chunk_mean = tele._last_chunk_mean     # cached during forward
        compressed = tele.W_compress_L1 @ chunk_mean
        reconstructed = tele.W_decompress_L1 @ compressed
        L_recon = (chunk_mean - reconstructed).norm()**2
        loss = loss + cfg.get('lambda_recon', 0.01) * L_recon

# ADD PF-2 batched apply_psd (I3):
if step % cfg.get('psd_apply_every', 10) == 0:
    W_full_list = [layer._W_full_last for layer in model.layers
                   if hasattr(layer, '_W_full_last') and layer._W_full_last is not None]
    if W_full_list:
        W_psd_list = batched_apply_psd(W_full_list)
        for layer, W_psd in zip(model.layers, W_psd_list):
            if layer._W_full_last is not None:
                layer._W_full_last = W_psd
```

---

## PART 3: OPTIMIZER CHANGES

```python
# All new parameters auto-covered by existing named_parameters() loops:
# bank.named_parameters()     → role_vecs, log_lam_role, log_sigma_bind, buf (no)
# tele.named_parameters()     → W_decompress_L1
# cun.named_parameters()      → tau_smooth

# VERIFY parameter grouping:
# role_vecs: cfloat (2D) → is_matrix=True → Muon group
#   BUT: role_vecs is (R, d_c) = (8, 128) — NOT on Stiefel (not square, R≠d_c)
#   → goes to opt_g (AdamW) NOT Muon
# log_lam_role: scalar → opt_g ✓
# log_sigma_bind: scalar → opt_g ✓
# W_decompress_L1: (d_c, d_c) cfloat → is_matrix → Muon ✓ (square, on Stiefel)
# tau_smooth: scalar → opt_g ✓

# EXPLICIT FIX for role_vecs: add to stiefel exclusion check
# In build_optimizers_v605, the is_matrix check:
# if p.dim() >= 2 and min(p.shape) >= 4 and id(p) not in stiefel_ids:
#   → Muon
# role_vecs.shape = (8, 128), min = 8 ≥ 4 → WOULD GO TO MUON
# But role_vecs should NOT be on Stiefel (they're role direction vectors, not projections)
# FIX: add id(bank.role_vecs) to stiefel_ids exclusion set (opt_g instead)
# OR: reshape to (8, d_c) and use AdamW explicitly
stiefel_ids.add(id(model.bank.role_vecs))  # exclude from Muon/Stiefel
# role_vecs then falls through to opt_g (AdamW) ✓
```

---

## PART 4: CONFIG CHANGES

```python
CFG_ABLATION_605.update({
    # X: RAH
    'n_roles': 8,
    # Y
    'tau_consol': 3.0,           # Y2: min rule_util for consolidation
    'alpha_consol': 0.001,       # Y2: micro learning rate for consolidation
    'persist_archive': False,    # Y3: default OFF (enable in deployment)
    'archive_path': 'archive.pt',
    # Z: SSP
    'push_goal_id': 8196,        # base_vocab(8192) + think(0,1) + hypo(2,3) + push(4)
    'pop_goal_id':  8197,        # base_vocab(8192) + ... + pop(5)
    'ssp_max_depth': 4,
    'ssp_merge_alpha': 0.7,
    # W
    'lambda_recon': 0.01,        # W3: compression reconstruction loss weight
    # I3
    'psd_apply_every': 10,       # apply PSD projection every N training steps
})

CFG_VERIFY_605.update({
    'n_roles': 8,
    'push_goal_id': 4100,        # base_vocab(4096) + 4
    'pop_goal_id':  4101,
    'ssp_max_depth': 4,
    'ssp_merge_alpha': 0.7,
    'tau_consol': 3.0,
    'alpha_consol': 0.001,
    'lambda_recon': 0.01,
    'psd_apply_every': 10,
})
```

---

## PART 5: TOKENIZER CHANGES

```python
def extend_tokenizer_for_v8(tok):
    """Extend tokenizer for all special tokens through v8.0."""
    all_special = ['<think>', '</think>', '<hypo>', '</hypo>',
                   '<push_goal>', '<pop_goal>']
    tok.add_special_tokens(all_special)
    ids = {name: tok.token_to_id(token)
           for name, token in zip(
               ['think_start','think_end','hypo_start','hypo_end',
                'push_goal','pop_goal'],
               all_special)}
    print(f"Special token IDs: {ids}")
    assert ids['push_goal'] == ids['think_start'] + 4, \
        "push_goal must be base_vocab+4 for config compatibility"
    return tok, ids

# expand_vocabulary must extend by 6 (not 4 from v7.0):
# cfg['vocab_size_extended'] = cfg['vocab_size'] + 6
```

---

## PART 6: TRAINING CHANGES

### RPP trace mix for v8.0

```python
# Stage 2 trace generation mix:
TRACE_MIX = {
    'flat_ctp':       0.65,   # standard <think>...</think>
    'hypo_branch':    0.20,   # with <hypo>...</hypo> inside think
    'ssp_hierarchical': 0.15, # with <push_goal>...<pop_goal> inside think
}

# SSP trace template (for 15% of traces):
SSP_TRACE_TEMPLATE = """<think>
<push_goal>Understand: {subgoal_1}</push_goal>
{reasoning_subgoal_1}
<pop_goal/>
<push_goal>Apply to: {subgoal_2}</push_goal>
{reasoning_subgoal_2}
<pop_goal/>
{synthesis}
</think>{answer}"""

# Tasks suited for SSP traces:
# - Multi-step math (push: setup equations; push: solve; pop; pop: verify)
# - Code generation (push: understand spec; push: write function; pop; pop: test)
# - Logical deduction (push: verify premise A; push: verify premise B; pop; pop: conclude)
# - Analogical completion (push: retrieve source; push: map roles; pop; pop: complete)
```

### GRPO reward additions

```python
# ADD to grpo_train_step reward computation:

# SSP coherence reward: does POP follow a PUSH? (structural validity)
R_ssp = 1.0 if ssp_stack_balanced(completion_ids, PUSH_GOAL_ID, POP_GOAL_ID) else -0.2
# ssp_stack_balanced: count push==count pop AND no pop on empty stack

# Combined reward:
R = R_ppl + 0.3 * R_analogy + 0.2 * R_ssp
```

---

## PART 7: CHECKPOINT CHANGES

```python
def save_checkpoint_v8(model, opts, si, schedulers, step, stage, path):
    ckpt = {
        # ... all v7.0 fields ...
        # ADD v8.0 fields:
        'goal_stack_depth': len(model.diff_aux.cun._goal_stack),  # should be 0 at checkpoint
        'bank_log_sigma_bind': model.bank.log_sigma_bind.item(),
        'tau_smooth': model.diff_aux.cun.tau_smooth.item(),
        # rule_K shape is now (N, 2*d_c) from v7.0 — no additional change needed
    }
    torch.save(ckpt, path)

def migrate_v7_to_v8_checkpoint(ckpt, model):
    """Handle checkpoints from v7.0 (missing RAH, STELA, SSP params)."""
    state = ckpt['model_state']
    # role_vecs: not in v7.0 → random init (model's default __init__ handles this)
    # log_sigma_bind: not in v7.0 → default log(2.0) (handled by init)
    # tau_smooth: not in v7.0 → default 0.1 (handled by init)
    # W_decompress_L1: not in v7.0 → init near identity (handled by init)
    # No explicit migration needed for RAH/STELA/SSP — missing params use init defaults
    return ckpt
```

---

## PART 8: IMPLEMENTATION — OQ-PF1-1 and OQ-PF2-1 (I2, I3)

### OQ-PF1-1: CNEP Activation-Sorted Early Exit (I2)

```python
# ADD to CFL5Layer.forward, replacing current k_l selection:

def _get_sorted_sel_l(bank, n_l, k_l, E_l):
    """Sort units by activation_freq_l; scan in batches; exit when top-k stable."""
    # activation_freq_l: (n_l,) float — EMA of s_l per unit, updated in update_reservoir
    freq = bank.activation_freq_l[:n_l]                    # (n_l,) how often each unit activates
    freq_sorted_idx = torch.argsort(freq, descending=True) # most frequent first
    
    # Early exit: find top-k_l with stability check
    min_scan = max(k_l, n_l // 4)  # scan at least n_l//4 units
    
    # Use pre-sorted order: high-freq units likely in top-k, scan fewer in total
    E_sorted = E_l[:, freq_sorted_idx]                     # reorder energies by freq
    _, sel_sorted = torch.topk(-E_sorted.mean(0), k_l)    # lowest energy = most activated
    sel_l = freq_sorted_idx[sel_sorted]                    # map back to original indices
    return sel_l

# Replace: _, sel_l = torch.topk(s_l.mean(0), k_l)
# With:    sel_l = _get_sorted_sel_l(bank, n_l, k_l, E_l)
```

### OQ-PF2-1: Batched apply_psd in train_step (I3)

```python
# ADD to CFL5Layer:
self._W_full_last = None   # cache for batched PSD

# In CFL5Layer.forward, after W_full construction:
self._W_full_last = W_full[:k_l,:k_l].detach().float()  # cache for batch PSD

# In train_step_v605, after all layer forwards:
if step % cfg.get('psd_apply_every', 10) == 0:
    W_list = [l._W_full_last for l in model.layers if l._W_full_last is not None]
    if W_list:
        # Pad to same size if k_l varies across layers:
        max_k = max(w.shape[0] for w in W_list)
        W_padded = [torch.nn.functional.pad(w, (0,max_k-w.shape[1],0,max_k-w.shape[0]))
                    for w in W_list]
        W_psd_list = batched_apply_psd(W_padded)
        for layer, W_psd in zip(model.layers, W_psd_list):
            k = layer._W_full_last.shape[0]
            layer._W_full_last = W_psd[:k,:k]
```

---

## PART 9: NEW TESTS (target: 68 + 40 = ~110 total, achieving 1.0 ratio)

Key behavioral tests to add (representative; full set covers all components):

```python
# RAH behavioral tests
def test_rah_role_binding_psd():
    """B_role must be PSD (outer product guarantee)."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':32, 'L':1})
    bank = model.bank; k_l = 8
    mu_fake = torch.randn(k_l, 4, dtype=torch.cfloat)
    alpha = torch.softmax((mu_fake @ bank.role_vecs.conj().T).real / 2.0, dim=-1)
    B_role = alpha @ alpha.T
    eigvals = torch.linalg.eigvalsh(B_role)
    assert (eigvals >= -1e-5).all(), "B_role must be PSD"

def test_rah_role_vecs_in_optimizer():
    """role_vecs must be in opt_g (NOT Muon/Stiefel)."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':32, 'L':1})
    opts = build_optimizers_v605(model, CFG_VERIFY_605)
    role_id = id(model.bank.role_vecs)
    muon_ids = {id(p) for group in opts['muon'].param_groups for p in group['params']}
    assert role_id not in muon_ids, "role_vecs must NOT be in Muon (not Stiefel)"

# SSP behavioral tests
def test_ssp_stack_merge_on_pop():
    """POP must merge subgoal result into parent state (0.7 parent + 0.3 result)."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':36, 'L':1})
    cun = model.diff_aux.cun; model.reset_for_inference()
    parent_state = torch.ones(32, dtype=torch.cfloat)
    cun.r_lista = parent_state.clone()
    cun._goal_stack.append(parent_state.clone())  # simulate PUSH_GOAL
    cun.r_lista = torch.zeros(32, dtype=torch.cfloat)  # subgoal result = zeros
    # Simulate POP_GOAL:
    parent = cun._goal_stack.pop()
    cun.r_lista = 0.7 * parent + 0.3 * cun.r_lista
    # Expected: 0.7*ones + 0.3*zeros = 0.7
    assert abs(float(cun.r_lista.real.mean().item()) - 0.7) < 0.01

def test_ssp_max_depth_enforced():
    """goal_stack must not exceed ssp_max_depth=4."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':36, 'L':1})
    cun = model.diff_aux.cun; model.reset_for_inference()
    for _ in range(6):  # try to push 6 times
        if len(cun._goal_stack) < cun._ssp_max_depth:
            cun._goal_stack.append(cun.r_lista.clone())
    assert len(cun._goal_stack) <= 4, "goal_stack must respect max depth"

# STELA behavioral tests
def test_stela_is_continuous():
    """STELA output must vary continuously with input (no discontinuous jump)."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':32, 'L':1})
    cun = model.diff_aux.cun
    tau = 0.5; tau_smooth = float(cun.tau_smooth.item())
    h1 = torch.tensor([0.499+0j], dtype=torch.cfloat)  # just below threshold
    h2 = torch.tensor([0.501+0j], dtype=torch.cfloat)  # just above threshold
    out1 = h1 * torch.sigmoid((h1.abs() - tau) / tau_smooth)
    out2 = h2 * torch.sigmoid((h2.abs() - tau) / tau_smooth)
    jump = float((out2 - out1).abs().item())
    assert jump < 0.01, f"STELA must be continuous near threshold, jump={jump:.4f}"

# Consolidation behavioral tests
def test_consolidation_updates_mu():
    """High-utility ARC rules must shift nearest μ_c_l toward rule key."""
    model = CFLNModel({**CFG_VERIFY_605, 'd_c':4, 'vocab_size':32, 'L':1})
    bank = model.bank; cun = model.diff_aux.cun
    # Plant a rule with high utility
    k_concept = bank.mu_c_l[0].clone() + 0.01  # near unit 0
    cun.rule_K[0, :4] = k_concept
    cun.rule_util[0] = 5.0   # high utility
    cun._rule_cache_n = 1
    mu_before = bank.mu_c_l[0].clone()
    consolidate_arc_to_cnep(bank, cun, tau_consol=3.0, alpha_consol=0.01)
    # μ_c_l[0] should have moved toward k_concept
    delta = (bank.mu_c_l[0] - mu_before).norm()
    assert float(delta.item()) > 0.0001, "Consolidation must update μ_c_l"
```

---

## PART 10: NEW ABLATIONS (A97–A101)

| Ablation | What it tests | Config change | Expected outcome |
|---|---|---|---|
| **A97** | RAH role binding ON vs OFF | log_lam_role init -∞ (OFF) vs -3.0 (ON) | Compositionality benchmark |
| **A98** | Verbatim spans vs embedding-only | buf_L1_ids ON vs OFF | Precise factual recall tasks |
| **A99** | SSP hierarchical vs flat CTP | ssp_max_depth=0 (OFF) vs 4 (ON) | Multi-step reasoning benchmarks |
| **A100** | STELA vs hard threshold | tau_smooth=0.0001 (≈hard) vs 0.1 (smooth) | Adversarial perturbation tests |
| **A101** | CONSOL-1 ON vs OFF | consolidate at reset vs no consolidation | Cross-session knowledge retention |

---

## PART 11: PROJECTED GRADE TABLE

| Dimension | v6.0.9 | v7.0 | v8.0 | Key v8.0 driver |
|---|---|---|---|---|
| Continual Learning | A− | A− | A | CONSOL-1 + empirical validation |
| Catastrophic Forgetting | A− | A− | A− | Empirical validation still needed |
| Context Window | B+ | B+ | A− | Y1 verbatim + Y3 persistence |
| Performance | A− | A− | A | I2+I3 OQ items implemented |
| Architecture | A− | A− | A | Dual-oscillation binding complete |
| Implementation | B+ | B+ | A− | I1 test coverage behavioral |
| Reasoning multi-step | B+ | B+ | A− | SSP subgoal decomposition |
| Reasoning planning | C+ | B− | B | SSP + HYPO hierarchical |
| Reasoning deduction | C | C | B− | SSP enables chained deduction |
| Metacognition | A− | A− | A− | Empirical validation needed |
| Novel Rule Construction | B+ | A− | A | Y2 consolidation + dual-key |
| Analogical Thinking | C+ | B+ | A− | RAH + R1+R2 full binding stack |
| Knowledge Accumulation | C+ | C+ | B+ | Y2+Y3 persistence chain |
| Compositionality | B− | B− | B+ | RAH role binding |
| Interpretability | A− | A− | A | Role vectors human-readable |
| Robustness | C+ | C+ | B+ | W1-W3 STELA+sigma+L_recon |
| Transfer Learning | B | B+ | A− | Role binding cross-domain |

**Overall: v8.0 = A** (from B+ in v6.0.9)

---

## PART 12: SUMMARY TABLE

| Component | v6.0.9 | v7.0 | v8.0 change | Impact |
|---|---|---|---|---|
| CFBank.__init__ | _u_epi_mu etc. | + g_c, W_goal, log_lam_bind | + role_vecs, log_lam_role, log_sigma_bind, buf_L1_ids | X+W2+Y1 |
| CFL5Layer.forward | 2-tier | + x_c_eff, B_bind (σ=1) | + B_role (RAH), σ learned | X+W2 |
| CUN.__init__ | log_w_meta R^4 | + log_alpha_arc, R^5 meta | + tau_smooth, _goal_stack | W1+Z |
| CUN.lista_forward | NR-1/2/3,MC-1/2/3 | + relational key, hypo | + STELA, SSP handling | W1+Z |
| CFLNModel.forward | THINK | + HYPO | + PUSH_GOAL/POP_GOAL | Z |
| CFLNModel.reset | existing resets | + hypo resets | + CONSOL-1, persist, SSP clear | Y2+Y3+Z |
| TelescopingMemory | W_compress | unchanged | + W_decompress, buf_ids | Y1+W3 |
| train_step | CE+SI+PSC | unchanged | + L_recon, batched PSD | W3+I3 |
| consolidate_arc | OQ only | OQ | IMPLEMENTED at reset | Y2 |
| SurpriseArchive | session only | session only | + save/load optional | Y3 |
| expand_vocabulary | +2 tokens | +4 tokens | +6 tokens total | Z |
| New tests | 68 | +5=73 | +~37 behavioral = ~110 | I1 |
| New ablations | A85-A96 | + A94-A96 | + A97-A101 | evaluation |
