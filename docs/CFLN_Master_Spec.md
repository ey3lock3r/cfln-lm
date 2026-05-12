# CFLN Master Specification (v9.0)

> **Single authoritative document**. All prior versioned specs (v6.0.9 through v9.0) are archived in `docs/archive/`. This file is the sole source of truth.

*Expert Panel — May 2026*

---

## OVERVIEW

CFLN (Complex-domain Full-Layer Network) is a research language model architecture combining:

- **Single complex domain**: `to_complex` at input only (`ComplexEmbedding`), `to_real` at output only (logit head). All internal tensors `torch.cfloat`.
- **CNEP** two-tier memory bank (local RQ-entmax + persistent softmax). Global tier removed v6.0.8.
- **Titans** associative memory with Wirtinger gradient updates. Self-corrects on domain shift.
- **LISTA** sparse working memory with session reservoir. STELA smooth thresholding (v8.0).
- **CRoPE** multiplicative position encoding. NTK scaling for 1M-token context.
- **CS-GAT** Chebyshev K=3 spectral graph attention over Hermitian W_full adjacency.
- **CTP** CFLN Think Protocol: explicit thinking tokens forming a LISTA reasoning chain.
- **Continual learning stack**: SI (displacement-only Ω) + alpha_freeze + Fisher-KL (v9.0) + domain detection.
- **VQ-Telescope**: unified CNEP-rooted memory space replacing W_compress (v9.0 Step 3).
- **Precision-weighted U_meta**: self-calibrating metacognition replacing fixed log_w_meta (v9.0 Step 2).

**Grade projection**: Overall B+ (v6.0.9) → A (v9.0). See Part 10.

---

## PART 0: ARCHITECTURE SUMMARY

### Cognitive Processing Loop (v9.0 complete)

```
INPUT      → x_c_eff = x_c + goal_scale × g_c                        [R4]
ROUTING    → CNEP(x_c_eff); k-shot centroid refine on young units     [R4 + SE-1]
BINDING    → W_full += lam_bind×B_bind + lam_role×B_role + lam_comp×B_comp  [R1 + X + COMP-H]
AGGREGATE  → CS-GAT K=3 Chebyshev over enriched W_full
RECONSTRUCT→ LISTA (STELA smooth) + reservoir augment (SE-3)          [W1 + SE-3]
RETRIEVE   → ARC dual-key [concept+relational] + verbatim spans        [R2 + Y1]
REASON     → CTP think + HYPO branch + SSP D=4 stack                  [R3 + Z]
             B=2 beams scored by Q_BEAM (F1+F2+F3+F4+F5+D1)          [TS-1 + Q-BEAM]
EVALUATE   → U_meta_v4 [5: repr, epi_cal, hop, temp, hypo]            [R3+]
             → Precision-weighted U_meta (v9.0 Step 2)
STORE      → ARC writes dual-key + H_seq + micro_consolidate_arc()    [R2 + Y2 + KA-MC]
PERSIST    → SurpriseArchive + VQ-Telescope L1/L2/L3 buffers          [Y3 + Step 3]
OUTPUT     → DCG+ deferred commitment (adaptive threshold §1.74)
```

### Special Vocabulary (6 tokens added to base vocab)

```
base+0: <think>      THINK_START_ID
base+1: </think>     THINK_END_ID
base+2: <hypo>       HYPO_START_ID
base+3: </hypo>      HYPO_END_ID
base+4: <push_goal>  PUSH_GOAL_ID
base+5: </push_goal> POP_GOAL_ID        (</push_goal> token = POP signal)
```

### New Parameters Summary (v7.0 through v9.0)

| Parameter | Shape | Init | Optimizer | Section |
|---|---|---|---|---|
| log_lam_bind | scalar | -3.0 | opt_g | §1.30 |
| log_sigma_bind | scalar | log(2.0) | opt_g | §1.41 |
| log_alpha_arc | scalar | 0.0 | opt_g | §1.31 |
| g_c | (d_c,) cfloat buffer | zeros | — | §1.33 |
| W_goal_detect | (1, d_c) real | zeros | opt_g | §1.33 |
| log_lam_goal | scalar | -3.0 | opt_g | §1.33 |
| role_vecs | (R=8, d_c) cfloat | QR ortho | opt_g (not Muon/Stiefel) | §1.35 |
| log_lam_role | scalar | -3.0 | opt_g | §1.35 |
| log_lam_composition | scalar | -3.0 | opt_g | §1.55 |
| tau_smooth | scalar | 0.1 | opt_g | §1.40 |
| log_sigma_bind | scalar | log(2.0) | opt_g | §1.41 |
| eps_beam_scale | scalar | 0.1 | opt_g | §1.47 |
| log_w_beam | (3,) real | zeros | opt_g | §1.46 |
| **W_rc_bridge** | **(d_r_lista, d_r_node) cfloat** | random/√d_r_node | **opt_g (was buffer)** | **§1.50** |
| fisher (dict) | same shapes as AdamW params | zeros | buffer (not trained) | §1.57 |
| sigma_sq_buffer | (5,) float list | [1.0×5] | plain list | §1.58 |
| log_precision | (5,) float list | [0.0×5] | updated by precision rule | §1.58 |
| buf_L1_w_full | (K_L1, N_max_l) float32 | zeros | register_buffer | §1.59 |
| buf_L2_w_full | (K_L2, N_max_l) float32 | zeros | register_buffer | §1.59 |
| buf_L3_w_full | (K_L3, N_max_l) float32 | zeros | register_buffer | §1.59 |
| fisher_unit | (N_max_l,) float32 | zeros | buffer | §1.63 |
| log_cal_scale | scalar | log(0.15) | opt_g | §1.72 |
| log_blend_alpha | scalar | log(0.8) | opt_g | §1.73 |

### Removed Parameters (v9.0)

| Removed | Was in | Replaced by | Section |
|---|---|---|---|
| `log_w_meta` (5 scalars) | CUN | `log_precision` (5 floats, self-calibrating) | §1.58 |
| `_log_w_rec` (5 floats) | CUN | `sigma_sq_buffer` (plain list, init 1.0) | §1.58 |
| `W_compress_L1/L2/L3` | TelescopingMemory | VQ routing weight vectors | §1.59 |
| `W_decompress_L1` | TelescopingMemory | Centroid mean reconstruction | §1.59 |
| `L_recon` training term | train_step | `L_vq` encoder commitment | §1.59 |
| SA cosine dedup (`tau_sa_dedup`) | SurpriseArchive | `E_min_raw` Welford threshold | §1.68 |
| MC-2 session-adaptive `_log_w_rec` EMA | CUN.lista_forward | Precision update (§1.58) | §1.58 |
| `mask_embed` (d_c,) cfloat | CFBank | (removed — MDLM mask token embedded via embed table) | §1.44 |

**W_rc_bridge is the only parameter that changes status (buffer → nn.Parameter).**

### Removed Config Keys (v9.0)

These heuristic keys are replaced by adaptive mechanisms and must be **deleted** from any config dict:

| Removed key | Replaced by | Section |
|---|---|---|
| `surprise_threshold` | Welford E_min adaptive threshold | §1.68 |
| `spawn_threshold` | Welford E_min adaptive spawn | §1.69 |
| `ssp_stuck_threshold` | Think-budget natural timeout | §1.71 |
| `ssp_merge_alpha` | Q_BEAM quality-weighted merge | §1.65 |
| `tau_proto` | `tau_proto_min` (softer, U_epi-gated) | §1.70 |
| `beam_B` | Adaptive `B_eff` from U_meta | §1.66 |
| `lambda_recon` | `lambda_vq` (VQ-Telescope) | §1.59 |

---

## PART 0B: CRITICAL TYPE FACTS

Read before writing any code. These are the most common sources of subtle bugs.

### T1 — `bank.alpha_freeze` is a scalar float, NOT subscriptable

```python
bank.alpha_freeze          # float, e.g. 0.7 — the histogram percentile threshold
bank.alpha_freeze[i]       # ← TypeError: float is not subscriptable — NEVER DO THIS
```

The per-unit frozen signal is:
```python
bank.is_sensory_l[i]       # bool Tensor (N_max_l,) — True when unit i is frozen
```

Whenever the spec says "frozen unit check" or references `alpha_freeze` per-unit, use `bank.is_sensory_l[i]`.
Applies to: §1.57 (Fisher skip), §1.62 (test), §1.63 (C1 Fisher-magnitude trigger).

### T2 — `r_lista` is always detached (BPTT disabled by design)

```python
self.r_lista = (...).detach()    # always — no gradient flows through r_lista across steps
r_seed       = W_bridge @ rho_weighted  # r_seed IS differentiable (before detach)
```

The `L_bridge` loss (§1.50) provides gradient to `W_bridge` through `r_seed` *before* detach.

### T3 — `U1` and `U2` are NEVER trained

```python
stiefel_ids = {id(W_l), id(W_p), id(U1), id(U2)}  # U1/U2 excluded from ALL optimizers
```

They are Haar-random unitary matrices, fixed at init. Do not add to any optimizer.

### T4 — VQ codes store full routing weight vectors, not index arrays

```python
# WRONG: store sel_l indices and compute Jaccard
# CORRECT: store full N_max_l-dimensional weight vector
s_l_full = torch.zeros(bank.N_max_l)
s_l_full[sel_l] = s_l[:k_l_eff]               # scatter routing weights into dense vector
bank.buf_L1_w_full[ptr] = s_l_full.detach()   # (N_max_l,) float32, sparse
```

### T5 — `sigma_sq_buffer` must init to 1.0, not 0.0

```python
self.sigma_sq_buffer = [1.0, 1.0, 1.0, 1.0, 1.0]   # NOT [0.0, ...]
```

If zero: `-0.5 * log(1e-6) = 6.9`, clamped to 3.0, giving ~20× weight on the first token.

### T6 — `L_vq` must detach the codebook, not the encoder

```python
z_approx = bank.mu_c_l[sel_l].mean(0).detach()  # codebook side detached (trained by CNEP)
L_vq = (chunk_mean - z_approx).norm()**2          # gradient flows to chunk_mean (encoder)
```

Never write `chunk_mean.detach()` in `L_vq` — that was the bug fixed in §1.51.

### T7 — Fisher accumulation ordering in train_step

```
loss.backward()
→ accumulate_fisher()      ← BEFORE clip_grad_norm_ (unclipped = true curvature)
→ clip_grad_norm_()
→ optimizer.step()
```

Fisher after clipping systematically underestimates importance of high-gradient parameters.

---

## PART 0C: IMPLEMENTATION ORDER

Apply changes in this exact group order. Each group is internally self-consistent. Do not start a later group before the earlier one compiles and tests pass.

### Group 1 — Infrastructure (no dependencies)
| Section | What | Files |
|---|---|---|
| §1.36 | `buf_L1_ids` verbatim span buffer | bank.py |
| §1.38 | `persist_archive` flag | surprise.py |
| §1.43 | k-shot centroid buffers (`_proto_count`, `_proto_sum`) | bank.py |
| §1.44 | `mask_embed` parameter | bank.py |
| §1.50 | `W_rc_bridge` buffer → nn.Parameter | model.py |
| §1.57 | `fisher` dict buffer + `accumulate_fisher()` | train_step.py, bank.py |
| §1.68/69 | Welford E_min buffers (`_Emin_mean`, `_Emin_var`, `_Emin_n`) | bank.py |
| §1.72 | `log_cal_scale` parameter | bank.py |
| §1.73 | `log_blend_alpha` parameter | cun.py (diffusion.py) |
| §1.77 | Config key changes (add new, remove heuristic keys) | config.py |

### Group 2 — Binding (depends on Group 1)
| Section | What | Files |
|---|---|---|
| §1.30/1.41 | `B_bind` phase kernel + `log_sigma_bind` | cfl5.py |
| §1.35 | `B_role` RAH + `role_vecs` (QR ortho init) | cfl5.py, bank.py |
| §1.55 | `B_comp = B_bind ⊙ B_role` (Hadamard, `log_lam_composition`) | cfl5.py |
| §1.33 | Goal-anchored context `g_c` | bank.py, cfl5.py |
| §1.64 | Precision-adaptive `k_l_eff` | cfl5.py |

### Group 3 — Reasoning (depends on Groups 1–2)
| Section | What | Files |
|---|---|---|
| §1.31 | Two-key ARC (`phi_rel`, `log_alpha_arc`) | diffusion.py |
| §1.32 | HYPO mode (`r_lista_hypo`, HYPO_START/END tokens) | model.py, diffusion.py |
| §1.34 | `U_meta_v4` five-signal (add `U_hypo`, extend to 5) | diffusion.py |
| §1.40 | STELA smooth threshold (`tau_smooth`, clamp min=1e-3) | diffusion.py |
| §1.45 | Reservoir-augmented LISTA reconstruction | diffusion.py |
| §1.58 | Precision-weighted U_meta (`log_precision`, `sigma_sq_buffer`) | diffusion.py |

### Group 4 — Memory (depends on Groups 1–2)
| Section | What | Files |
|---|---|---|
| §1.59 | VQ-Telescope (remove `W_compress`, add `buf_L1_w_full`) | bank.py, telescoping.py |
| §1.51 | Compression gradient fix (remove `.detach()` from `chunk_mean` in `L_compress`) | telescoping.py |
| §1.42 | `L_recon` → `L_vq` transition | telescoping.py, training/train_step.py |

### Group 5 — Planning/Reasoning Stack (depends on Groups 1–4)
| Section | What | Files |
|---|---|---|
| §1.39 | SSP goal stack PUSH/POP D=4 (add 6 vocab tokens) | model.py, diffusion.py |
| §1.47 | `r_lista` beam `B_eff` (`eps_beam_scale`, `log_w_beam`) | diffusion.py |
| §1.46 | `Q_BEAM` composite F1–F5+D1 (`compute_Q_beam`) | diffusion.py |
| §1.48 | Lyapunov goal proxy (`r_lista_goal_proxy`) | cfl5.py, diffusion.py |
| §1.49 | CSP arc-consistency via SSP stack (F5) | diffusion.py |
| §1.52 | Lyapunov timeout auto-POP (remove `_stuck_count`, use think-budget §1.71) | diffusion.py, model.py |
| §1.53 | `phi_rel` richness D1 in Q_BEAM | diffusion.py |
| §1.65 | Q_BEAM-weighted SSP merge (`_last_Q_BEAM_score`) | diffusion.py |
| §1.66 | U_meta-adaptive beam `B_eff` | diffusion.py |
| §1.71 | Remove `N_stuck` config (think-budget is natural timeout) | diffusion.py, config.py |

### Group 6 — Continual Learning / Emergence (depends on all groups)
| Section | What | Files |
|---|---|---|
| §1.37 | CONSOL-1 (`consolidate_arc_to_cnep`, called before reset) | bank.py, model.py |
| §1.54 | Micro-consolidation per-chunk | telescoping.py |
| §1.57 | Fisher-KL (`accumulate_fisher`, `L_KL`, ordering) | training/train_step.py |
| §1.63 | Fisher-magnitude alpha_freeze (`fisher_unit` buffer → `is_sensory_l`) | bank.py, training/train_step.py |
| §1.67 | Fisher-scaled consolidation rates | bank.py |

### Group 7 — Training Objectives (depends on all groups)
| Section | What | Files |
|---|---|---|
| §1.44 | MDLM masked token training (Stage 0, `L_mlm`) | training/train_step.py |
| §1.50 | `L_bridge` (W_bridge predictive coding loss) | cfl5.py, training/train_step.py |
| §1.56 | ROB-L/S regularisers (`L_lipschitz`, `L_sigma_reg`) | training/train_step.py |
| §1.60 | Full `train_step_v900` ordering | training/train_step.py |

### Group 8 — Emergent Parameter Replacements (depends on all groups)
| Section | What | Removes |
|---|---|---|
| §1.63 | Fisher-magnitude freeze (parallel trigger) | `alpha_freeze` 85th-pct-only logic |
| §1.64 | `k_l_eff = f(U_epi_cal)` | fixed `k_l=40` |
| §1.65 | `merge_weight = sigmoid(Q_BEAM)` | `ssp_merge_alpha=0.7` |
| §1.66 | `B_eff = f(U_meta)` | fixed `B=2` |
| §1.67 | `alpha_eff = alpha_base/(1+fisher_unit)` | fixed `alpha_consol`/`alpha_micro` |
| §1.68 | Welford `E_min` surprise | `surprise_threshold=0.5` |
| §1.69 | Welford spawn threshold | `spawn_threshold=3.0` |
| §1.70 | U_epi-gated k-shot | `tau_proto=0.6` |
| §1.71 | Think-budget timeout | `N_stuck=12` |
| §1.72 | `log_cal_scale` in MC-1 | hardcoded `0.15` |
| §1.73 | `log_blend_alpha` in r_lista | hardcoded `0.8` |
| §1.74 | U_epi-adaptive DCG+ commit | `commit_threshold=0.4` |
| §1.75 | U_temporal rule_util decay | hardcoded `0.999999` |

---

## PART 1: MATH SPECIFICATION

### §1.1 CNEP Energy

```
E_i(x_c) = ‖W_i·(x_c−μ_c_i)‖² = Re((x_c−μ_c_i)^H·W_i^H·W_i·(x_c−μ_c_i))
```

E_i(μ_c_i)=0, E_i≥0 always, differentiable everywhere.

v6.0.7 PF-1 — Efficient CNEP (inference only): activation-sorted index; early-exit batched scan (batch=256, min=n_l//4); conditional top-k reuse if `‖Δx_c‖<0.05 AND max|ΔE_top-k|<0.1`.

---

### §1.2 Two-Tier CNEP Design

| Tier | Routing | Normalization | Update |
|---|---|---|---|
| Local (n_l dynamic) | RQ: (1+E/ℓ²)^{-α} | entmax-1.5 + floor | SI-weighted Stiefel lr |
| Persistent (n_p, slow lr) | Soft-exp: exp(−E/ℓ²) | softmax | lr_persist=1e-6 + SI protection |

**v6.0.8: Two-tier design.** Global tier removed (n_g, W_g, mu_c_g). Role covered by alpha_freeze-protected local units + persistent softmax. CS-GAT k² drops 10,816→1,600 (6.76×). n_l_default += 64 to compensate.

---

### §1.3 Multiplicative CRoPE with NTK Scaling (R7)

```
x_c_k_out = x_c_k · exp(i·t·θ_k),   θ_k = 1/(rope_base^{2k/d_c})
rope_base  = base_orig × (L_target/L_train)^{d_c/(d_c−2)}   ≈ 5.25M for 1M-token context
```

`|exp(iθ)|=1` — magnitude preserved.

**CRoPE placement**: only at (1) CFL-5 residual (absolute position) and (2) Titans Q_t query inside `titans_query`. Never before CNEP energy — distance-to-centre is position-independent.

---

### §1.4 Titans Gradient-Based Memory (R3)

```
K_t=W_K·ē_c,  V_t=W_V·ē_c,  Q_t=W_Q·ē_c        (ē_c = chunk mean)
e_t = M_{t-1}·K_t − V_t                           (prediction error = surprise)
s_t = (e_t^H·e_t).real                            (surprise scalar ≥ 0)
θ_t = sigmoid((w_θ ⊙ ē_c.real).sum())             (input-dependent decay)
uw_t = 1 − sigmoid(k·(cos(ē_c,ē_{c,prev}) − τ))  (null-update weight)
M_t  = θ_t·M_{t-1} − uw_t·η·e_t⊗K_t^H           (rank-1 Wirtinger update)
r_t  = M_t · Q_t                                   (retrieval)
```

Self-correction: on domain shift, large e_t → large M update → M adapts within ~10 chunks. No forced reset needed. Titans M update suppressed during CTP thinking tokens (`_in_thinking_mode=True`).

---

### §1.5 Domain Detection → SI Snapshot Trigger

```
s_mag_ema  ← 0.99·s_mag_ema + 0.01·s_t           (slow normalizer)
s_norm_t   = s_t / (s_mag_ema + ε)
s_domain_ema ← 0.90·s_domain_ema + 0.10·s_norm_t (fast detector)
domain_shift_detected = s_domain_ema > τ_dom       (flag only — no M reset)
```

Three detection channels: (1) Titans EMA, (2) M4 routing diversity monitor, (3) SlowDriftDetector.

---

### §1.6 Hierarchical Telescoping Memory (R5)

Three FIFO buffers: L1=K_L1 chunks (C_chunk precision), L2=K_L1×32, L3=K_L1×32×32 tokens. Retrieval via softmax-weighted Hopfield per level; 4-way independent sigmoid gate (cooperative, not competitive). O(192×d_c) retrieval ops.

**v6.0.6 position-indexed skip**: high-surprise chunks (Titans s_t > 90th-pct running threshold) tagged in 64-slot skip ring for direct position-indexed recall. Additive gate with L1 Hopfield.

**v9.0**: W_compress/W_decompress replaced by VQ-Telescope (§1.59). See §1.42 removal note.

---

### §1.7 Surprise Archive (R6)

Min-heap permanent store of highest-surprise chunks. N_archive=256, adaptive 80th-pct threshold, W_warmup=32 chunk exclusion. Complements FIFO telescoping.

**v9.0 §1.68**: cosine dedup (`tau_sa_dedup`) replaced by Welford E_min threshold. `add_vq()` replaces `add()` — stores buffer pointer + E_min_raw score (VQ-Telescope §1.59). `SurpriseArchive.add_vq(buf_L1_ptr, E_min_raw)`.

---

### §1.8 mHC Highway (R2)

n_hc=2 parallel complex streams, doubly-stochastic mixing B_l∈R^{2×2}, ‖B_l‖₂≤1.

---

### §1.9 Muon Optimizer (R1)

`G_ortho = NewtonSchulz5(G_raw/‖G_raw‖_F)`. Applied to real stacking of complex params. SI omega increments by predictable `-lr_muon·min(m,n)` per step.

---

### §1.10 Reactive lam_P (M4 Monitor)

`lam_P_eff_l = lam_P_base_l · correction_l`. Non-differentiable safety net for routing collapse.

---

### §1.11 Adaptive α_freeze

v6.0.9 base: 85th percentile of α_l distribution. Units with α_i > α_freeze → sensory (W_l, μ_c_l frozen, domain-tagged for reversibility).

**v9.0 §1.63**: replaced by Fisher-magnitude threshold (1.5σ above mean fisher_unit). See §1.63.

---

### §1.12 SI Regularization

```
L_SI = (c_SI/2)·Σ_i Ω_i·|θ_i − θ_i*|²
Ω_i ← ρ·Ω_prev + (1−ρ)·|Δθ_i|²     (displacement-only; no gradient term)
```

v6.0.7 PF-2: batched `apply_psd` across all L layers via single `torch.linalg.eigh` call.

**v9.0 §1.57**: SI-Stiefel (W_l, W_p) split from Fisher-KL (AdamW params). See §1.57.

---

### §1.13 LISTA Iterative Reasoning

```
h_0 = sigmoid(log_β_rs)·W_rs·r_lista^{t-1}   (warm start; = 0 when r_lista=0)
h_k = shrink_c(einsum('ij,bj->bi', S, h_{k-1}) + U1^H·x_c, τ_k)    k = 1..N_iter
x_k = U2·h_k

τ_k = exp(log_τ_schedule[k]) · base_τ · γ^k · clamp(min=1e-3)
S_init = I_{d_c} − U2·U1^H                (iter=1 with h_0=0 ≡ CUN)
‖S‖₂ ≤ ρ_max=0.95                         (5-step power iteration)
γ ≥ 0.1                                    (floor prevents tau→0 at large k)
```

**v8.0 §1.40 STELA**: hard shrink replaced by smooth threshold (§1.40).

---

### §1.14 Per-Sequence LRU

`per_sequence_memory=True`: LRU maintains separate h state per batch sequence. Eliminates cross-sequence contamination.

**v6.0.6 Selective gating** (Mamba-inspired):
```
W_select ∈ R^{d_ssm × d_c}: learned selection matrix (init 0)
λ_eff^t = λ_base × (1 + 0.1 × sigmoid(W_select @ x_c^t.real))
h^t     = λ_eff^t ⊙ h^{t-1} + x_c^t @ B_c^H
```

---

### §1.15 Hopfield Capacity

```
capacity ≈ 0.14 × n_l² / d_c    (Ramsauer 2020)
k_max    = floor(0.10 × n_l² / d_c)  (30% safety margin)
```

Filter to top-k_max most-similar units before Hopfield completion.

---

### §1.16 W_compress Gradient

**v6.0.9 fix (§1.51 clarification)**: L_compress must NOT use `.detach()` on chunk_mean. See §1.51 for full fix.

```
L_compress = ‖chunk_mean − W^H(W·chunk_mean)‖²     (chunk_mean NOT detached)
```

**v9.0 §1.59**: W_compress_L1/L2/L3 and W_decompress_L1 removed entirely (VQ-Telescope).

---

### §1.17 Node Fourier Reservoir — Predictive CNEP

Each CNEP local-tier unit i carries reservoir state ρ_i^t ∈ C^{d_r_node}.

**Fourier eigenvalues (fixed, multi-scale):**
```
λ_k = ρ_group(k) · exp(i·2π·(k+0.5)/d_r_node),   k = 0..d_r_node-1
ρ_group(k): d_r_node//4 dims each at ρ_fast=0.85, ρ_mid=0.90, ρ_node=0.95, ρ_slow=0.99
```
Requires d_r_node divisible by 4. ~6/10/20/100 token memory scales.

**Reservoir dynamics:**
```
e_i^t = W_enc @ (W_i · (x̄_c^t − μ_c_i))          (projection error; W_enc FIXED buffer)
ρ_i^t = λ ⊙ ρ_i^{t-1} + e_i^t    (active units)
ρ_i^t = λ ⊙ ρ_i^{t-1}             (inactive: free decay only)
```

v5.9.6 I2 surprise-salience gate: `e_i^t ← e_i^t · clamp(s_norm_t, 0.3, 2.0)`.

**Predictive prototype (psi_for only — NOT used in routing E_l):**
```
δ_i^t = W_dec @ ρ_i^t
μ_pred_i^t = μ_c_i + exp(log_scale_l[i]) · δ_i^t
```

v6.0.6: per-unit spectral filter `log_decode_scale ∈ R^{n_l × d_r}` (init 0).

---

### §1.18 LISTA Session Reservoir — Cross-Token Reasoning

```
r_lista ∈ C^{d_r_lista}          (resets at document boundary)
λ_lista = ρ_lista · exp(i·2πk/d_r_lista)

warm = W_rs @ r_lista
h_0^t = sigmoid(log_β_rs) · warm

e_lista = mean_b(h_N)
r_lista ← λ_lista ⊙ r_lista + W_ri @ e_lista    (W_ri FIXED buffer; fully detached)
```

W_rs: trained readout. W_ri: fixed random buffer (ESN design; zero gradient confirmed v5.9.5 B2).

**v9.0 §1.73**: fixed 0.8/0.2 blend replaced by learned `log_blend_alpha`.

---

### §1.19 RC Bridge — Unified Two-Scale Reservoir (v5.9.6 I4 / §1.50 v9.0)

```
rho_weighted^t = Σ_i s_l_mean[i] · ρ_i^t    ∈ C^{d_r_node}
r_seed^t       = W_bridge · rho_weighted^t    ∈ C^{d_r_lista}
r_lista^t      ← blend_alpha · r_lista^{t-1} + (1−blend_alpha) · r_seed^t
```

**v9.0 §1.50**: W_bridge changed from `register_buffer` to `nn.Parameter`. Trained via L_bridge predictive coding loss. See §1.50.

---

### §1.20 U_meta Gate — Self-Regulating Warm Start

```
β_eff^t = sigmoid(log_β_rs) · max(0.1, 1.0 − 0.7 · U_meta^{t-1})
h_0^t   = β_eff^t · W_rs · r_lista^{t-1}
```

High U_meta → weak warm start (poor prior). Floor 0.1 prevents complete kill.

---

### §1.21 Adaptive LISTA Depth

```
N_min      = max(2, ⌊N_max · r_min⌋)        r_min = lista_min_ratio (default 0.25)
N_adaptive = clamp(N_min + ⌊(N_max−N_min)·U_meta^{t-1}⌋, N_min, N_max)
```

Early exit: `k ≥ N_min AND δ_k < δ_stuck · r_conv (default 0.5)`.

---

### §1.22 Epistemic Uncertainty U_epistemic

```
E_min^t  = min_{i: s_l_i > 1/n_l} E_i(x_c)
H_route^t = −Σ_i s_l_i · log(s_l_i)
e_norm   = E_min / (_e_min_ema + ε)
h_norm   = H_route / (_h_route_ema + ε)
U_epistemic = sigmoid(α · (e_norm · h_norm − 1.0))    α=2.0
```

v6.0.7 MC-1: rolling normalisation → `U_epi_cal = σ(0.15 × (U_epi − μ_U)/(σ_U+ε) + 0.5)`.

**v9.0 §1.72**: fixed scale 0.15 replaced by learned `log_cal_scale`.

---

### §1.23 Sparse Code Cache (CTX.A)

```
Cache C = {h_N^{t-k}}_{k=0}^{K-1}    shift-buffer
Query:    x̄_c ∈ C^{d_c}
Content:  a_k = softmax((x̄_c · h_k^H).real / √d_c)
Recency:  r_k = k / (K−1)
Combined: w_k = (1−γ)·a_k + γ·softmax(r_k)    γ=0.3
Gate:     g = σ(Re(W_cache_gate^H · x̄_c))
Augment:  h_0^t ← h_0^t + g · h_ret
```

K=32 default. Reset at document start.

---

### §1.24 Composite U_meta_v3 (v6.0.7 — superseded by §1.34 + §1.58)

Four signals (U_repr_q, U_epi_cal, U_hopfield, U_temporal) fused via learned softmax weights log_w_meta∈R^4.

**v7.0 §1.34**: extended to 5 signals (add U_hypo). **v9.0 §1.58**: log_w_meta replaced by precision-weighted U_meta. See §1.34 and §1.58.

---

### §1.25 Sequential Hebbian H_seq (R3.A)

```
H_seq[i,j] += η · 1[unit i ∈ sel_{t-1}] · 1[unit j ∈ sel_t]    η = 0.01
H_seq       ← (1 − λ_decay) · H_seq                              λ_decay = 0.005
H_seq       ∈ [0,1]

GAT augmentation: W_full[:k_l,:k_l] += λ_seq_gat · H_seq[sel%K_hebb, sel%K_hebb]
```

Updated only at last CFL layer + training. K_hebb×K_hebb = 16×16 = 256 floats.

---

### §1.25b Chebyshev Spectral GAT (CS-GAT, v6.0.6)

```
Ã = (W_full + W_full^H) / 2 + I × ε    (Hermitian symmetrisation + self-loops)
T_0 = h_in;   T_1 = Ã @ h_in
T_k = 2·Ã @ T_{k-1} − T_{k-2}          (Chebyshev recurrence, K_cheby=3)

h_out = Σ_{k=0}^{K_cheby} diag(θ_k) @ T_k
θ_k ∈ C^{d_c}: learned complex spectral filter per hop (Muon group)
```

PSD W_full → eigenvalues ≥ 0 → Chebyshev polynomials well-conditioned.

---

### §1.26 Adaptive Rule Consolidation (ARC) Cache (v6.0.6)

Ring buffer of N_rules=64 (v9.0 §1.76: 256) recently-discovered reasoning rules.

```
Rule write — DUAL TRIGGER (NR-1):
  Trigger A: escape fired + U_meta < 0.3
  Trigger B: U_epistemic > 0.6 + U_meta < 0.4
  → prototypical merge if max_sim(K_new, existing) > 0.7; else write new

Rule retrieval (NR-2/NR-3):
  x_query = x_c.mean(0) @ U1^H
  sim_k   = cosine(x_query, K_concept_i) weighted by sigmoid gate g_rule
  top-3 softmax (T=0.5) → v_blend → h_0 augmented
```

v7.0 §1.31: K_rule extended to (N_rules, 2×d_c) for dual-key (concept + relational).

---

### §1.27 Deferred-Commitment Generation (DCG+, v5.9.9)

Three-phase: Draft (M tokens parallel) → Reflect (optional K alternatives) → Selective Revision (monotonicity constraint). Commit score: `commit_i = σ(w[0]·(1−U_epi) + w[1]·z_contrib + w[2]·U_hop)`. `w_commit ∈ R^3` init 1.0.

2.6× faster than standard AR (T=256, M=8, R=2 rounds).

**v9.0 §1.74**: fixed commit threshold 0.4 replaced by `max(0.1, 1.0 − U_epi_cal)`.

---

### §1.28 CFLN Think Protocol — CTP (v6.0)

```
THINK_START_ID = original_vocab_size        (<think>)
THINK_END_ID   = original_vocab_size + 1    (</think>)
```

Thinking tokens form LISTA reasoning chain: each token's h_N seeds next token's warm start via r_lista.

Memory during thinking: r_lista/rho_l/H_seq/h_c_l updated. LRU/h_cache/rule_K/Titans M/telescoping/SurpriseArchive/proactive-SI suppressed.

```
L_CTP = (1/N) Σ_t weight_t · CE(logits_t, target_t)
weight_t = τ_think  ∈ {<think>} ∪ interior ∪ {</think>}   (inclusive)
         = 1.0      otherwise
τ_think = 0.5 (SFT phase), 0.0 (GRPO phase)
```

---

### §1.29 PSC–RPP–RL Reasoning Training Pipeline (v6.0.5)

**PSC Stage 1** — self-supervised pre-training using model's own predictions:
```
L_improve    = −log σ(CE_baseline − CE_thinking + margin=0.1)
L_economy    = (1−U_epi) × ‖r_lista^K − r_lista^0‖²
L_predictive = Σ_{n=3}^{5} U_epi^{t+n} × ‖W_pred@r_lista^K − h_N^{t+n}‖²
L_PSC        = L_LM + 1.0·L_improve + β(U_epi)·L_economy + 0.5·L_predictive
```

**RPP Stage 2** — continuous gradient optimisation of think embeddings → discretise → accept if CE improves ≥5%.

**GRPO Stage 4** — intrinsic reward `R_t = CE_baseline_t − CE_thinking_t`; G=8 rollouts; β=0.1 KL penalty.

---

### §1.30 Phase Similarity Kernel — Binding via Complex Phase (v7.0 R1)

```
φ_i = angle(bank.H_c_l[sel_i].mean())           scalar ∈ [−π,π] per selected unit
σ   = exp(log_sigma_bind)                        learned kernel width (§1.41 W2)
B_bind[i,j] = exp(−|φ_i − φ_j|² / σ²)          (k_l,k_l) real, PSD by RBF theorem
W_full[:k_l,:k_l] += exp(log_lam_bind) × B_bind
```

PSD safe: RBF kernel → always PSD → sum of PSD is PSD → apply_psd satisfied. ✓

---

### §1.31 Two-Key ARC Cache — Structural Retrieval (v7.0 R2)

```
k_concept = x_c.mean(0) @ U1.conj().T              (d_c,) existing key
phi_rel   = top_eigenvec(H_seq_sub[sel_k][:,sel_k]) (k_l,) recomputed every C_chunk=32 tokens
k_rel     = phi_rel @ psi_all[:k_l]                (d_c,) new relational key
K_rule    = concat([k_concept, k_rel])              (2×d_c,) stored; rule_K shape: (N_rules, 2×d_c)

α = sigmoid(log_alpha_arc)
sim = α × cos(q_concept, K_concept) + (1−α) × cos(q_rel, K_rel)
```

---

### §1.32 CTP Hypothetical Mode (v7.0 R3)

```
On HYPO_START_ID:
  r_lista_hypo  = r_lista.clone()     (branch)
  _in_hypo_mode = True
  g_c frozen

During HYPO: lista_forward uses r_lista_hypo; ARC writes SUPPRESSED

On HYPO_END_ID:
  U_hypo = sigmoid(‖r_lista_hypo − r_lista‖² / d_r_lista)
  _in_hypo_mode = False; r_lista_hypo = None
  g_c resumes updates
```

---

### §1.33 Goal-Anchored Context (v7.0 R4)

```
g_t     = σ(W_goal_detect @ x_c_mean.real)                     soft gate
g_c     ← g_t × x_c_mean + (1−g_t) × g_c                      soft goal update
x_c_eff = x_c + exp(log_lam_goal) × g_c.unsqueeze(0)           effective context
CNEP routing uses x_c_eff; g_c frozen during HYPO and PUSH_GOAL
```

---

### §1.34 U_meta_v4 — Five-Signal Metacognition (v7.0 R3+)

```
U_hypo    = sigmoid(‖r_lista_hypo − r_lista‖² / d_r_lista)    0 when not in HYPO
```

v6.0.9 base: U_meta_v4 = softmax(log_w_meta∈R^5) ⊙ [U_repr_q, U_epi_cal, U_hop, U_temp, U_hypo]

> **Replaced in v9.0 §1.58**: `log_w_meta` (5 scalars) and `_log_w_rec` (5 floats) removed. Replaced by precision-weighted U_meta with `sigma_sq_buffer` and `log_precision`. See §1.58.

---

### §1.35 Role Attention Heads — Structural Binding (v8.0 X)

```
α_{ij}   = softmax_j(Re(μ_c_l[sel_i] · r_j^H) / √d_c)        (k_l, R) role assignment
B_role   = α @ α.T                                              (k_l,k_l) PSD (Gram matrix)
W_full[:k_l,:k_l] += exp(log_lam_role) × B_role

role_vecs: (R=8, d_c) cfloat, QR-orthogonal init
           excluded from Muon/Stiefel via stiefel_ids
```

Synergy: B_bind = temporal phase alignment (theta-like) ⊗ B_role = structural role alignment (gamma-like). Hippocampal theta-gamma dual-oscillation analog.

---

### §1.36 Verbatim Span Buffer (v8.0 Y1)

```
buf_L1_ids: (K_L1, C_chunk) int32 register_buffer on CFBank
buf_L2_ids: (K_L2, C_chunk×32) int32 register_buffer

On L1 chunk write: buf_L1_ids[ptr] = chunk_token_ids
On retrieval:      also return buf_L1_ids[argmax(Hopfield_weights)]
```

---

### §1.37 OQ-CONSOL-1 — Automated Knowledge Consolidation (v8.0 Y2)

```python
def consolidate_arc_to_cnep(bank, cun, tau_consol=3.0, alpha_consol=0.001):
    for idx in range(cun._rule_cache_n):
        if rule_util[idx] < tau_consol: continue
        k_rule  = cun.rule_K[idx, :bank.d_c]
        nearest = argmin(‖k_rule − μ_c_l[:]‖)
        si_gate = (1 − SI_omega_norm[nearest]).clamp(0, 1)
        μ_c_l[nearest] += alpha_consol × si_gate × (k_rule − μ_c_l[nearest])
```

Called BEFORE session state reset in `reset_for_inference()`.

**v9.0 §1.67**: `alpha_consol` scaled by `1/(1 + fisher_unit[nearest])`.

---

### §1.38 SurpriseArchive Persistence (v8.0 Y3)

```python
cfg: persist_archive = False   (default off)
     archive_path    = 'archive.pt'

reset_for_inference():
    if cfg['persist_archive']:
        surprise_archive.save_state(cfg['archive_path'])
```

---

### §1.39 Subgoal Stack Protocol (v8.0 Z)

```
On PUSH_GOAL_ID: _goal_stack.append(r_lista.clone())   max depth D=4
On POP_GOAL_ID:  parent = _goal_stack.pop()
                 r_lista ← merge_weight × r_lista + (1−merge_weight) × parent
```

v6.0.9 base: merge_weight=0.7 (fixed). **v9.0 §1.65**: replaced by Q_BEAM-weighted merge. `g_c` frozen during PUSH_GOAL. ARC reads and writes allowed during subgoal.

---

### §1.40 STELA Smooth LISTA Thresholding (v8.0 W1)

```
Old: h = sign(h) × max(|h|−τ, 0)                     (discontinuous)
New: h = h × sigmoid((h.abs()−τ) / τ_smooth.clamp(min=1e-3))

τ_smooth: scalar nn.Parameter, init 0.1, opt_g
```

clamp(min=1e-3) prevents sign-flip if τ_smooth goes negative.

---

### §1.41 Learned Phase Kernel Width (v8.0 W2)

```
log_sigma_bind: scalar nn.Parameter, init log(2.0) ≈ 0.693, opt_g
σ = exp(log_sigma_bind)    used in §1.30 B_bind formula (replaces fixed σ=1.0)
```

---

### §1.42 Compression Reconstruction Loss (v8.0 W3)

> **Removed in v9.0**: §1.59 (VQ-Telescope) removes W_decompress_L1 entirely and replaces L_recon with L_vq (encoder commitment loss). W_compress_L1/L2/L3 also removed.

Original definition for reference:
```
W_decompress_L1 ∈ ℂ^{d_c × d_c}: init eye + 0.01×noise, Muon
L_recon = ‖chunk_mean − W_decompress @ W_compress @ chunk_mean‖²
loss += λ_recon × L_recon    (λ_recon = 0.01)
```

---

### §1.43 k-Shot Centroid Refinement (Addendum SE-1)

```
On spawn(idx, x_c):
  _proto_count[idx] = 1;  _proto_sum[idx] = x_c.mean(0)

Per routing to young unit idx (activation_freq < α_young=0.1):
  if U_epi_cal < 0.4 AND cosine_sim(x_c, μ_c_l[idx]) > tau_proto_min=0.4:
  AND _proto_count[idx] < K_proto_max=10:
    _proto_count[idx] += 1
    _proto_sum[idx]   += x_c.mean(0)
    μ_c_l[idx]         = _proto_sum[idx] / _proto_count[idx]

After K_proto_max exposures: alpha_freeze triggers → crystallise centroid

Buffers: _proto_count (N_max_l,) int32 register_buffer
         _proto_sum   (N_max_l, d_c) cfloat register_buffer
```

v6.0.9 base threshold was `tau_proto=0.6` (fixed cosine). **v9.0 §1.70**: replaced by U_epi-gated accumulation with `tau_proto_min=0.4`.

---

### §1.44 MDLM Masked Token Training (Addendum SE-2)

```
Stage 0 only:
  mask_positions = Bernoulli(p_mask=0.15) over (B, T)
  x_c[mask_positions] = embed(MASK_TOKEN_ID)   (standard embedding lookup)
  L_mlm = CE(logits[mask_positions], true_tokens[mask_positions])
  loss += λ_mlm × L_mlm    λ_mlm = 0.3
```

**Implementation** (masking at token-embedding level in `CFLNModel.forward`):
```python
if stage == 'stage0' and p_mask > 0:
    mask_positions = torch.bernoulli(
        torch.full(input_ids.shape, p_mask, device=input_ids.device)
    ).bool()
    # x_c_masked uses standard embed table — no separate mask_embed parameter
    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = MASK_TOKEN_ID
    x_c_masked = self.embed(masked_ids)   # ComplexEmbedding lookup
    x_c = x_c_masked
```

**v9.0 note**: The Addendum originally specified `mask_embed` as a `(d_c,) cfloat register_buffer`. v9.0 removes it — use `embed(MASK_TOKEN_ID)` standard table lookup instead (listed in Removed Parameters table, Part 0).

---

### §1.45 Reservoir-Augmented LISTA Reconstruction (Addendum SE-3)

```
x_c_recon = U2 @ h_N + W_dec_res @ rho_l[sel_l].mean(0)
```

Uses existing `W_dec_res` and `rho_l`. No new parameters. ~1.3K flops.
Echo State theory: reservoir state has 'echo state property' — fading memory of all past inputs.

---

### §1.46 Q_BEAM Multi-Field Beam Quality Composite (Addendum Q-BEAM)

Replaces MLP verifier. Parameter-free core; 3 optional scalars.

```
Q_beam_k = α_F×F1_k + α_R×F2_k + α_M×F3_k + α_V×F4_k + α_C×F5_k + α_D×D1_k

F1 (Thermodynamics):    -(E_min_raw × H_route_raw)
F2 (Predictive coding): -(‖r_seed_target − W_bridge@rho‖²)    [REQUIRED after §1.50]
F3 (MDL):               -‖h_N‖₁
F4 (Lyapunov):          -‖r_lista − r_lista_goal_proxy‖²
F5 (CSP):               min cosine_sim(r_lista, s) ∀ s ∈ _goal_stack  [0.0 if empty]
D1 (phi_rel richness):  ‖phi_rel‖                               (§1.53)

r_lista_goal_proxy = U1 @ g_c.conj()   (cached, amortised)

Weights: equal 1/N (default); optional log_w_beam ∈ R³ (F3,F4,F5 weights), init zeros, opt_g
```

---

### §1.47 r_lista Beam Search B=2 (Addendum TS-1)

Active during CTP think tokens only.

```
B_eff = max(1, round(1 + U_meta × (B_max−1)))   B_max=3   [§1.66 adaptive B]

Beam 1: r_lista_b1 = r_lista
Beam 2: r_lista_b2 = r_lista + eps_beam_scale × randn_like(r_lista)

h_b1, Q_b1 = lista_inner(x_c, r_lista_b1); compute_Q_beam(...)
h_b2, Q_b2 = lista_inner(x_c, r_lista_b2); compute_Q_beam(...)

w        = softmax([Q_b1, Q_b2])
h        = w[0]×h_b1 + w[1]×h_b2
r_lista  = (w[0]×r_lista_b1 + w[1]×r_lista_b2).detach()

L_diversity = -‖r_lista_b1 - r_lista_b2‖²   λ_diversity=0.01
```

---

### §1.48 Lyapunov Goal-Directed Planning (Addendum TS-3)

```
r_lista_goal_proxy = U1 @ g_c.conj()     (d_c,) cfloat; cached as _r_goal_proxy_cache
F4 in Q_BEAM: -‖r_lista_k − r_lista_goal_proxy‖²
```

SSP interaction: g_c frozen at PUSH_GOAL → r_lista_goal_proxy constant within subgoal → Lyapunov reference frame stable.

---

### §1.49 CSP Arc-Consistency via SSP Stack (Addendum TS-4)

```
F5 in Q_BEAM:
  consistency_k = min(cosine_sim(r_lista_k.real, s.real) for s in _goal_stack)
                = 0.0 if _goal_stack is empty
```

Weakest-link constraint propagation. Cost: stack_depth × d_r_lista dot products.

---

### §1.50 W_rc_bridge — Trained Parameter via Local Predictive Coding

**Change**: `register_buffer('W_rc_bridge', ...)` → `nn.Parameter(W_rc_bridge)`.

**Root cause**: r_seed.detach() severs gradient. Fix: local self-supervised loss before detach.

```
r_seed_target = (U1.conj() @ x_c.mean(0))[:d_r_lista]   (no new params, uses fixed U1)
L_bridge      = ‖r_seed_target − r_seed‖²
loss         += λ_bridge × L_bridge                       λ_bridge = 0.1
```

W_bridge learns: map reservoir state → LISTA-basis projection of x_c (predictive coding).

**Optimizer**: opt_g (AdamW). Excluded from Stiefel: `id(model.W_rc_bridge)` in stiefel_ids.

**F2 becomes REQUIRED**: once trained, `F2 = -‖r_seed_target − r_seed‖²` is L_bridge repurposed as inference quality signal.

**Test replacement**: delete `test_W_rc_bridge_is_buffer_not_param`; add `test_W_rc_bridge_is_trained_parameter`.

---

### §1.51 Compression Gradient Fix — Self-Organising Telescoping

```python
# BEFORE (broken v6.0.9):
L_compress = ((chunk_mean.detach() - x_recon_1).conj() *
               (chunk_mean.detach() - x_recon_1)).real.sum()

# AFTER (fixed):
L_compress = ((chunk_mean - x_recon_1).conj() *
               (chunk_mean - x_recon_1)).real.sum()
# .detach() removed → CFL5Layer receives gradient to produce compressible chunk means
```

**Safety**: `lambda_compress` reduced 0.01 → **0.001** (new upstream gradient would destabilise at old strength).

Note: L2/L3 targets (`_pending_L2/L3`) remain detached at storage (`c1_live.detach()`) — correct.

> **Note**: §1.59 (VQ-Telescope) subsequently removes W_compress entirely, making this fix a transitional step. The gradient design principle (no .detach() on encoder output for reconstruction losses) carries over to L_vq.

---

### §1.52 Lyapunov Timeout Auto-POP — Receding Horizon Planning (PLAN-B)

```
Session state on CUN:
  _stuck_count: list[int]    per stack depth, reset on PUSH
  _v_prev:      list[float]  previous V per depth

Per think token while in subgoal:
  V_curr = ‖r_lista − r_lista_goal_proxy‖²
  if V_curr >= _v_prev[-1]: _stuck_count[-1] += 1
  else: _stuck_count[-1] = 0
  _v_prev[-1] = V_curr

  if _stuck_count[-1] >= N_stuck:    # N_stuck = ssp_stuck_threshold config key
    parent = _goal_stack.pop()
    r_lista = parent               # discard failed subgoal (no merge)
```

**v9.0 §1.71**: N_stuck config key removed — think-token budget is the natural timeout.

---

### §1.53 phi_rel Richness Signal in Q_BEAM — Deduction Context (DED-D1)

```python
if phi_rel_cache is not None:
    D1 = float(phi_rel_cache.norm().item())   # relational richness signal
    signals.append(D1)
```

`phi_rel_cache` = `self._phi_rel_cache` from §1.31 (already computed). High D1 = strong relational structure → deduction well-supported.

---

### §1.54 Per-Chunk Micro-Consolidation — CLS Continuous Learning (KA-MC)

```python
def micro_consolidate_arc(bank, cun, cfg):
    """CLS micro-consolidation: top-1 ARC rule → μ_c_l per chunk."""
    n_r = getattr(cun, '_rule_cache_n', 0)
    if n_r == 0: return
    tau = cfg.get('tau_consol', 3.0)
    alpha_micro = cfg.get('alpha_micro', 0.0001)
    utils = cun.rule_util[:n_r]
    best = int(utils.argmax().item())
    if float(utils[best].item()) < tau: return
    k_rule = cun.rule_K[best, :bank.d_c].detach()
    with torch.no_grad():
        dists = (bank.mu_c_l[:bank.n_l] - k_rule).norm(dim=-1).real
        nearest = int(dists.argmin().item())
        freq = float(bank.activation_freq_l[nearest].item())
        si_gate = max(0.0, 1.0 - freq / cfg.get('alpha_young', 0.1))
        # §1.67: Fisher-scaled rate
        alpha_eff = alpha_micro / (1.0 + bank.fisher_unit[nearest])
        delta = alpha_eff * si_gate * (k_rule - bank.mu_c_l[nearest])
        bank.mu_c_l.data[nearest] += delta
```

Called at end of every `_update_telescoping()` (every C_chunk block).

---

### §1.55 Hadamard Composition Term — Structured Compositional Binding (COMP-H)

```
B_comp[i,j] = B_bind[i,j] × B_role[i,j]        element-wise product
PSD proof:    Schur product theorem — element-wise product of PSD matrices is PSD ✓
W_full[:k_l,:k_l] += exp(log_lam_composition) × B_comp
```

**CFL5Layer.forward W_full enrichment ordering**:
1. W_ll (RQ overlap) + H_mat (concurrent Hebbian) + H_seq (sequential Hebbian)
2. B_bind (phase RBF kernel, §1.30/§1.41)
3. B_role (role outer product, §1.35)
4. B_comp = B_bind × B_role (Hadamard AND-logic, §1.55) ← NEW in v9.0
5. CS-GAT Chebyshev aggregation over enriched W_full

---

### §1.56 Lipschitz Routing Regulariser + Phase Width Regulariser (ROB-L/S)

```python
# ROB-L: routing sharpness regulariser (young units only)
young_mask = (bank.alpha_freeze[:bank.n_l] == 0).bool()
if young_mask.any():
    L_lipschitz = bank.log_alp_l[:bank.n_l][young_mask].mean()
    loss += cfg.get('lambda_lipschitz', 0.001) * L_lipschitz

# ROB-S: phase kernel width regulariser
L_sigma_reg = torch.exp(-bank.log_sigma_bind)   # = 1/σ
loss += cfg.get('lambda_sigma_reg', 0.001) * L_sigma_reg
```

No new parameters. Uses existing `log_alp_l` and `log_sigma_bind`.

---

### §1.57 Hybrid Fisher-KL + SI-Stiefel — Principled CL Protection (Step 1)

```
Stiefel params (W_l, W_p):
  omega_i += |Δθ_i| × |grad_i|                              (SI displacement)
  L_SI_stiefel = β_SI × Σ_{Stiefel} omega_i × (θ_i − θ*_i)²

AdamW params (all others):
  fisher_i ← 0.99 × fisher_i + 0.01 × grad_i²              (Fisher EMA)
  L_KL = β_KL × Σ_{AdamW} fisher_i × (θ_i − θ*_i)²

Total CL: L_CL = L_KL + L_SI_stiefel
```

Fisher accumulation rules:
1. After `loss.backward()`, BEFORE `clip_grad_norm_()`
2. Only if `param.grad is not None`
3. Skip if unit is alpha_frozen

```python
def accumulate_fisher(model, stiefel_ids, bank, fisher):
    for name, param in model.named_parameters():
        if id(param) in stiefel_ids: continue
        if param.grad is None: continue
        if _is_unit_param(name) and _unit_is_frozen(name, bank): continue
        fisher[name] = 0.99 * fisher[name] + 0.01 * param.grad.detach()**2
```

**Early training**: anneal beta_KL from 0 to full over first 500 steps.

---

### §1.58 Precision-Weighted U_meta — Self-Calibrating Metacognition (Step 2)

> **Removes**: `log_w_meta` (5 scalars), `_log_w_rec` (5 floats), MC-2 session EMA block.

```python
sigma_sq_buffer: [1.0, 1.0, 1.0, 1.0, 1.0]   # init=1.0 (NOT 0.0 — prevents explosion)
log_precision:   [0.0, 0.0, 0.0, 0.0, 0.0]    # starts equal (unit variance → log_prec=0)
_precision_active: [False]*5

def update_precision(signals, sigma_sq_buffer, log_precision, _precision_active,
                     is_hypo_active, is_hopfield_active):
    for s, sig in enumerate(signals):
        val = float(sig)
        if s == 4 and not is_hypo_active:    continue  # U_hypo inactive
        if s == 2 and not is_hopfield_active: continue  # U_hopfield disabled
        if abs(val) > 1e-6: _precision_active[s] = True
        if not _precision_active[s]: continue
        sigma_sq_buffer[s] = 0.95 * sigma_sq_buffer[s] + 0.05 * val**2
        lp = -0.5 * math.log(sigma_sq_buffer[s] + 1e-6)
        log_precision[s] = max(-3.0, min(3.0, lp))

prec = torch.exp(torch.tensor(log_precision))
U_meta = (prec * signals_t).sum() / (prec.sum() + 1e-8)

# Training regulariser (prevents precision collapse):
L_precision = cfg.get('lambda_prec', 0.001) * torch.exp(torch.tensor(log_precision)).sum()
```

Reset policy: `sigma_sq_buffer = [1.0]*5` and `_precision_active = [False]*5` at session reset. `log_precision` NOT reset (accumulates across sessions).

---

### §1.59 VQ-Telescope — Unified Representation Space (Step 3)

> **Removes**: `W_compress_L1/L2/L3`, `W_decompress_L1`, `L_recon`, SA cosine dedup.

```python
# ADD to CFBank:
buf_L1_w_full: (K_L1, N_max_l) float32 register_buffer
buf_L2_w_full: (K_L2, N_max_l) float32 register_buffer
buf_L3_w_full: (K_L3, N_max_l) float32 register_buffer
# Memory: (128+32+32) × 2048 × 4 = 1.5MB at production scale

def vq_telescope_update(chunk_mean, s_l_full, E_min_raw, chunk_token_ids, bank, sel_l, cfg):
    ptr = bank._L1_ptr % bank.K_L1
    bank.buf_L1_w_full[ptr] = s_l_full.detach()   # full routing weight vector
    bank.buf_L1_ids[ptr]    = chunk_token_ids       # verbatim spans (§1.36)
    if E_min_raw > cfg.get('surprise_threshold', 0.5):
        bank.surprise_archive.add_vq(ptr, E_min_raw)
    # VQ encoder commitment loss
    z_approx = bank.mu_c_l[sel_l].mean(0).detach()   # codebook target, detached
    L_vq = (chunk_mean - z_approx).norm()**2           # gradient → encoder only
    bank._L1_ptr += 1
    # L2/L3 averaging (every C_L2=32 L1 chunks; every C_L2×C_L3 for L3)
    ...
    return L_vq

def vq_telescope_retrieve(s_l_full_query, bank, return_ids=False):
    sim_L1 = bank.buf_L1_w_full[:n_l1] @ s_l_full_query
    top_L1 = int(sim_L1.argmax().item())
    r_L1   = (bank.buf_L1_w_full[top_L1].unsqueeze(-1) *
              bank.mu_c_l[:bank.n_l].T).sum(-1)   # weighted centroid mean
    # analogously for r_L2, r_L3
    ...
```

Gradient correctness: `chunk_mean` grad flows to encoder; `mu_c_l` is detached (codebook trained by CNEP routing only). No circular dependency. ✓

---

### §1.60 Training Step Ordering (Steps 1–3 Combined)

```python
def train_step_v900(batch, model, opts, si, fisher, cfg, step):
    logits, info = model(batch)
    loss = compute_loss(logits, batch, info, cfg)
    loss += cfg['beta_KL']         * compute_L_KL(model, fisher, cfg)       # §1.57
    loss += cfg['beta_SI_stiefel'] * compute_L_SI_stiefel(model, si, cfg)   # §1.57
    loss += cfg.get('lambda_prec', 0.001) * compute_L_prec(model)           # §1.58
    if hasattr(bank, '_last_L_vq'):
        loss += cfg.get('lambda_vq', 0.01) * bank._last_L_vq                # §1.59
    loss.backward()
    accumulate_fisher(model, si.stiefel_ids, model.bank, fisher)  # BEFORE clip
    clip_grad_norm_(all_params, max_norm=1.0)
    opts['opt_g'].step(); opts['opt_u'].step(); opts['muon'].step()
```

---

### §1.61 New Config Keys (Steps 1–3)

```python
CFG.update({
    'beta_KL':            0.5,     # AdamW params KL weight
    'beta_SI_stiefel':    0.25,    # Stiefel SI weight (half of beta_KL)
    'beta_KL_warmup':     500,     # steps to anneal beta_KL from 0
    'lambda_prec':        0.001,   # precision entropy regulariser
    'lambda_vq':          0.01,    # VQ encoder commitment weight
    'surprise_threshold': 0.5,     # E_min_raw threshold for SA (Welford §1.68)
})
```

---

### §1.62 New Tests (Steps 1–3)

See PART 8 test inventory.

---

### §1.63 Fisher-Magnitude alpha_freeze — Principled Unit Protection (C1)

> **Replaces**: 85th percentile fixed threshold.

```
fisher_unit_i    = mean(fisher_dict entries for W_l[i] parameters)
freeze_threshold = μ_fisher + 1.5 × σ_fisher
alpha_freeze[i]  = 1 if fisher_unit_i > freeze_threshold else 0
```

Evaluated every 100 steps. New buffer: `fisher_unit: (N_max_l,) float32`.

---

### §1.64 Precision-Adaptive k_l — FEP Exploration/Exploitation (C2)

> **Replaces**: k_l=40 fixed.

```
k_l_eff = k_l_min + round((k_l_max - k_l_min) × U_epi_cal)
k_l_min = 10   (focused, confident routing)
k_l_max = 40   (exploratory, uncertain routing)
```

W_full pre-allocated to k_l_max=40; only k_l_eff entries used. No new parameters.

---

### §1.65 Q_BEAM-Weighted SSP Merge (C3)

> **Replaces**: fixed 0.7/0.3 merge ratio.

```
merge_weight = sigmoid(self._last_Q_BEAM_score)   ∈ (0, 1)
r_lista ← (1 - merge_weight) × parent + merge_weight × r_lista
```

New attribute: `_last_Q_BEAM_score: float on CUN`, updated in lista_forward.

---

### §1.66 U_meta-Adaptive Beam Width B (C4)

> **Replaces**: fixed B=2.

```
B_eff = max(1, round(1 + U_meta × (B_max - 1)))
B_max = 3   (cfg: beam_B_max)
```

At U_meta=0: B_eff=1 (no beam, saves 0.25% overhead). Average overhead decreases vs fixed B=2.

---

### §1.67 Fisher-Scaled Consolidation Rates (C5)

> **Replaces**: fixed alpha_consol=0.001, alpha_micro=0.0001.

```
alpha_effective = alpha_base / (1.0 + fisher_unit[nearest])
delta = alpha_effective × (k_rule − μ_c_l[nearest])
```

Applies to both `consolidate_arc_to_cnep` (§1.37) and `micro_consolidate_arc` (§1.54).

---

### §1.68 Welford E_min Surprise Detection (C6)

> **Replaces**: fixed surprise_threshold=0.5.

```
bank._Emin_n   += 1
delta           = E_min_raw - bank._Emin_mean
bank._Emin_mean += delta / bank._Emin_n
bank._Emin_var  += delta * (E_min_raw - bank._Emin_mean)

sigma_Emin = sqrt(bank._Emin_var / bank._Emin_n + 1e-8)
surprise   = E_min_raw > bank._Emin_mean + 2.0 × sigma_Emin
```

New buffers: `_Emin_mean, _Emin_var, _Emin_n` (3 values) on CFBank.

---

### §1.69 Welford-Based Spawn Threshold (C11)

> **Replaces**: fixed spawn_threshold E_min>3.0.

```
spawn_condition = E_min_raw > bank._Emin_mean + 2.5 × sigma_Emin
```

Reuses identical Welford buffers from §1.68. 2.5σ (vs 2.0σ for archive) — spawning more selective.

---

### §1.70 U_epi-Gated k-Shot Accumulation (C7)

> **Replaces**: tau_proto=0.6 fixed cosine threshold.

```
gate_proto = (U_epi_cal < 0.4) AND (cosine_sim(x_c, μ_c_l[idx]) > tau_proto_min=0.4)
```

Accumulates only when routing is confident AND similarity is adequate.

---

### §1.71 Think-Budget Natural Planning Timeout (C8)

> **Removes**: N_stuck config key entirely.

The think-token budget K_think is the natural timeout. Each subgoal gets the full think budget; if V never improves, auto-POP (§1.52) fires when budget exhausted.

---

### §1.72 Learned MC-1 Calibration Scale (C9)

> **Replaces**: fixed scale 0.15 in Welford normalisation.

```python
# BEFORE: U_epi_cal = σ(0.15 × (U_epi_raw - μ_U) / (σ_U + ε) + 0.5)
# AFTER:
U_epi_cal = σ(exp(log_cal_scale) × (U_epi_raw - μ_U) / (σ_U + ε) + 0.5)

log_cal_scale: scalar nn.Parameter, init log(0.15)≈-1.9, opt_g
L_cal = 0.001 × (exp(log_cal_scale) - 0.15)²   # regulariser
```

---

### §1.73 Learned r_lista Blend Alpha (C10)

> **Replaces**: fixed 0.8/0.2 warm-start blend.

```python
blend_alpha = exp(log_blend_alpha).clamp(0.5, 0.95)
r_lista^t ← blend_alpha × r_lista^{t-1} + (1-blend_alpha) × r_seed^t

log_blend_alpha: scalar nn.Parameter, init log(0.8)≈-0.223, opt_g
```

---

### §1.74 U_epi-Adaptive DCG+ Commit Threshold (C12)

> **Replaces**: fixed commit_score threshold=0.4.

```python
commit_threshold = max(0.1, 1.0 - U_epi_cal)
commit_condition = commit_score > commit_threshold
```

At U_epi=0.35 (certain): threshold=0.65. At U_epi=0.65 (uncertain): threshold=0.35.

---

### §1.75 Adaptive Rule Utility Decay (D5)

> **Replaces**: fixed 0.999999 per-token decay.

```python
decay_k = 0.999999 × (1.0 - 0.0001 × self._last_u_temporal)
rule_util[k] *= decay_k
```

High U_temporal → faster decay (half-life ~69K tokens). Low → slower (~693K tokens).

---

### §1.76 Config Changes Summary (C1–C12 + D3 + D5)

```python
CFG_ABLATION_605.update({
    'k_l_min':            10,
    'k_l_max':            40,
    'beam_B_max':         3,
    'tau_proto_min':      0.4,
    'episodic_rule_cache_n': 256,   # was 64
    # REMOVED (now emergent):
    # 'surprise_threshold'    → §1.68 Welford
    # 'ssp_stuck_threshold'   → §1.71 think-budget
    # 'ssp_merge_alpha'       → §1.65 Q_BEAM-weighted
    # 'spawn_threshold'       → §1.69 Welford
})
```

---

### §1.77 New Parameters Summary (C1–C12)

| Parameter | Shape | Init | Group | Replaces |
|---|---|---|---|---|
| `fisher_unit` | (N_max_l,) float32 | zeros | buffer | 85th-pct alpha_freeze |
| `_last_Q_BEAM_score` | float on CUN | 0.0 | — | ssp_merge_alpha=0.7 |
| `_Emin_mean` | float on CFBank | 0.0 | buffer | surprise_threshold=0.5 |
| `_Emin_var` | float on CFBank | 0.0 | buffer | spawn_threshold=3.0 |
| `_Emin_n` | int on CFBank | 0 | buffer | (Welford count) |
| `log_cal_scale` | scalar | log(0.15) | opt_g | MC-1 fixed scale 0.15 |
| `log_blend_alpha` | scalar | log(0.8) | opt_g | r_lista blend 0.8 |

---

## PART 2: CODE CHANGES

The v6.0.9 base implementations (ComplexEmbedding, ComplexLRU, TitansComplexMemory, ComplexHierarchicalOCNEncoder, CoactivationRegister, AlphaHistogram, CFBank, ComplexGATLayer, HopfieldRetrieval, CFL5Layer, IterativeRefinementModule, ComplexUnitaryDenoisingNet, TelescopingMemory, SurpriseArchive, ExemplarDormancyBuffer, DynamicLocalBank, CFLNModel, SIRegularizer, MuonOptimizer, train_step, generate_cfln_dcg_plus, generate_cfln_ctp, compute_ctp_loss) are archived in `docs/archive/CFLN_v609_Master_Spec.md`. The current authoritative implementations are in `src/cfln/` with the following v7.0→v9.0 changes reflected below.

### 2.1 CFBank.__init__ Additions (v7.0 through v9.0)

```python
# §1.30+1.41 Phase binding
self.log_lam_bind   = nn.Parameter(torch.tensor(-3.0))
self.log_sigma_bind = nn.Parameter(torch.tensor(0.693))   # log(2.0)

# §1.33 Goal context
self.register_buffer('g_c', torch.zeros(d_c, dtype=torch.cfloat))
self.W_goal_detect = nn.Parameter(torch.zeros(1, d_c))
self.log_lam_goal  = nn.Parameter(torch.tensor(-3.0))

# §1.35 Role binding
R_roles = cfg.get('n_roles', 8)
_raw = torch.randn(R_roles, d_c, dtype=torch.cfloat)
_Q, _ = torch.linalg.qr(_raw.T)
self.role_vecs    = nn.Parameter(_Q.T[:R_roles].contiguous())
self.log_lam_role = nn.Parameter(torch.tensor(-3.0))

# §1.36 Verbatim spans (buf_L1_ids already in v6.0.9 base)

# §1.43 SE-1: k-shot refinement
self.register_buffer('_proto_count', torch.zeros(N_max_l, dtype=torch.int32))
self.register_buffer('_proto_sum',   torch.zeros(N_max_l, d_c, dtype=torch.cfloat))

# §1.55 COMP-H: Hadamard composition
self.log_lam_composition = nn.Parameter(torch.tensor(-3.0))

# §1.59 VQ-Telescope buffers
self.register_buffer('buf_L1_w_full', torch.zeros(K_L1, N_max_l, dtype=torch.float32))
self.register_buffer('buf_L2_w_full', torch.zeros(K_L2, N_max_l, dtype=torch.float32))
self.register_buffer('buf_L3_w_full', torch.zeros(K_L3, N_max_l, dtype=torch.float32))

# §1.63 Fisher unit importance
self.register_buffer('fisher_unit', torch.zeros(N_max_l, dtype=torch.float32))

# §1.68 Welford E_min statistics
self._Emin_mean = 0.0; self._Emin_var = 0.0; self._Emin_n = 0

# REMOVE from v6.0.9 base: mask_embed (no longer a bank parameter)
```

### 2.2 CFLNModel.__init__ — W_bridge Change

```python
# REMOVE (v5.9.7):
# self.register_buffer('W_rc_bridge', W_bridge_init)

# ADD (v9.0):
self.W_rc_bridge = nn.Parameter(W_bridge_init)   # trained via L_bridge §1.50
```

### 2.3 CUN.__init__ Additions

```python
# §1.31 Two-key ARC: rule_K shape (N, d_c) → (N, 2*d_c)
self.register_buffer('rule_K', torch.zeros(N, 2*d_c, dtype=torch.cfloat))
self.log_alpha_arc = nn.Parameter(torch.tensor(0.0))

# §1.34 U_meta_v4 (BASE; replaced by §1.58 precision in v9.0):
# log_w_meta and _log_w_rec REMOVED — replaced by:
self.sigma_sq_buffer    = [1.0] * 5          # plain list, reset per session
self.log_precision      = [0.0] * 5          # plain list, persists across sessions
self._precision_active  = [False] * 5

# §1.39 SSP
self._goal_stack  = []
self._stuck_count = []   # §1.52 PLAN-B
self._v_prev      = []

# §1.40 STELA
self.tau_smooth = nn.Parameter(torch.tensor(0.1))

# §1.47 Beam
self.eps_beam_scale = nn.Parameter(torch.tensor(0.1))

# §1.46 Q-BEAM optional weights
self.log_w_beam = nn.Parameter(torch.zeros(3))

# §1.65 Q-BEAM score for SSP merge
self._last_Q_BEAM_score: float = 0.0

# §1.72 Learned calibration scale
self.log_cal_scale = nn.Parameter(torch.tensor(-1.897))   # log(0.15)

# §1.73 Learned blend alpha
self.log_blend_alpha = nn.Parameter(torch.tensor(-0.223))  # log(0.8)
```

### 2.4 CFL5Layer.forward — Complete v9.0 Ordering

```python
def forward(self, x_c, training=True, ...):

    # 1. GOAL CONTEXT (§1.33)
    x_c_mean = x_c.mean(0)
    if not bank._in_hypo_mode and not bool(bank._goal_stack_frozen):
        g_t = sigmoid(bank.W_goal_detect @ x_c_mean.real)
        bank.g_c = (g_t * x_c_mean + (1-g_t) * bank.g_c).detach()
    x_c_eff = x_c + exp(bank.log_lam_goal) * bank.g_c.unsqueeze(0)

    # 2. CNEP ROUTING with adaptive k_l (§1.64)
    k_l_eff = k_l_min + round((k_l_max - k_l_min) * bank._u_epistemic_last)
    E_l, sel_l, s_l = route(x_c_eff, bank, k_l=k_l_eff)
    E_min_raw  = E_l.min(dim=-1).values.mean().item()
    H_route_raw = -(s_l * (s_l+1e-9).log()).sum(-1).mean().item()

    # 3. k-SHOT CENTROID REFINEMENT (§1.43/§1.70)
    for sel_idx in sel_l:
        _refine_centroid_if_young(bank, sel_idx, x_c_mean, cfg)

    # 4. W_full enrichment — B_bind (§1.30/§1.41)
    phi_sel  = angle(bank.H_c_l[sel_l].mean(dim=(-2,-1)))
    phi_diff = phi_sel.unsqueeze(0) - phi_sel.unsqueeze(1)
    sigma_sq = exp(2.0 * bank.log_sigma_bind)
    B_bind   = exp(-(phi_diff**2) / sigma_sq)
    W_full[:k_l,:k_l] += exp(bank.log_lam_bind) * B_bind

    # 5. W_full enrichment — B_role (§1.35)
    alpha_role = softmax((bank.mu_c_l[sel_l] @ bank.role_vecs.conj().T).real / d_c**0.5, dim=-1)
    B_role     = alpha_role @ alpha_role.T
    W_full[:k_l,:k_l] += exp(bank.log_lam_role) * B_role

    # 5b. W_full enrichment — B_comp (§1.55)
    B_comp = B_bind * B_role
    W_full[:k_l,:k_l] += exp(bank.log_lam_composition) * B_comp

    # 6. CS-GAT aggregation
    h_filt = cs_gat(W_full, psi_all, K_CHEBY)

    # 7. Lyapunov goal proxy (§1.48)
    r_lista_goal_proxy = _get_goal_proxy(self, bank, cun)

    # 8. lista_forward with all kwargs
    h_N, meta_info = cun.lista_forward(
        x_c, hopfield=hopfield, bank=bank,
        u_temporal=u_temporal_val, u_hypo=bank._u_hypo,
        r_lista_goal=r_lista_goal_proxy,
        E_min_raw=E_min_raw, H_route_raw=H_route_raw)

    # 9. L_bridge (§1.50)
    if training and hasattr(self, 'W_rc_bridge'):
        rho_weighted = (s_w.unsqueeze(-1) * bank.rho_l[sel_bridge]).sum(0)
        r_seed = self.W_rc_bridge @ rho_weighted
        with no_grad():
            r_seed_target = (cun.U1.conj() @ x_c_mean)[:r_seed.shape[0]]
        self._last_L_bridge = (r_seed_target.detach() - r_seed).norm()**2
```

### 2.5 CUN.lista_forward — Key v7.0–v9.0 Changes

```python
def lista_forward(self, x_c, ..., r_lista_goal=None,
                  E_min_raw=None, H_route_raw=None):

    # §1.31: relational key (cached every 32 tokens)
    if self._phi_rel_step % 32 == 0 and bank:
        self._phi_rel_cache = top_eigenvec(H_seq_sub).detach()
    self._phi_rel_step += 1

    # §1.32: HYPO mode active beam selection
    r_lista_active = (self._r_lista_hypo if self._in_hypo_mode else self.r_lista)

    # §1.40 STELA: smooth threshold in K-iteration loop
    for k in range(N_adaptive):
        h = h * sigmoid((h.abs() - tau) / self.tau_smooth.clamp(min=1e-3))

    # §1.45 SE-3: reservoir augmentation
    x_c_recon = self.U2 @ h_N + self.W_dec_res @ bank.rho_l[sel_l].mean(0)

    # §1.46–§1.47: Q-BEAM + beam search (during think tokens, adaptive B §1.66)
    in_think = getattr(self, '_in_think_mode', False)
    B_eff = max(1, round(1 + U_meta_prev * (B_max - 1)))
    if in_think and B_eff >= 2:
        r_lista_b2 = self.r_lista + self.eps_beam_scale.abs() * randn_like(self.r_lista)
        h_b2 = self._lista_inner(x_c, r_lista_b2, N_adaptive, tau)
        Q_b1 = compute_Q_beam(h_N, self.r_lista, r_lista_goal, ...)
        Q_b2 = compute_Q_beam(h_b2, r_lista_b2, r_lista_goal, ...)
        w = softmax(stack([Q_b1, Q_b2]), dim=0)
        h_N = w[0]*h_N + w[1]*h_b2
        self.r_lista = (w[0]*self.r_lista + w[1]*r_lista_b2).detach()
        self._last_Q_BEAM_score = float(max(Q_b1, Q_b2))

    # §1.58: precision-weighted U_meta (replaces log_w_meta softmax)
    update_precision(signals, self.sigma_sq_buffer, self.log_precision,
                     self._precision_active, is_hypo_active, is_hopfield_active)
    prec    = torch.exp(torch.tensor(self.log_precision))
    U_meta  = (prec * signals_t).sum() / (prec.sum() + 1e-8)

    # §1.32: ARC writes suppressed in HYPO
    if not self._in_hypo_mode:
        K_new = cat([k_concept_new, q_rel_new])   # (2×d_c,) dual key §1.31
        # ... NR-1/NR-2/NR-3 write logic ...
    
    # §1.75: adaptive rule decay
    decay_k = 0.999999 * (1.0 - 0.0001 * self._last_u_temporal)
    self.rule_util[:n_r] *= decay_k
```

### 2.6 CFLNModel.reset_for_inference — Full v9.0 Order

```python
def reset_for_inference(self):
    # 1. Consolidate ARC → μ_c_l BEFORE clearing (§1.37)
    consolidate_arc_to_cnep(self.bank, self.diff_aux.cun)

    # 2. Persist SurpriseArchive (§1.38, optional)
    if self.cfg.get('persist_archive', False):
        self.bank.surprise_archive.save_state(self.cfg['archive_path'])

    # 3. SSP + Lyapunov state (§1.39/§1.52)
    self.diff_aux.cun._goal_stack  = []
    self.diff_aux.cun._stuck_count = []
    self.diff_aux.cun._v_prev      = []

    # 4. HYPO state (§1.32)
    self.bank.g_c.zero_()
    self.bank._in_hypo_mode  = False
    self.bank._r_lista_hypo  = None
    self.bank._u_hypo        = 0.0

    # 5. Precision reset (§1.58) — sigma_sq reset, log_precision persists
    self.diff_aux.cun.sigma_sq_buffer   = [1.0] * 5
    self.diff_aux.cun._precision_active = [False] * 5

    # 6. Standard resets
    self.diff_aux.cun.reset_lista_reservoir()
    # ... ARC utility reset, sparse cache reset, etc. ...
```

### 2.7 train_step_v900 — Additional Loss Terms

```python
# §1.50 W_bridge
for layer in model.layers:
    if hasattr(layer, '_last_L_bridge'):
        loss += cfg.get('lambda_bridge', 0.1) * layer._last_L_bridge

# §1.44 SE-2 MDLM (Stage 0 only)
if stage == 'stage0' and cfg.get('p_mask', 0) > 0:
    loss += cfg['lambda_mlm'] * L_mlm

# §1.56 ROB-L/S
young_mask = (bank.alpha_freeze[:bank.n_l] == 0).bool()
if young_mask.any():
    loss += cfg.get('lambda_lipschitz', 0.001) * bank.log_alp_l[:bank.n_l][young_mask].mean()
loss += cfg.get('lambda_sigma_reg', 0.001) * torch.exp(-bank.log_sigma_bind)

# §1.47 TS-1 diversity
if hasattr(cun, '_last_beam_diversity'):
    loss += cfg.get('lambda_diversity', 0.01) * (-cun._last_beam_diversity**2)

# §1.57 Fisher-KL + SI-Stiefel
loss += cfg['beta_KL']         * compute_L_KL(model, fisher, cfg)
loss += cfg['beta_SI_stiefel'] * compute_L_SI_stiefel(model, si, cfg)

# §1.58 Precision regulariser
loss += cfg.get('lambda_prec', 0.001) * compute_L_prec(model)

# §1.59 VQ commitment
if hasattr(bank, '_last_L_vq') and bank._last_L_vq is not None:
    loss += cfg.get('lambda_vq', 0.01) * bank._last_L_vq

# Note: §1.42 L_recon REMOVED (VQ-Telescope §1.59 replaces it)
```

### 2.8 Micro-Consolidation Call Site

```python
# ADD at end of _update_telescoping():
if hasattr(self, 'diff_aux') and hasattr(self.diff_aux, 'cun'):
    micro_consolidate_arc(self.bank, self.diff_aux.cun, self.cfg)
```

### 2.9 SurpriseArchive — VQ Simplified

```python
class SurpriseArchive:
    def add_vq(self, buf_L1_ptr, E_min_raw):
        """Replaces add() with embedding. No cosine dedup (VQ codes naturally diverse)."""
        if len(self.archive) >= self.N_archive:
            min_idx = min(range(len(self.archive)),
                         key=lambda i: self.archive[i]['score'])
            self.archive[min_idx] = {'ptr': buf_L1_ptr, 'score': E_min_raw}
        else:
            self.archive.append({'ptr': buf_L1_ptr, 'score': E_min_raw})
```

---

## PART 3: OPTIMIZER CHANGES

```python
stiefel_ids = {
    id(bank.W_l), id(bank.W_p),           # Muon (Stiefel Cayley retraction)
    id(cun.U1),   id(cun.U2),             # NEVER trained (fixed unitaries)
    id(bank.role_vecs),                    # opt_g AdamW (excluded from Stiefel — (R,d_c) shape)
    id(model.W_rc_bridge),                 # opt_g AdamW (excluded from Stiefel — ESN conservative)
}

# opt_g (AdamW) auto-covers via named_parameters():
#   bank: log_lam_bind, log_sigma_bind, W_goal_detect, log_lam_goal,
#         role_vecs, log_lam_role, log_lam_composition
#   cun:  log_alpha_arc, tau_smooth, eps_beam_scale, log_w_beam,
#         log_cal_scale, log_blend_alpha
#   model: W_rc_bridge  ← NEW in v9.0

# Muon covers: W_l, W_p, theta_cheby (CS-GAT), W_gate_mem, W_decompress (REMOVED in v9.0)
# Note: W_decompress_L1 removed (§1.59); Muon list shrinks accordingly.
```

---

## PART 4: CONFIG

```python
CFG_ABLATION_900 = {
    # ── v6.0.9 base keys ──
    'd_c': 256, 'vocab_size': 32768, 'n_l': 2112, 'n_p': 128, 'L': 6,
    'n_heads_gat': 4, 'd_e_l': 64, 'd_e_p': 64,
    'd_ssm_fast': 32, 'S_f': 32, 'C_chunk': 512,
    'per_sequence_memory': True,
    'K_L1': 128, 'K_L2': 32, 'K_L3': 32, 'N_archive': 256,
    'surprise_warmup_chunks': 32,
    'eta_titans': 0.01, 'theta_decay_init': 0.99,
    'null_threshold_init': 0.95, 'k_null': 50.0, 'beta_null_aux': 0.01,
    'domain_alpha': 0.90, 'domain_mag_alpha': 0.99, 'domain_threshold_init': 3.0,
    'rope_L_train': 2048, 'rope_L_target': 1_048_576,
    'T_diff': 50, 'n_fourier': 8,
    'c_SI': 0.5, 'rho_SI': 0.999, 'beta_SI': 3.0, 'N_dormant': 512,
    'D_g': 4, 'K_hebb': 8, 'K_stats': 4, 'D_bptt': 4, 'n_layers_diff': 2,
    'N_iter_refine': 8, 'N_hop_refine': 4,
    'use_hopfield_refine': True, 'use_escape_refine': True, 'lambda_lista': 0.1,
    'gradient_checkpointing': True, 'grad_accum_steps': 2,
    'si_warmup_steps': 500,
    'lr_muon': 1e-3, 'lr_muon_diff': 1e-4, 'lr_persist': 1e-6,
    'lr_start': 1e-3, 'lr_end': 1e-4,
    'lambda_compress': 0.001,   # v9.0 fix: reduced from 0.01
    'merge_sample': 32,
    'delta_stuck': 0.1, 'delta_min': 0.01, 'epsilon_esc': 0.05,
    'schedule_grad_clip': 0.5,
    'd_r_node': 8, 'rho_node': 0.95,
    'd_r_lista': 32, 'rho_lista': 0.99,
    'rho_fast': 0.85, 'rho_mid': 0.90, 'rho_slow': 0.99,
    'think_threshold': 0.5, 'max_think_tokens': 64, 'tau_think': 0.5,
    'sparse_code_cache_K': 32, 'episodic_rule_cache_n': 256,   # v9.0: 64→256
    'lista_min_ratio': 0.25, 'lista_convergence_ratio': 0.5,
    'si_proactive_threshold': 0.8, 'proactive_cooldown': 20,
    'T': 256, 'B': 8,
    'memory_thresholds': {
        'eps_s': 0.01, 'eps_p': 0.001, 'eps_split': 0.5,
        'eps_merge': 0.95, 'r_reset': 0.3, 'eps_H': 1e-4
    },
    # ── v7.0 keys ──
    'sigma_bind': 1.0, 'arc_dual_key': True,
    'hypo_start_id': 32770, 'hypo_end_id': 32771,
    'use_goal_context': True,
    # ── v8.0 keys ──
    'n_roles': 8, 'tau_consol': 3.0, 'alpha_consol': 0.001,
    'persist_archive': False, 'archive_path': 'archive.pt',
    'push_goal_id': 32772, 'pop_goal_id': 32773,
    'ssp_max_depth': 4,
    'psd_apply_every': 10,
    # ── Addendum keys ──
    'K_proto_max': 10, 'alpha_young': 0.1,
    'p_mask': 0.15, 'lambda_mlm': 0.3,
    'lambda_diversity': 0.01,
    # ── v9.0 W_bridge ──
    'lambda_bridge': 0.1,
    # ── v9.0 micro-consolidation ──
    'alpha_micro': 0.0001,
    # ── v9.0 ROB-L/S ──
    'lambda_lipschitz': 0.001, 'lambda_sigma_reg': 0.001,
    # ── v9.0 Step 1: Fisher-KL ──
    'beta_KL': 0.5, 'beta_SI_stiefel': 0.25, 'beta_KL_warmup': 500,
    # ── v9.0 Step 2: Precision U_meta ──
    'lambda_prec': 0.001,
    # ── v9.0 Step 3: VQ-Telescope ──
    'lambda_vq': 0.01,
    # ── v9.0 adaptive params ──
    'k_l_min': 10, 'k_l_max': 40,
    'beam_B_max': 3,
    'tau_proto_min': 0.4,
    # ── REMOVED keys (heuristics replaced by adaptive) ──
    # 'surprise_threshold' → §1.68 Welford
    # 'ssp_stuck_threshold' → §1.71 think-budget
    # 'ssp_merge_alpha' → §1.65 Q_BEAM-weighted
    # 'spawn_threshold' → §1.69 Welford
    # 'lambda_recon' → §1.59 VQ-Telescope removes W_decompress
    # 'tau_proto' → replaced by 'tau_proto_min'
    # 'beam_B' → replaced by adaptive B_eff
}
```

---

## PART 5: TOKENIZER

```python
def extend_tokenizer_v9(tok):
    tok.add_special_tokens([
        '<think>', '</think>',
        '<hypo>',  '</hypo>',
        '<push_goal>', '</push_goal>'
    ])
    return tok, {
        'think_start': tok.token_to_id('<think>'),
        'think_end':   tok.token_to_id('</think>'),
        'hypo_start':  tok.token_to_id('<hypo>'),
        'hypo_end':    tok.token_to_id('</hypo>'),
        'push_goal':   tok.token_to_id('<push_goal>'),
        'pop_goal':    tok.token_to_id('</push_goal>'),   # </push_goal> = POP signal
    }
# vocab_size_extended = vocab_size + 6
```

---

## PART 6: TRAINING PIPELINE

### Stage 0 — LM Warmup

- SE-2 MDLM masking active (p_mask=0.15, λ_mlm=0.3)
- L_bridge accumulates for W_rc_bridge from first forward
- SE-1 k-shot centroid refinement active from first token
- VQ-Telescope update replaces W_compress update
- Fisher accumulation begins (builds slowly; Stiefel SI provides coverage from step 0)

### Stage 1 — PSC Pre-Training (10% budget)

- PSC three-term loss (L_improve + L_economy + L_predictive)
- W_bridge now improves PSC by seeding better r_lista states

### Stage 2 — RPP-STaR Trace Generation (offline)

RPP trace mix:
```python
TRACE_MIX = {
    'flat_ctp':         0.65,   # standard <think>...</think>
    'hypo_branch':      0.20,   # with <hypo>...</hypo> inside think
    'ssp_hierarchical': 0.15,   # with <push_goal>...<pop_goal> inside think
}
```

**Hypothetical trace template** (20% of traces — analogy benchmarks, Raven's matrices):
```python
HYPO_TRACE_TEMPLATE = """<think>
What do I know about {source_A} and {source_B}?
<hypo>Suppose {source_A} didn't have property X. Would {relation} still hold?</hypo>
The consequence divergence is {u_hypo_value:.2f} — hypothesis is {significant}.
Therefore {target_C} : {answer_D} by the same {relation}.
</think>{answer_D}"""
```

**SSP hierarchical trace template** (15% of traces — multi-step decomposition):
```python
SSP_TRACE_TEMPLATE = """<think>
<push_goal>Understand: {subgoal_1}</push_goal>
{reasoning_subgoal_1}
<pop_goal/>
<push_goal>Apply to: {subgoal_2}</push_goal>
{reasoning_subgoal_2}
<pop_goal/>
{synthesis}
</think>{answer}"""
```

Tasks suited for SSP traces: multi-hop question answering, logical deduction, analogical completion.

### Stage 3 — SFT on RPP Traces

- W_bridge L_bridge continues training
- τ_think = 0.5

### Stage 4 — GRPO Fine-Tuning

- Beam search active (B_eff adaptive per §1.66)
- G=8 rollouts, β=0.1 KL penalty, π_ref = frozen SFT checkpoint
- τ_think = 0.0

**Combined GRPO reward** (v7.0 §5.3 + v8.0 §6):
```python
R_ppl     = CE_baseline_t - CE_thinking_t          # intrinsic perplexity reduction
R_analogy = 1.0 if completion follows A:B::C:? pattern else 0.0
R_ssp     = 1.0 if ssp_stack_balanced(completion_ids, PUSH_GOAL_ID, POP_GOAL_ID) else -0.2
# ssp_stack_balanced: count(push)==count(pop) AND no pop on empty stack

R = R_ppl + 0.3 * R_analogy + 0.2 * R_ssp
```

**Analogy evaluation pairs** (for A94/A95 ablations):
```python
ANALOGY_EVAL_PAIRS = [
    # (source_A, source_B, target_C, correct_D)
    ('ice', 'water', 'wax', 'liquid'),
    ('king', 'queen', 'man', 'woman'),
    ('hot', 'cold', 'fast', 'slow'),
    ('bark', 'tree', 'skin', 'body'),
    # Add 20+ pairs for statistical significance
]
```

---

## PART 7: CHECKPOINT PROTOCOL

### Saving

```python
ckpt.update({
    'bank_g_c':         model.bank.g_c.cpu(),
    'bank_u_hypo':      model.bank._u_hypo,
    'bank_proto_count': model.bank._proto_count.cpu(),
    'bank_proto_sum':   model.bank._proto_sum.cpu(),
    'W_rc_bridge':      model.W_rc_bridge.data.cpu(),   # now in state_dict as Parameter
    # VQ-Telescope buffers saved automatically via state_dict (register_buffer)
    # sigma_sq_buffer and _precision_active: save as lists in ckpt
    'cun_sigma_sq_buffer':   model.diff_aux.cun.sigma_sq_buffer,
    'cun_log_precision':     model.diff_aux.cun.log_precision,
    'bank_Emin_mean':        model.bank._Emin_mean,
    'bank_Emin_var':         model.bank._Emin_var,
    'bank_Emin_n':           model.bank._Emin_n,
})
```

### v6.0.9 → v9.0 Migration Notes

1. **W_rc_bridge**: was `register_buffer`, now `nn.Parameter`. Use `load_state_dict(strict=False)` or manually copy `ckpt['W_rc_bridge']` to `model.W_rc_bridge.data`.

2. **rule_K shape**: was `(N_rules, d_c)`, now `(N_rules, 2*d_c)`. On load (v700 §6.2):
```python
if 'model_state' in ckpt:
    old_rule_K = ckpt['model_state'].get('diff_aux.cun.rule_K')
    if old_rule_K is not None and old_rule_K.shape[-1] == d_c:
        pad = torch.zeros(*old_rule_K.shape[:-1], d_c,
                          dtype=old_rule_K.dtype, device=old_rule_K.device)
        ckpt['model_state']['diff_aux.cun.rule_K'] = torch.cat([old_rule_K, pad], dim=-1)
        print("v6.0.9→v7.0 checkpoint migration: rule_K padded to 2×d_c")
```
Zero-init relational half will train from scratch.

3. **sigma_sq_buffer / log_precision**: not in state_dict (plain lists). On load: init to `[1.0]*5` and `[0.0]*5` respectively if not in checkpoint.

4. **buf_L1/L2/L3_w_full**: new `register_buffer`s. Not present in v6.0.9 checkpoints — initialized to zeros automatically (cold start acceptable).

5. **log_w_meta / _log_w_rec**: were saved in v7.0/v8.0 checkpoints. Drop silently when loading into v9.0 model (`strict=False`).

6. **W_compress/W_decompress**: drop when loading v8.0 checkpoint into v9.0 model.

---

## PART 8: TEST INVENTORY

### Implemented (v6.0.9 base — 68 tests)

All function names from `tests/test_utils.py` (and the v6.0.9 Part 8 spec):

```
test_stiefel_constraints_after_update        test_W_enc_res_is_buffer_not_param
test_W_ri_is_buffer_not_param                test_update_res_flag_prevents_multi_decay
test_nr1_trigger_b_uses_bank                 test_mc2_log_w_rec_updates
test_mc3_u_temporal_buffer                   test_u_epi_calibration_buffers
test_no_dead_params_after_global_removal     test_dual_trigger_increases_write_rate
test_topk_rule_retrieval                     test_u_temporal_init
test_two_tier_no_global                      test_selective_lru_gating
test_arc_merge_reduces_entries               test_sa_dedup_no_duplicate
test_psc_loss_shapes                         test_rpp_trace_improves_ce
test_star_rpp_acceptance_rate                test_think_id_survives_checkpoint_load
test_lam_sg_lam_h_bounded                    test_hseq_norm_bounded_by_mx
test_forward_rejects_empty_sequence          test_ctp_mode_isolation
test_hseq_device_correctness                 test_prev_sel_l_reset_at_document_boundary
test_wll_cache_cleared_by_train_step         test_dcg_pos_offset_revision_uses_zero
test_lam_seq_gat_receives_gradient           test_lambda_hebb_receives_gradient
test_dcg_pos_offset_correct                  test_expand_vocabulary_expands_bias
test_expand_vocabulary_no_double_call        test_ctp_think_loop_single_processing
test_w_commit_receives_gradient              test_titans_chunk_accum_cleared_on_thinking_exit
test_expand_vocabulary_adds_two_tokens       test_thinking_mode_gates_h_cache
test_titans_m_not_updated_in_thinking_mode   test_compute_ctp_loss_weights
test_thinking_mode_reset_on_reset_for_inference  test_z_val_in_info_dict
test_logits_in_aux_dict                      test_rule_cache_uses_semantic_key
test_w_commit_param_exists                   test_adaptive_lista_depth_reduces_iters
test_u_epistemic_range                       test_sparse_code_cache_fills_and_shifts
test_hopfield_confidence_stored              test_u_meta_v2_backward_compat
test_sequential_hebbian_updates              test_proactive_snapshot_uses_u_epistemic
test_seq_mode_reset_after_nonsq_training     ~~test_W_rc_bridge_is_buffer_not_param~~ (replaced by test_w_rc_bridge_is_parameter)
test_hcl_phase_used_in_psi_for              test_s_norm_last_stored_in_titans
test_multiscale_rho_assertion                test_multiscale_rho_values
test_salience_gate_scales_reservoir          test_memory_gate_independent
test_rc_bridge_shapes                        test_U_meta_gate_suppresses_warmstart
test_domain_confidence_range                 test_reservoir_phase_unit_magnitude
test_node_reservoir_backward_compat          test_lista_warmstart_backward_compat
test_lista_warmstart_updates_r_lista         test_prune_remaps_rho_l
test_crope_magnitude_preserved               test_complex_layer_norm_phase_preserved
test_verify_stiefel_after_init               test_entmax15_sums_to_one
```

Note: `test_W_rc_bridge_is_buffer_not_param` is superseded by `test_w_rc_bridge_is_parameter` (§1.50). Net: 68 - 1 + 1 = 68 base tests.

### v7.0 Additional Tests (5 tests)

| Function | Section |
|---|---|
| `test_phase_kernel_psd` | §1.30 |
| `test_dual_key_arc_shape` | §1.31 |
| `test_goal_register_zeros_after_reset` | §1.32/§1.33 |
| `test_g_c_dtype` | §1.33 |
| `test_u_meta_v4_five_signals` | §1.34 |

### v8.0 Additional Tests (6 tests)

| Function | Section |
|---|---|
| `test_b_role_psd` | §1.35 |
| `test_role_vecs_not_in_muon` | §1.35 |
| `test_ssp_stack_push_pop` | §1.39/§1.65 |
| `test_ssp_max_depth` | §1.39 |
| `test_stela_continuity` | §1.40 |
| `test_consol_updates_mu` | §1.37 |

### Addendum Additional Tests (6 tests)

| Function | Section |
|---|---|
| `test_se1_kshot_centroid_update` | §1.43 |
| SE-3: reservoir augmentation changes x_c_recon norm | §1.45 |
| `test_q_beam_parameter_free` | §1.46 |
| `test_ts1_soft_select_weights_sum_to_one` | §1.47 |
| TS-3: r_lista_goal_proxy = U1 @ g_c.conj() shape correct | §1.48 |
| TS-4: F5=0.0 when _goal_stack empty | §1.49 |

### v9.0 New Tests (16 tests)

| Function | Section |
|---|---|
| `test_w_rc_bridge_is_parameter` (replaces `test_W_rc_bridge_is_buffer_not_param`) | §1.50 |
| `test_w_rc_bridge_not_in_muon` | §1.50 |
| `test_l_compress_gradient_flows` | §1.51 |
| test_lambda_compress_is_0001 | §1.51 |
| test_fisher_not_updated_for_frozen_units | §1.57 |
| test_fisher_accumulated_before_clip | §1.57 |
| `test_sigma_sq_buffer_init` | §1.58 |
| `test_precision_inactive_pathway` | §1.58 |
| `test_vq_buf_dtype` | §1.59 |
| `test_l_vq_gradient_to_encoder` | §1.59 |
| `test_b_comp_psd` (Schur product theorem) | §1.55 |
| test_micro_consolidate_calls_fisher_scaled | §1.54/§1.67 |
| test_welford_emin_mean_converges | §1.68 |
| test_adaptive_k_l_range (10 ≤ k_l_eff ≤ 40) | §1.64 |
| test_blend_alpha_clamp (0.5 ≤ blend ≤ 0.95) | §1.73 |
| test_L_bridge_has_gradient_to_W_bridge | §1.50 |

**Total**: 68 + 5 + 6 + 6 - 1 + 16 = **100 tests** (matches v9.0 target)

---

## PART 9: ABLATIONS SUMMARY

| ID | Description | Driver |
|---|---|---|
| A85–A90 | PSC/GRPO ablation series | Stage pipeline validation |
| A93 | 2-tier vs 3-tier CNEP | CL grade validation |
| A94 | HYPO ON/OFF | Counterfactual reasoning |
| A95 | Dual-key vs single-key ARC | Structural retrieval |
| A96 | Goal context ON/OFF | Planning coherence |
| A97 | RAH role binding ON/OFF | Compositionality |
| A98 | Verbatim spans ON/OFF | Precise recall |
| A99 | SSP vs flat CTP | Multi-step reasoning |
| A100 | STELA vs hard threshold | Adversarial robustness |
| A101 | CONSOL-1 ON/OFF | Cross-session knowledge |
| A102 | SE-1 k-shot vs 1-shot | Novel concept recognition |
| A103 | SE-2 MDLM ON/OFF | Few-shot completion |
| A104 | SE-3 reservoir ON/OFF | Novel domain sparse coding |
| A105 | Q-BEAM vs U_meta-only | Beam quality signal |
| A106 | TS-3 Lyapunov vs plain beam | Goal-directed trajectory |
| A107 | W_bridge trained vs fixed | L_bridge effect on r_lista |
| A108 | Compress gradient on/off | §1.51 fix on memory fidelity |
| A109 | PLAN-B timeout ON/OFF | Multi-step planning dead-ends |
| A110 | phi_rel D1 in Q_BEAM | Deduction benchmark |
| A111 | Micro-consolidation ON/OFF | Knowledge retention |
| A112 | B_comp Hadamard ON/OFF | SCAN compositional generalisation |
| A113 | ROB-L/S ON/OFF | Adversarial perturbation resistance |
| A114 | Fisher-KL vs SI (AdamW) | Forgetting across 10+ domains |
| A115 | Precision vs fixed log_w_meta | Metacognitive signal quality |
| A116 | VQ-Telescope vs W_compress | Semantic retrieval quality |

---

## PART 10: GRADE PROJECTIONS

| Dimension | v6.0.9 | v9.0 | Key v9.0 drivers |
|---|---|---|---|
| Continual Learning | A− | A | CONSOL-1 + MDLM robust representations |
| Catastrophic Forgetting | A− | A− | Fisher-KL (empirical validation pending) |
| Context Window | B+ | A | Verbatim spans + VQ-Telescope persistence |
| Performance | A− | A | PF1+PF2 + adaptive beam |
| Architecture | A− | A | THETA-GAMMA binding + complete cognitive loop |
| Implementation | B+ | A− | 99 tests; W_bridge fix closes design debt |
| Reasoning (multi-step) | B+ | A | Beam B_eff + Q-BEAM principled |
| Reasoning (planning) | C+ | B+ | SSP + Lyapunov goal-directed |
| Reasoning (deduction) | C | B | CSP arc-consistency + phi_rel D1 |
| Metacognition | A− | A− | Precision U_meta; U_epi↔CE unvalidated |
| Novel Rule Construction | B+ | A | CONSOL-1 + dual-key ARC |
| Analogical Thinking | C+ | A− | Phase binding + RAH roles + relational key |
| Knowledge Accumulation | C+ | B+ | CONSOL-1 + VQ-Telescope persistence |
| Compositionality | B− | B+ | RAH + B_comp Hadamard (SCAN) |
| Sample Efficiency | B | A− | SE-1+SE-2+SE-3+W_bridge trained |
| Interpretability | A− | A | Role vectors human-readable; W_bridge reveals reservoir mapping |
| Robustness | C+ | A− | STELA + σ_bind learned + ROB-L/S |
| Transfer Learning | B | A− | Role binding cross-domain + goal-directed |

**Overall: B+ (v6.0.9) → A (v9.0)**

---

## APPENDIX: OPEN QUESTIONS

### Memory & Retrieval

**OQ-TELEPOS-1** (v6.0.6): Telescoping L1 position-indexed skip ring math design specified (§1.6: 64-slot, high-surprise positions via Titans s_t) but not fully implemented in TelescopingMemory. Requires per-chunk surprise tracking, 90th-pct running threshold, retrieve_at_position() API.

**OQ-RC-1**: Fourier reservoir frequencies uniform (2πk/d_r). Natural language has rhythmic structure at ~3–5 (phrase), ~15–25 (sentence), ~50–100 (paragraph) tokens. Consider log-spaced or trainable frequencies initialized to linguistic scales. Deferred to Phase 5 ablation.

**OQ-RC-2**: Node reservoir update uses projection error. For units with very small activation, reservoir input may be numerically insufficient. Consider explicit activation-weight scaling.

**OQ-COMPRESS-1** (carried): Contrastive L_compress alternative. Deferred to A55.

### Continual Learning

**OQ-CONSOL-1** (v6.0.7, resolved v8.0 Y2): Rule→CNEP consolidation implemented in §1.37. Micro-consolidation added in §1.54. Cross-session persistence achieved.

**OQ-GLOBALTIER-1** (COMPLETED v6.0.8): CNEP global tier removed. Performance saving confirmed (21% per-token flop reduction). Ablation A93 confirms functional equivalence.

**OQ-LOWRANK-1** (v6.0.7): Low-rank CNEP `W_l = A_l × B` factorisation. B∈C^{r×d_c} shared (r=16), A_l∈C^{d_e_l×r} per-unit Stiefel. Expected: 37.5% CNEP flop reduction. Deferred for ablation A92.

### Performance

**OQ-PF1-1** (v6.0.8): CNEP activation-sorted early exit (inference). Sort n_l units by activation_freq_l; scan in batches of 256; exit when top-k stable. Expected: 40–50% CNEP inference reduction. Inference-only state complexity defers implementation.

**OQ-PF2-1** (v6.0.9): Batched apply_psd wired into training loop (each layer projects independently now). Expected: 4–6× faster PSD projection. `batched_apply_psd` defined but not yet wired into train_step call chain.

### Binding & Semantics

**OQ-BIND-1** (v7.0): Does B_bind co-activating coefficient (exp(log_lam_bind)) converge to positive values (phase binding beneficial) or negative (binding harmful)? If negative after 5K steps, B_bind should be removed.

**OQ-BIND-2** (v7.0): At what σ_bind value (exp(log_sigma_bind)) does B_bind stop being PSD-beneficial? Monitor log_sigma_bind during training; alert if σ < 0.5 (RBF becomes too sharp → near-zero binding for all but identical phases).

**OQ-RELKEY-1** (v7.0): Does the relational key k_rel (top eigenvec of H_seq_sub) produce stable representations across 32-token windows? If phi_rel flips sign (eigenvector sign ambiguity), rule retrieval may degrade. Consider: always orient phi_rel to have positive inner product with prev phi_rel.

### Reasoning & Planning

**OQ-HYPO-1** (v7.0): Does HYPO mode produce measurably different r_lista trajectories compared to standard thinking? If ‖r_lista_hypo − r_lista‖ stays near 0 throughout HYPO blocks, the branch mechanism is not functioning.

**OQ-GOAL-1** (v7.0): Does g_c converge to meaningful goal representations? Monitor cosine similarity between g_c at PUSH_GOAL and the eventual POP_GOAL r_lista. High similarity → goal was achieved.

**OQ-v600-1**: Does r_lista stabilise (low delta between consecutive thinking steps) on structured reasoning tasks? Validates LISTA chain CoT mechanism.

**OQ-v600-2**: Does `show_thinking=True` reveal coherent reasoning traces after STaR fine-tuning?

**OQ-v600-3**: At what think_threshold does CTP + DCG+ achieve optimal quality-vs-compute on held-out reasoning benchmark?

### Metacognition

**OQ-v598-2**: Does U_epistemic correlate better with actual prediction error (CE on held-out set) than U_meta? If r(U_epistemic, CE) > r(U_meta, CE): U_epistemic should replace unc_w in STI head.

**OQ-META-1** (carried): U_meta tensors ready. Combining U_meta and U_epistemic for adaptive compute gating is now feasible (v5.9.4+).

### Miscellaneous

**OQ-v596-2**: After I4 (RC bridge), do cosine similarity trajectories of W_rc_bridge @ rho_l[i] and r_lista converge over training? If similarity > 0.7 after 5K steps, bridge successfully unifies two scales.

**OQ-GATE-1** (carried): W_gate_mem under Muon (orthogonal rows). Consider moving to opt_g.

**OQ-v597-3**: After C2 (log_hop_blend), does alpha_b converge toward temporal (→0) or content (→1)? If content dominates (alpha_b < 0.3 after 5K steps), pursue Hopfield-seeded LISTA as full replacement for temporal warm start.

---

*END — CFLN v9.0 Master Specification*
*v6.0.9 base · R1–R4 (v7.0) · X/Y/Z/W (v8.0) · SE/TS/Q-BEAM (Addendum) · Steps 1–3 + C1–C12 (v9.0)*
*Single complex domain · THETA-GAMMA dual-oscillation binding · Precision metacognition · VQ-Telescope unified memory*
*May 2026*
