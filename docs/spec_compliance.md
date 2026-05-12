# CFLN Spec Compliance Checklist

**Source of truth**: `docs/CFLN_Master_Spec.md` (v9.0)
**Last audited**: 2026-05-11 (full file reads of all 10 source modules + gap analysis against v900_Master_Spec, Implementation_Plan, AI_Instructions)
**Status legend**: ✅ PASS · ❌ FAIL · ⚠️ PARTIAL · 🔍 NEEDS VERIFY

---

## How to use

Each row maps a spec requirement to an exact `file:line` location and gives a current status.
When you change a file, find its rows here and re-verify. A row is only PASS when the code
exactly matches the spec formula — not just "close".

---

## Part 0: Architecture Invariants

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| A1 | §OVERVIEW | `to_complex` only in ComplexEmbedding | `modules/embedding.py` | ✅ | All other modules receive cfloat |
| A2 | §OVERVIEW | `to_real` only at logit output | `modules/sti_head.py` | ✅ | No other `to_real` calls |
| A3 | §OVERVIEW | All internal tensors `torch.cfloat` | throughout | ✅ | Verified: no float32 in routing/LISTA/bank |
| A4 | §1.2 | Global tier removed (no n_g, W_g, mu_c_g) | `modules/bank.py` | ✅ | Absent from __init__ |
| A5 | §1.2 | n_l default = 2048 + 64 | `config.py:18` | ✅ | `n_l=2_112` |
| A6 | §1.3 | CRoPE only at CFL-5 residual + Titans Q | `modules/cfl5.py`, `modules/titans.py` | ✅ | Not applied before CNEP energy |
| A7 | §1.3 | `rope_base ≈ 5.25e6` NTK-scaled | `config.py:85` | ✅ | `rope_base=5_250_000.0` |
| A8 | CLAUDE.md | RMS norm only; never `F.layer_norm` on complex | throughout | ✅ | `complex_layer_norm` used |
| A9 | §1.12 | SI: displacement-only Ω (no velocity) | `modules/si.py` | ✅ | Confirmed in prior review |
| A10 | §1.12 | W_l, W_p on Stiefel via `batched_cayley_retraction` | `training/train_step.py:21` | ✅ | `stiefel_update_v58` |
| A11 | T3 | U1, U2 NEVER trained — excluded from all optimizers | `training/optimizers.py` | ✅ | In `muon_exclude_ids` + `adamw_exclude_ids` |
| A12 | T3 | U1, U2 Haar-random fixed buffers | `modules/diffusion.py` | ✅ | Init random, `requires_grad=False` |

---

## Part 1: CFBank Parameters and Init (§1.1, §1.2, §1.11, §1.30, §1.33, §1.35, §1.57–§1.72)

| # | Spec | Parameter | Location | Status | Notes |
|---|---|---|---|---|---|
| B1 | §1.30 | `log_lam_bind` scalar, init -3.0, opt_g | `modules/bank.py` | ✅ | Present |
| B2 | §1.41 | `log_sigma_bind` scalar, init log(2.0), opt_g | `modules/bank.py` | ✅ | Present |
| B3 | §1.33 | `g_c` (d_c,) cfloat register_buffer zeros | `modules/bank.py` | ✅ | Present |
| B4 | §1.33 | `W_goal_detect` (1,d_c) real zeros, opt_g | `modules/bank.py` | ✅ | Present |
| B5 | §1.33 | `log_lam_goal` scalar init -3.0, opt_g | `modules/bank.py` | ✅ | Present |
| B6 | §1.35 | `role_vecs` (R=8,d_c) cfloat, QR ortho init, opt_g (not Muon/Stiefel) | `modules/bank.py` | ✅ | QR when n_roles<=d_c; excluded from Muon via `muon_exclude_ids` |
| B7 | §1.35 | `log_lam_role` scalar init -3.0, opt_g | `modules/bank.py` | ✅ | Present |
| B8 | §1.55 | `log_lam_composition` scalar init -3.0, opt_g | `modules/bank.py` | ✅ | Present |
| B9 | §1.57 | `fisher_unit` (N_max_l,) float32 zeros register_buffer | `modules/bank.py` | ✅ | Present |
| B10 | §1.59 | `buf_L1_w_full` (K_L1,N_max_l) float32 register_buffer | `modules/bank.py` | ✅ | Present |
| B11 | §1.59 | `buf_L2_w_full` (K_L2,N_max_l) float32 register_buffer | `modules/bank.py` | ✅ | Present |
| B12 | §1.59 | `buf_L3_w_full` (K_L3,N_max_l) float32 register_buffer | `modules/bank.py` | ✅ | Present |
| B13 | §1.68/69 | `_Emin_mean`, `_Emin_var`, `_Emin_n` Welford buffers | `modules/bank.py` | ✅ | Present as plain floats |
| B14 | §1.72 | `log_cal_scale` init log(0.15), opt_g | `modules/bank.py` | ✅ | Present |
| B15 | §0 Removed | `mask_embed` ABSENT | `modules/bank.py` | ✅ | Not present |
| B16 | §0 Removed | `W_compress_L1/L2/L3` ABSENT | `modules/bank.py` | ✅ | Not present |
| B17 | T1 | `alpha_freeze` is scalar float, NOT Tensor | `modules/bank.py:90` | ✅ | `self.alpha_freeze=0.7` scalar |
| B18 | T1 | `is_sensory_l` is bool Tensor (N_max_l,) | `modules/bank.py` | ✅ | `register_buffer('is_sensory_l', ...)` |
| B19 | §1.43 | `_proto_count` (N_max_l,) int32 register_buffer | `modules/bank.py` | 🔍 | Verify present; was listed in audit |
| B20 | §1.43 | `_proto_sum` (N_max_l,d_c) cfloat register_buffer | `modules/bank.py` | 🔍 | Verify present |

---

## Part 2: CFL5Layer Binding (§1.30, §1.33, §1.35, §1.55, §1.64)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| C1 | §1.30 | `B_bind[i,j] = exp(-|φ_i-φ_j|² / σ²)` | `modules/cfl5.py:150` | ✅ | `exp(-phi_diff²/sigma_sq_bind.clamp(1e-6))` |
| C2 | §1.30 | `W_full += exp(log_lam_bind) * B_bind` | `modules/cfl5.py:~152` | ✅ | Applied |
| C3 | §1.35 | `B_role = α @ α.T` (Gram, PSD) | `modules/cfl5.py:157` | ✅ | `alpha_role @ alpha_role.T` |
| C4 | §1.35 | `W_full += exp(log_lam_role) * B_role` | `modules/cfl5.py:~159` | ✅ | Applied |
| C5 | §1.55 | `B_comp = B_bind ⊙ B_role` (Hadamard) | `modules/cfl5.py:161` | ✅ | `B_bind * B_role` |
| C6 | §1.55 | `W_full += exp(log_lam_composition) * B_comp` | `modules/cfl5.py:~163` | ✅ | Applied |
| C7 | §1.33 | `x_c_eff = x_c + exp(log_lam_goal) * g_c` computed | `modules/cfl5.py:47` | ✅ | **Fixed 2026-05-11** — goal block moved before routing; `x_c_eff` used in both `compute_energies` calls |
| C8 | §1.64 | `k_l_eff = k_l_min + round((k_l_max-k_l_min)*U_epi_cal)` | `modules/cfl5.py` | ✅ | Present (variable named `k_l` not `k_l_eff`) |

---

## Part 3: ComplexUnitaryDenoisingNet / LISTA (§1.13, §1.18, §1.19, §1.31, §1.34, §1.40, §1.45, §1.58, §1.73)

| # | Spec | Parameter / Behaviour | Location | Status | Notes |
|---|---|---|---|---|---|
| D1 | §1.40 | `tau_smooth` scalar init 0.1, `nn.Parameter` | `modules/diffusion.py` | ✅ | Present |
| D2 | §1.40 | STELA: `z*sigmoid((z.abs()-τ)/τ_smooth.clamp(1e-3))` | `modules/diffusion.py` | ✅ | Confirmed |
| D3 | §1.58 | `sigma_sq_buffer=[1.0]*5` (NOT 0.0) | `modules/diffusion.py` | ✅ | `[1.0,1.0,1.0,1.0,1.0]` |
| D4 | §1.58 | `log_precision=[0.0]*5` plain list | `modules/diffusion.py` | ✅ | Confirmed |
| D5 | §1.58 | `_precision_active=[False]*5` | `modules/diffusion.py` | ✅ | Confirmed |
| D6 | §1.58 | `log_w_meta` ABSENT (replaced by precision) | `modules/diffusion.py` | ✅ | Not in __init__ |
| D7 | §1.58 | Precision-weighted U_meta: `(prec*signals).sum()/(prec.sum()+1e-8)` | `modules/diffusion.py` | ✅ | Confirmed |
| D8 | §1.73 | `log_blend_alpha` init log(0.8), `nn.Parameter` | `modules/diffusion.py` | ✅ | Present |
| D9 | §1.31 | `rule_K` shape `(N_rules, 2*d_c)` for dual-key | `modules/diffusion.py` | ✅ | Confirmed |
| D10 | §0 Removed | `_log_w_rec` ABSENT | `modules/diffusion.py` | ✅ | Not present |
| D11 | T2 | `r_lista` always `.detach()` | `modules/diffusion.py` | ✅ | BPTT disabled |
| D12 | §1.19 / §1.50 | `W_rc_bridge` is `nn.Parameter` (not register_buffer) | `modules/diffusion.py` or `model.py` | ✅ | `modules/model.py` init |
| D13 | §1.45 | Reservoir-augmented LISTA reconstruction (SE-3) | `modules/diffusion.py:316-324` | ✅ | `rho_sel_se3 = bank.rho_l[sel].mean(0)` stored; used by IterativeRefinement |
| D14 | §1.38 | `reset_lista_reservoir`: resets `sigma_sq_buffer` to `[1.0]*5` | `modules/diffusion.py` | ✅ | Confirmed |
| D15 | §1.38 | `reset_lista_reservoir`: does NOT reset `log_precision` | `modules/diffusion.py` | ✅ | Confirmed (by design — session memory) |
| D16 | §1.34 | Five-signal U_meta_v4 (U_repr_q, U_epi_cal, U_hop, U_temp, U_hypo) | `modules/diffusion.py:400-418` | ✅ | All 5 in `raw_signals`; precision-weighted fusion confirmed |

---

## Part 4: CFLNModel (§1.37, §1.38, §1.39, §1.44, §1.50, §1.65)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| E1 | §1.50 | `W_rc_bridge = nn.Parameter(...)` init random/√d_r_node | `modules/model.py` | ✅ | Confirmed |
| E2 | §1.37 | CONSOL-1 called BEFORE reset in `reset_for_inference` | `modules/model.py:215` | ✅ | CONSOL-1 first, archive second |
| E3 | §1.38 | `surprise_archive.save_state(path)` called when `persist_archive=True` | `modules/model.py:219-223` | ✅ | **Fixed** — `save_state`/`load_state` now exist in `surprise.py` |
| E4 | §1.39 | PUSH_GOAL: `_goal_stack.append(r_lista.clone())`, max D=4 | `modules/model.py:373-389` | ✅ | Confirmed |
| E5 | §1.65 | POP_GOAL merge: `sigmoid(_last_Q_BEAM_score)` (not fixed 0.7) | `modules/model.py:385` | ✅ | `torch.sigmoid(torch.tensor(cun._last_Q_BEAM_score))` |
| E6 | §1.44 | MDLM: masked ids use standard embed table (no `mask_embed`) | `modules/model.py` | ✅ | Uses `embed(masked_ids)` |
| E7 | §1.28 | Titans M update suppressed during `_in_thinking_mode` | `modules/model.py` + `modules/titans.py` | ✅ | Flag respected |
| E8 | §1.38 | `sigma_sq_buffer` reset to `[1.0]*5` inside `reset_lista_reservoir` | `modules/model.py` (via diffusion) | ✅ | Confirmed |

---

## Part 5: TelescopingMemory / VQ-Telescope (§1.6, §1.42, §1.51, §1.54, §1.59)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| F1 | §1.59 | `vq_telescope_update`: stores `s_l_full.detach().float()` | `modules/telescoping.py` | ✅ | Full routing vector stored |
| F2 | T6 | `L_vq`: codebook detached, encoder NOT: `bank.mu_c_l[sel_l].detach().mean(0)` | `modules/telescoping.py` | ✅ | `nearest_centroids = bank.mu_c_l[sel_l].detach().mean(0)` |
| F3 | §1.59 | `W_compress_L1/L2/L3` and `W_decompress_L1` ABSENT | `modules/telescoping.py` | ✅ | Not present |
| F4 | §1.54 | `micro_consolidate_arc` called per chunk in `_update_telescoping` | `modules/model.py:~320` | ✅ | Called in `_update_telescoping` |
| F5 | §1.68/69 | Welford E_min update in `_update_telescoping` | `modules/model.py:315-321` | ✅ | Lines 315-321 confirmed |
| F6 | §1.59 | `vq_telescope_retrieve`: dot product in routing-weight space | `modules/telescoping.py` | ✅ | Confirmed |

---

## Part 6: SurpriseArchive (§1.7, §1.38, §1.68)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| G1 | §1.38 | `save_state(path)` method exists | `modules/surprise.py` | ✅ | **Added 2026-05-11** |
| G2 | §1.38 | `load_state(path)` method exists | `modules/surprise.py` | ✅ | **Added 2026-05-11** |
| G3 | §1.68 | `add_vq(buf_ptr, e_min_raw)` method exists | `modules/surprise.py:56` | ✅ | Present |
| G4 | §1.68 | `retrieve_vq(s_l_full_query, bank)` method exists | `modules/surprise.py:74` | ✅ | Present |
| G5 | §1.7 | `maybe_add(c_k, s_t)` still present for non-VQ path | `modules/surprise.py:29` | ✅ | Coexists |
| G6 | §0 Removed | `tau_sa_dedup` cosine dedup replaced by Welford E_min (VQ path) | `modules/surprise.py` | ✅ | VQ path uses `e_min_raw` score |

---

## Part 7: Training Step (§1.57, §1.58, §1.60, T7)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| H1 | T7 | Fisher accumulated AFTER `backward()`, BEFORE `clip_grad_norm_` | `training/train_step.py:376-407` | ✅ | Correct ordering |
| H2 | §1.57 | Fisher loop excludes Stiefel ids `{W_l, W_p}` | `training/train_step.py:378` | ✅ | `_stiefel_ids` set |
| H3 | §1.57 | Fisher loop excludes sensory units via `update_fisher_magnitude_freeze` | `training/train_step.py` | ✅ | **Fixed 2026-05-11** — broken `_sensory_param_ids` gate removed; per-unit freeze handled correctly by `update_fisher_magnitude_freeze` |
| H4 | §1.57 | Per-unit W_l scalar Fisher accumulated separately | `training/train_step.py:390-396` | ✅ | `_wl_g2 = W_l.grad[:n_l].abs().mean(dim=(-2,-1))` |
| H5 | §1.42/§1.59 | `L_compress` REMOVED from `L_pass1` | `training/train_step.py` | ✅ | **Fixed 2026-05-11** — only `L_vq` active |
| H6 | §1.59 | `L_vq` added to `L_pass1` | `training/train_step.py:325-328` | ✅ | Present with `lambda_vq` weight |
| H7 | §1.50 | `L_bridge` summed over CFL layers, added to `L_pass1` | `training/train_step.py:317-323` | ✅ | Present |
| H8 | §1.56 | ROB-L `L_lipschitz` on young units | `training/train_step.py:335-338` | ✅ | Present |
| H9 | §1.56 | ROB-S `L_sigma_reg` on `log_sigma_bind` | `training/train_step.py:340-342` | ✅ | Present |
| H10 | §1.44 | MDLM `L_mlm` on stage 0 only | `training/train_step.py:363-372` | ✅ | Present |
| H11 | §1.58 | `L_precision` monitoring term (non-differentiable sum) | `training/train_step.py:344-348` | ✅ | Present |
| H12 | §1.57 | Fisher-KL penalty `L_KL` with warmup and EMA | `training/train_step.py:350-361` | ✅ | Present |
| H13 | §1.60 | `update_fisher_magnitude_freeze` called once per 100 steps | `training/train_step.py:399-400` | ✅ | **Fixed 2026-05-11** — duplicate call after Stiefel update removed |
| H14 | §1.60 | W_ll cache cleared after `opt_p.step()` | `training/train_step.py:423` | ✅ | `_layer._W_ll_cache.clear()` |

---

## Part 8: Optimizers (§3.1)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| I1 | T3 | U1, U2 excluded from Muon | `training/optimizers.py` | ✅ | In `muon_exclude_ids` |
| I2 | T3 | U1, U2 excluded from AdamW | `training/optimizers.py` | ✅ | In `adamw_exclude_ids` |
| I3 | §1.35 | `role_vecs` → AdamW `opt_g` (not Muon, not Stiefel) | `training/optimizers.py` | ✅ | Confirmed |
| I4 | §1.50 | `W_rc_bridge` → AdamW `opt_g` | `training/optimizers.py` | ✅ | Confirmed |
| I5 | §1.59 | `W_compress_L1/L2/L3`, `W_decompress` ABSENT from optimizers | `training/optimizers.py` | ✅ | Comment at line 59 confirms |

---

## Part 9: v9_ops (§1.37, §1.46, §1.48, §1.63, §1.67)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| J1 | §1.67 | `consolidate_arc_to_cnep`: Fisher-scaled `alpha_eff = alpha/(1+fisher_unit)` | `modules/v9_ops.py` | ✅ | `alpha_eff = alpha_consol / (1.0 + _fisher)` |
| J2 | §1.63 | `update_fisher_magnitude_freeze`: threshold = mean + 1.5×std → writes `is_sensory_l` | `modules/v9_ops.py` | ✅ | Confirmed |
| J3 | §1.63 | `micro_consolidate_arc`: per-chunk ARC→CNEP | `modules/v9_ops.py` | ✅ | Present |
| J4 | §1.46 | `compute_Q_beam`: F1–F5 + D1 all computed | `modules/v9_ops.py` | ✅ | Confirmed |
| J5 | §1.46 F2 | `compute_Q_beam` F2: goal-proxy Lyapunov `-(r_lista-r_goal_proxy).norm()²` | `modules/v9_ops.py:112-113` | ✅ | `r_goal_proxy` passed from diffusion; `None` guard present |
| J6 | §1.48 | Goal proxy `r_lista_goal` passed to `compute_Q_beam` via `r_goal_proxy` | `modules/diffusion.py:341-346` | ✅ | `r_lista_goal` is top of `_goal_stack` (or `None` when empty) |
| J7 | §1.49 | CSP arc-consistency (F5): min cosine over SSP stack | `modules/v9_ops.py:116-129` | ✅ | `goal_stack` iterated; floor at 0.1 |

---

## Part 10: Config Hygiene (§1.77, Part 0 Removed Keys)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| K1 | §1.77 | `surprise_threshold` ABSENT from config | `config.py` | ✅ | Not present |
| K2 | §1.77 | `spawn_threshold` ABSENT from config | `config.py` | ✅ | Not present |
| K3 | §1.77 | `ssp_stuck_threshold` ABSENT from config | `config.py` | ✅ | Not present |
| K4 | §1.77 | `ssp_merge_alpha` ABSENT from config | `config.py` | ✅ | Not present |
| K5 | §1.77 | `tau_proto` ABSENT (replaced by `tau_proto_min`) | `config.py` | ✅ | Only `tau_proto_min` present |
| K6 | §1.77 | `beam_B` ABSENT (replaced by adaptive `B_eff`) | `config.py` | ✅ | Not present |
| K7 | §1.77 | `lambda_recon` ABSENT (replaced by `lambda_vq`) | `config.py` | ✅ | Not present |
| K8 | §1.76 | `N_rules = 256` (raised from 64 in v9.0) | `config.py:74` | ✅ | `N_rules: int = 256` |

---

## Part 11: DCG+ Inference (§1.74, §1.75)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| L1 | §1.74 | Adaptive commit threshold: `max(0.1, 1.0 - U_epi_cal)` | `inference/dcg.py:45-49` | ✅ | **Fixed 2026-05-11** — sentinel `commit_threshold=None`; adaptive computed from `model.bank._u_epistemic_last` |
| L2 | §1.74 | Adaptive threshold sampled once per block (not per revision round) | `inference/dcg.py:44-49` | ✅ | Computed after Phase 1 `model()`, stored in `_commit_thr`; all 3 phases use same value |
| L3 | §1.74 | Backward compatibility: caller can still pass explicit float | `inference/dcg.py:19,48-49` | ✅ | `commit_threshold=None` sentinel; `else: _commit_thr = float(commit_threshold)` |
| L4 | §1.75 | `w_commit` 3-scalar learned calibration weights (`nn.Parameter`) | `modules/model.py` | ✅ | `model.w_commit` used in `dcg.py:62` |

---

## Part 12: reset_for_inference Completeness (§1.37, §1.38, §1.39, §1.52, §1.58)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| M1 | §1.37 | CONSOL-1 (`consolidate_arc_to_cnep`) called BEFORE session clear | `modules/model.py:215` | ✅ | First line in `reset_for_inference` |
| M2 | §1.38 | `sigma_sq_buffer` reset to `[1.0]*5` on new session | `modules/diffusion.py:156` | ✅ | Inside `reset_lista_reservoir` |
| M3 | §1.58 | `_precision_active` reset to `[False]*5` on new session | `modules/diffusion.py:157` | ✅ | Inside `reset_lista_reservoir` |
| M4 | §1.39 | SSP goal stack cleared: `_goal_stack=[]` | `modules/diffusion.py:159` | ✅ | Inside `reset_lista_reservoir` |
| M5 | §1.52 | `_stuck_count`, `_v_prev` cleared | `modules/diffusion.py:160-161` | ✅ | Inside `reset_lista_reservoir` |
| M6 | §1.32 | HYPO state cleared: `_in_hypo_mode=False`, `_r_lista_hypo=None` | `modules/bank.py:284-285` | ✅ | Inside `bank.reset_reservoir` |

---

## Part 13: CS-GAT / Spectral (§1.3, §1.31)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| N1 | §1.3 | CS-GAT `n_heads_gat=4` (config default) | `modules/bank.py:28`, `config.py` | ✅ | `n_heads_gat=4` in `CFBank.__init__` default |
| N2 | §1.31 R2 | Dual-key ARC: `rule_K` shape `(N_rules, 2*d_c)` — concept key + relational key | `modules/diffusion.py:96` | ✅ | `rule_K` shape confirmed `(N_rules, 2*d_c)` |
| N3 | §1.31 R2 | Relational eigenvector `_phi_rel_cache` written by CFL5Layer `eigh` on `H_seq_sub` | `modules/cfl5.py:121-124` | ✅ | `bank._phi_rel_cache=_eigvecs[:,-1]` |
| N4 | §1.31 R2 | Dual-key mixing: concept sim + relational sim used in ARC retrieval | `modules/diffusion.py:249-264` | ✅ | `K_concept`, `K_rel` split; relational sim via `phi_rel` |
| N5 | §1.32 | ARC write SUPPRESSED during HYPO (`_in_hypo_mode=True`) | `modules/diffusion.py:438-439` | ✅ | `not _in_hypo` guard before write |

---

## Part 14: Training ROB + Online Params (§1.56, §1.74, §1.69)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| O1 | §1.56 | ROB-L `L_lipschitz` uses `young_mask = activation_freq < alpha_young` | `training/train_step.py:342` | ✅ | `_ym=bank.activation_freq_l[:_n9]<cfg.get('alpha_young',0.1)` |
| O2 | §1.69 | Welford spawn: `E_min_raw > _Emin_mean + 2.5σ` (n≥10) else routing-saturation fallback | `training/train_step.py:98-107` | ✅ | **Fixed 2026-05-11** — `_last_E_min_raw` cached in `_update_telescoping`; Welford gate in `memory_update_v605` |
| O3 | §1.69 | `bank._last_E_min_raw` cached in `_update_telescoping` for spawn gate | `modules/model.py:317` | ✅ | **Fixed 2026-05-11** — `bank._last_E_min_raw = E_min_raw` before Welford update |
| O4 | §1.72 | `log_cal_scale` used in `compute_u_epistemic` sigmoid calibration | `modules/bank.py:194-195` | ✅ | `_cal_scale = float(torch.exp(self.log_cal_scale))` applied to `u_raw` |
| O5 | §1.65 | `_last_Q_BEAM_score` written in beam selection (not just at POP_GOAL) | `modules/diffusion.py:357` | ✅ | Written every beam-selection pass |

---

## Part 15: Optimizer Assignments for v9.0 Params (§3.1)

| # | Spec | Requirement | Location | Status | Notes |
|---|---|---|---|---|---|
| P1 | §3.1 | `log_cal_scale` → AdamW `opt_g` (scalar gate param) | `training/optimizers.py` | ✅ | No exclusion; falls into `opt_g` via non-Stiefel, non-Muon sweep |
| P2 | §3.1 | `log_blend_alpha` (§1.73 diffusion) → AdamW `opt_g` | `training/optimizers.py` | ✅ | In `diff_aux.cun` params; non-Stiefel → `opt_g` |
| P3 | §3.1 | `log_w_beam` → AdamW `opt_g` | `training/optimizers.py` | ✅ | In `diff_aux.cun` params; non-Stiefel → `opt_g` |
| P4 | §3.1 | `w_commit` (dcg calibration) → AdamW `opt_g` | `training/optimizers.py` | ✅ | In `model` top-level params; non-Stiefel → `opt_g` |

---

## Open Issues (items requiring follow-up)

*All open issues resolved as of 2026-05-11.*
| ~~🔍 VERIFY~~ | ~~OI-3~~ | ~~`alpha_histogram.get_alpha_freeze()` return type~~ | `modules/alpha_hist.py:33` | ✅ CLOSED | `return float(math.exp(...))` — always Python float |
| ~~🔍 VERIFY~~ | ~~OI-4~~ | ~~SE-3 reservoir-augmented LISTA reconstruction~~ | `modules/diffusion.py:316-324` | ✅ CLOSED | `_last_rho_sel` stored correctly; no additional params needed |
| ~~🔍 VERIFY~~ | ~~OI-5~~ | ~~5-signal U_meta fusion~~ | `modules/diffusion.py:400-418` | ✅ CLOSED | All 5 signals fused via precision weights |
| ~~🔍 VERIFY~~ | ~~OI-6~~ | ~~`_proto_count`/`_proto_sum` buffers~~ | `modules/bank.py:154-155` | ✅ CLOSED | Both register_buffers present |

---

## Fixed Items (resolved in this session)

| Date | ID | Fix |
|---|---|---|
| 2026-05-11 | FIX-1 | Added `save_state(path)` and `load_state(path)` to `SurpriseArchive` (`surprise.py`) |
| 2026-05-11 | FIX-2 | `_L_compress_accum` (multi-chunk accumulation) now used in `train_step.py` L_vq term and cleared after use; previously discarded before use |
| 2026-05-11 | FIX-3/OI-9 | Removed broken `_sensory_param_ids` gate from Fisher loop — `enumerate(mu_c_l)` yielded row-tensor views whose `id()` never matched params; sensory freeze delegated to `update_fisher_magnitude_freeze` |
| 2026-05-11 | OI-1 | Goal block moved before routing in `cfl5.py`; `x_c_eff` now used in both `compute_energies` calls — goal-anchored routing active |
| 2026-05-11 | OI-2 | Removed duplicate `update_fisher_magnitude_freeze` call after Stiefel update in `train_step.py` |
| 2026-05-11 | OI-7 | `bank._last_s_l_full` cache added to `bank.py`; set in `vq_telescope_update` (`telescoping.py`); used as retrieval query in `_retrieve_all_memory` (`model.py`) — was always zeros |
| 2026-05-11 | OI-8 | Both `_update_telescoping` call sites in `model.py` now pass `s_l_full`, `sel_l`, `E_min_raw` built from CFL layer info — VQ writes now activated |
| 2026-05-11 | FIX-4 | §1.74 DCG+ adaptive commit threshold: `commit_threshold=None` sentinel added; `_commit_thr = max(0.1, 1.0 - U_epi_cal)` computed from `model.bank._u_epistemic_last` after Phase 1 |
| 2026-05-11 | FIX-5 | §1.69 Welford spawn threshold: `bank._last_E_min_raw` cached in `_update_telescoping`; `memory_update_v605` uses `E_min_raw > _Emin_mean + 2.5σ` when n≥10, falls back to `s_l.max() < eps_s` |
