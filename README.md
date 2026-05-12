# CFLN-LM

**Complex-domain Full-Layer Network** — a research language model architecture implementing complex-valued neural computation throughout, continual learning via Synaptic Intelligence, and a PSC→RPP→GRPO reasoning training pipeline.

Implementation of **CFLN v9.0** from the master specification in `docs/CFLN_Master_Spec.md`.

---

## Architecture

CFLN processes all internal representations as `torch.cfloat` tensors. The single complex domain rule: `to_complex` is called exactly once (in `ComplexEmbedding`) and `to_real` exactly once (at the logit output).

### Core components

| Component | Module | Description |
|---|---|---|
| **ComplexEmbedding** | `modules/embedding.py` | Separate real/imag `nn.Embedding` → `torch.complex` |
| **ComplexLRU** | `modules/lru.py` | HiPPO-LegS SSM, ~200-token soft context, selective gating (v6.0.6) |
| **TitansComplexMemory** | `modules/titans.py` | Gradient-based association matrix M with Wirtinger rank-1 updates; three-channel domain detection |
| **ComplexHierarchicalOCNEncoder** | `modules/encoder.py` | Fast (LRU) + slow (Titans) encoder; position-agnostic output for CNEP |
| **CFBank** | `modules/bank.py` | Two-tier CNEP (local RQ-entmax + persistent softmax); node Fourier reservoir per unit; global tier removed v6.0.8 |
| **ComplexGATLayer** | `modules/gat.py` | Chebyshev Spectral GAT (K=3 hops) on Hermitian adjacency |
| **CFL5Layer** | `modules/cfl5.py` | One CFL-5 layer: predictive psi_for + sequential Hebbian + CS-GAT + reservoir update |
| **ComplexMHCHighway** | `modules/highway.py` | 2-stream multi-head complex highway with doubly-stochastic mixing |
| **TelescopingMemory** | `modules/telescoping.py` | 3-level (L1/L2/L3) hierarchical FIFO for 4K/32K/1M token context |
| **SurpriseArchive** | `modules/surprise.py` | Min-heap of 256 high-surprise chunks; VQ path uses Welford E_min threshold |
| **ComplexSTIHead** | `modules/sti_head.py` | STI prediction head → logits via `to_real` |
| **SynapticIntelligence** | `modules/si.py` | Online SI with displacement-only Ω; graded domain transition response |
| **ComplexUnitaryDenoisingNet** | `modules/diffusion.py` | LISTA sparse working memory with session reservoir `r_lista`, ARC cache, sparse code cache |
| **DynamicLocalBank** | `modules/dynamic_bank.py` | Prune / split / merge for local units |
| **MuonOptimizer** | `modules/muon.py` | Orthogonal gradient descent via Newton-Schulz5 for Stiefel params |
| **CFLNModel** | `modules/model.py` | Full model assembly; `_pos_offset` counter; begin_document resets |

### Key hyperparameter defaults (v9.0)

```python
d_c       = 256        # complex feature dim
n_l       = 2112       # local units (2048 + 64 for global-tier removal)
n_p       = 128        # persistent units
L         = 6          # CFL-5 stack depth
C_chunk   = 512        # tokens per chunk
K_L1/L2/L3 = 128/32/32 # telescoping buffer sizes
N_archive = 256        # surprise archive capacity
rope_base ≈ 5.25e6    # NTK-scaled for 1M-token context
```

### Architecture invariants

- All internal tensors `torch.cfloat` — only logits and losses in `float32`
- RMS norm only (`complex_layer_norm`) — never `F.layer_norm` on complex (phase-corrupting)
- CRoPE applied only at: CFL-5 residual (absolute position) and Titans query — never before CNEP energy
- `W_l`, `W_p` live on the Stiefel manifold; updated via `batched_cayley_retraction`
- SI omega uses displacement-only Ω (no velocity term); persistent tier strongly protected

---

## Training pipeline

**4-stage PSC→RPP→GRPO pipeline:**

| Stage | Method | Description |
|---|---|---|
| 0 | Standard LM | Pre-training; `log_scale_l` frozen (prevents W_dec_res calibration race) |
| 1 | PSC | Self-supervised: L_PSC = L_LM + α·L_improve + β(U_epi)·L_economy + γ·L_predictive |
| 2 | RPP-STaR | Offline trace generation via gradient optimisation (top-50 restriction, 70–90% acceptance) |
| 3 | SFT | Supervised fine-tuning on RPP traces (τ_think = 0.5) |
| 4 | GRPO | Optional RL fine-tuning (G=8 rollouts, β=0.1 KL weight) |

**Three-domain continual learning protocol:**

| Phase | Domain | Dataset | Steps | SI action |
|---|---|---|---|---|
| 1 | A — Narrative | TinyStories | 5 K | SI snapshot |
| 2 | B — Code | GitHub Python | 5 K | SI snapshot; eval A |
| 3 | C — Scientific | PubMed QA | 5 K | eval A + B |
| Recovery | A | TinyStories | 2 K | eval A recovery |

Pass criterion: backward-transfer penalty BWT_A, BWT_B < +2.0 PPL; no collapse (>3× initial PPL).

**Training step** (`train_step_v605`) enforces 14 ordering invariants including:
- SI snapshot before forward pass
- W_ll cache cleared after `opt_p.step()` (stale cache → routing misalignment)
- Stiefel retraction via Cayley map after grad clip
- Memory update (prune/split/merge) every 50 steps

---

## Inference

### CFLN Think Protocol (CTP)
When `U_epistemic > think_threshold`, the model generates `<think>…</think>` scratchpad tokens before the output token. Thinking updates `r_lista` without advancing generation. `_in_thinking_mode` suppresses Titans M updates during thinking.

### DCG+ Generation
Three-phase deferred-commitment generation: Draft → Reflect → Selective Revision → Commit. `model._pos_offset` is reset to 0 before the revision pass.

---

## Setup

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

---

## Usage

```bash
# Smoke test — 3 train steps on random data with tiny config
uv run python main.py --train

# Train with custom config
uv run python main.py --train --config configs/my_config.json

# Generate with CTP
uv run python main.py --generate --prompt "Once upon a time"

# Run tests
uv run pytest -x -v
```

---

## File layout

```
cfln-lm/
  main.py                        CLI entry point
  pyproject.toml                 uv project config
  docs/
    CFLN_Master_Spec.md          Authoritative spec (v9.0) — read before modifying any component
  src/cfln/
    config.py                    CFLNConfig dataclass (all §7 hyperparameters)
    utils.py                     §0.3 utility functions (CRoPE, entmax, Stiefel/Cayley, ...)
    modules/
      embedding.py               §2.1  ComplexEmbedding
      lru.py                     §2.2  ComplexLRU
      titans.py                  §2.3  TitansComplexMemory
      encoder.py                 §2.4  ComplexHierarchicalOCNEncoder
      coact.py                   §2.5  CoactivationRegister
      alpha_hist.py              §2.6  AlphaHistogram
      bank.py                    §2.7  CFBank + node Fourier reservoir
      gat.py                     §2.8  ComplexGATLayer (CS-GAT)
      hopfield.py                §2.9  HopfieldRetrieval
      cfl5.py                    §2.10 CFL5Layer
      highway.py                 §2.11 ComplexMHCHighway
      telescoping.py             §2.12 TelescopingMemory
      surprise.py                §2.13 SurpriseArchive
      sti_head.py                §2.14 ComplexSTIHead
      uncertainty.py             §2.15 ComplexUncertaintyModule, PerLayerLamPSchedule, ...
      si.py                      §2.19 SynapticIntelligence, ExemplarDormancyBuffer, DomainTransitionHandler
      diffusion.py               §2.20 DiffusionAuxiliaryModule, ComplexUnitaryDenoisingNet (LISTA)
      dynamic_bank.py            §2.21 DynamicLocalBank
      muon.py                    §2.22 MuonOptimizer
      psc_loss.py                §2.22b PSCLoss
      model.py                   §2.23 CFLNModel, §2.24 IterativeRefinementModule
      v9_ops.py                  §1.37/§1.46/§1.63/§1.67 consolidate_arc_to_cnep, compute_Q_beam, micro_consolidate_arc
      monitoring.py              §2.25 SlowDriftDetector, DocumentStreamingContext, NeedleInHaystackEvaluator
    training/
      optimizers.py              §3.1  build_optimizers_v605 (5-tuple)
      train_step.py              §3.5  train_step_v605 (14 invariants), psc_train_step, memory_update_v605
      curriculum.py              Three-domain CL protocol + curriculum sampler
    inference/
      ctp.py                     §4    generate_cfln_ctp, compute_ctp_loss
      dcg.py                     §4    generate_cfln_dcg_plus
  tests/
    test_utils.py                Numerical invariants: CRoPE magnitude, RMS norm phase, Stiefel, entmax
```

---

## Spec reference

The full mathematical derivations, expert panel decisions, ablation notes, and implementation warnings are in `docs/CFLN_Master_Spec.md`. That document is the single source of truth — consult it before modifying any component.
