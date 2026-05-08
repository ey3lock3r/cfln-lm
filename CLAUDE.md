# CLAUDE.md

Behavioral guidelines for this project. Merge with Karpathy-style LLM coding discipline.

---

## Project: CFLN-LM

**CFLN** (Complex-domain Full-Layer Network) is a research language model architecture implementing:
- Complex-valued tensors throughout (`torch.cfloat`) — single `to_complex` at input, single `to_real` at output
- CNEP (Complex Neural Energy Prototype) two-tier memory bank (local RQ-entmax + persistent softmax)
- Titans associative memory with Wirtinger gradient updates
- LISTA sparse working memory with session reservoir
- CRoPE (Complex RoPE) with NTK scaling for 1M-token context
- CS-GAT (Complex Spectral Graph Attention Transformer) with magnetic Laplacian
- Continual learning stack: SI (synaptic intelligence) + alpha_freeze + domain detection
- Fourier reservoir nodes + LISTA session reservoir

Spec source: `docs/CFLN_v609_Master_Spec.md` — this is the single authoritative document.

---

## Package Management & Execution

**Always use `uv`** for all package management and running:

```bash
uv add <package>           # install dependency
uv run python main.py      # run scripts
uv run pytest              # run tests
uv run python -c "..."     # one-liners
```

Never use `pip`, `python` directly, or `conda`.

---

## Coding Discipline

### 1. Think Before Coding

- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style even if you'd do it differently.
- Remove only imports/variables/functions YOUR changes made unused.

### 4. Goal-Driven Execution

Transform tasks into verifiable goals before coding:
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Add component X" → match spec exactly, verify numerics match spec formulas

---

## Architecture Invariants (from spec)

These must never be violated:

| Rule | Detail |
|---|---|
| **Single complex domain** | `to_complex` only in `ComplexEmbedding`, `to_real` only at logit output |
| **RMS norm only** | Never `F.layer_norm` on complex — use `complex_layer_norm()` (RMS, phase-preserving) |
| **CRoPE placement** | Only on CFL-5 residual + Titans query. Never before CNEP energy computation |
| **Two-tier CNEP** | Local (RQ + entmax-1.5) + Persistent (exp + softmax). Global tier removed in v6.0.8 |
| **Wirtinger gradients** | Titans `M` updates use Wirtinger calculus — no autograd shortcuts |
| **SI omega** | Displacement-only Ω (no velocity term). Strong SI protection on persistent tier |
| **Stiefel manifold** | `W_l`, `W_p` live on Stiefel — update via `batched_cayley_retraction` |
| **dtype** | All internal tensors `torch.cfloat`. Only logits and losses in `float32` |

---

## Key Hyperparameter Defaults (v6.0.9)

```python
d_c      = 256        # complex feature dim
n_l      = 2048 + 64  # local units (64 added for global tier removal)
n_p      = 128        # persistent units
L        = 6          # CFL-5 stack depth
C_chunk  = 512        # tokens per chunk
K_L1     = 128        # telescoping L1 buffer
K_L2     = 32         # telescoping L2 buffer
K_L3     = 32         # telescoping L3 buffer
N_archive = 256       # surprise archive
rope_base ≈ 5.25e6   # NTK-scaled for 1M-token context
```

---

## Testing

```bash
uv run pytest -x          # stop on first failure
uv run pytest -x -v       # verbose
```

Numerical tests should verify against closed-form spec formulas (e.g., RQ kernel value, entmax normalization, CRoPE magnitude preservation `|exp(iθ)|=1`).

---

## File Layout

```
main.py                   # entry point (stub)
pyproject.toml            # uv project config
docs/CFLN_v609_Master_Spec.md  # authoritative spec — read before implementing any component
```
