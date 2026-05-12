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

Spec source: `docs/CFLN_Master_Spec.md` — this is the single authoritative document (v9.0).

---

## Tool Usage (RTK token savings)

**Always use Bash commands instead of native tools for reading and searching.** RTK intercepts all Bash calls via the PreToolUse hook and strips output down to only what matters (60–96% token savings). Native tools (Read, Grep, Glob) bypass RTK entirely.

| Task | Use this (Bash → RTK) | Not this (native) |
|---|---|---|
| Read a file | `Bash(cat file.py)` | Read tool |
| Search for pattern | `Bash(grep -n "pattern" file.py)` | Grep tool |
| Find files | `Bash(find src/ -name "*.py")` | Glob tool |

**Exceptions** — keep using native tools for:
- `Edit` and `Write` — no Bash equivalent
- Short one-liner reads where RTK overhead exceeds savings

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

### 0. Resolving Ambiguity — Expert Panel Protocol

Before touching any ambiguous code, convene a **virtual expert panel** to reach a grounded decision:

**When to invoke**: any time a fix has multiple plausible interpretations, a spec section is silent on an edge case, a numerical formula could be read two ways, or conflicting signals exist across spec versions.

**Tier 1 — Core panel (always present, every decision):**
- **ML Researcher** — is the math correct and stable? numerical consequences?
- **Software Architect** — does this fit the module structure? is there a simpler design?
- **Spec Author** — what was the original intent? which version introduced this?
- **Devil's Advocate** — what breaks under this interpretation? what's the failure mode?

**Tier 2 — Domain specialists (invoked when the question touches their area):**

| Specialist | Invoke when touching |
|---|---|
| **Numerical Analyst** | complex float stability, gradient magnitudes, clamping/epsilon choices, init scales |
| **Continual Learning Specialist** | SI omega, Fisher-KL, alpha_freeze, forgetting vs. plasticity tradeoffs, domain detection |
| **Optimization Theorist** | Stiefel manifold, Cayley retraction, Muon/Newton-Schulz, Wirtinger calculus, lr schedules |
| **Memory Systems Architect** | CNEP energy routing, Titans M updates, LISTA warm-start, VQ-Telescope, Telescoping FIFO |
| **GNN / Spectral Expert** | CS-GAT, Chebyshev polynomials, Hermitian adjacency, PSD conditions, W_full binding terms |
| **PyTorch Internals Expert** | `torch.cfloat` edge cases, conjugate bits, `view_as_real`, gradient hooks, `register_buffer` vs `nn.Parameter` |
| **Training Dynamics Expert** | loss weight interactions, multi-pass backward ordering, grad clip vs. Fisher accumulation timing |

**Protocol**:
1. State the ambiguity clearly — quote the exact spec line(s) in question.
2. Identify which Tier 2 specialists are relevant and add them to the panel.
3. List every plausible interpretation (2–4 options).
4. Have each panelist weigh in, citing evidence from `docs/CFLN_Master_Spec.md` or code.
5. Record the **Decision** and **Rationale** in `docs/spec_compliance.md` (or inline comment) before writing any code.
6. If consensus is impossible, escalate to the user before proceeding.

**Never guess silently** — a wrong silent fix is worse than a visible question.

---

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

## Key Hyperparameter Defaults (v9.0)

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
docs/CFLN_Master_Spec.md       # authoritative spec (v9.0) — read before implementing any component
```
