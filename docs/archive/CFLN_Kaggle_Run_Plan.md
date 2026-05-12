# CFLN on Kaggle — Complete Run Plan
# v6.0.9 · 5 training stages · 7 ablations · checkpoint-resumable

---

## 1. KAGGLE ENVIRONMENT FACTS

| Resource | Free | Pro |
|---|---|---|
| GPU | T4 (16GB) or P100 (16GB) | T4×2 or P100 |
| Session limit | 9h (no internet) / 12h (internet on) | same |
| Disk (working) | 20GB `/kaggle/working` | 20GB |
| Disk (dataset) | 100GB (private dataset) | 100GB |
| RAM | 13GB | 16GB |
| Internet | toggleable | toggleable |

**Key constraint:** session dies after limit → every stage must checkpoint.
**Key strategy:** save to Kaggle Dataset between sessions, load from Dataset on resume.

---

## 2. MODEL SIZE AT CFG_ABLATION_605 (current default)

```python
CFG_ABLATION_605 = {
    'd_c': 64, 'n_l': 320, 'n_p': 32, 'L': 3,
    'vocab_size': 8192, 'd_e_l': 16, 'd_e_p': 32
}
```

| Metric | Value |
|---|---|
| Parameters | ~8-12M (ABLATION scale) |
| Model memory | ~50-100MB |
| Adam state | ×3 → ~300MB total |
| Activation peak (B=8, T=256) | ~1-2GB |
| **Fits T4 16GB?** | **✓ comfortably** |

**For larger experiments**, scale to production (d_c=128, n_l=2048, L=6):
- Parameters ~28M, training memory ~2GB, fits T4 with B=4

---

## 3. NOTEBOOK STRUCTURE — One notebook per stage

```
cfln_stage0_lm_warmup.ipynb          # Stage 0: LM baseline
cfln_stage1_psc.ipynb                # Stage 1: PSC pretraining
cfln_stage2_rpp_star.ipynb           # Stage 2: RPP-STaR trace gen
cfln_stage3_sft.ipynb                # Stage 3: SFT on traces
cfln_stage4_grpo.ipynb               # Stage 4: GRPO finetuning
cfln_ablations.ipynb                 # Ablations A85-A93
```

Each notebook:
1. **CELL 1** — install + import
2. **CELL 2** — load checkpoint from Kaggle Dataset (if resuming)
3. **CELL 3** — training loop with auto-save every N steps
4. **CELL 4** — save checkpoint to `/kaggle/working` then upload to Dataset

---

## 4. CHECKPOINT STRATEGY

### Save format (use torch.save with full state)
```python
def save_checkpoint(model, opts, si, step, stage, path):
    """Save everything needed to resume exactly."""
    ckpt = {
        'step': step,
        'stage': stage,
        'model_state': model.state_dict(),
        # Optimizer states
        'opt_g_state':    opts['opt_g'].state_dict(),
        'opt_u_state':    opts['opt_u'].state_dict(),
        'muon_state':     opts['muon'].state_dict(),
        # SI omega buffers (critical for CL)
        'si_omega':       {n: p.clone() for n, p in si.omega.items()},
        'si_theta_0':     {n: p.clone() for n, p in si.theta_0.items()},
        # Non-parameter session state
        'bank_x_c_prev':      model.bank._x_c_prev_bank.clone(),
        'bank_ema_delta':     model.bank._ema_delta_bank.clone(),
        'titans_M':           model.encoder.titans.M.clone(),
        # Metadata
        'cfg': model.cfg,
        'loss_history': loss_history,  # list of floats
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: step={step}, stage={stage}")

def load_checkpoint(path, model, opts, si):
    """Resume from checkpoint."""
    ckpt = torch.load(path, map_location='cuda')
    model.load_state_dict(ckpt['model_state'])
    opts['opt_g'].load_state_dict(ckpt['opt_g_state'])
    opts['opt_u'].load_state_dict(ckpt['opt_u_state'])
    opts['muon'].load_state_dict(ckpt['muon_state'])
    # Restore SI
    for n, p in ckpt['si_omega'].items():
        si.omega[n] = p
    for n, p in ckpt['si_theta_0'].items():
        si.theta_0[n] = p
    # Restore bank state
    model.bank._x_c_prev_bank.copy_(ckpt['bank_x_c_prev'])
    model.bank._ema_delta_bank.copy_(ckpt['bank_ema_delta'])
    model.encoder.titans.M.copy_(ckpt['titans_M'])
    return ckpt['step'], ckpt['stage'], ckpt['loss_history']
```

### Upload to Kaggle Dataset (persist between sessions)
```python
import subprocess

def push_to_dataset(local_path, dataset_name="cfln-checkpoints"):
    """Push checkpoint to Kaggle Dataset for persistence."""
    subprocess.run([
        "kaggle", "datasets", "version",
        "-p", str(local_path.parent),
        "-m", f"checkpoint step {step}",
        "--dir-mode", "zip"
    ], check=True)
    print(f"Uploaded to dataset: {dataset_name}")
```

---

## 5. DATA SETUP

### Recommended dataset: WikiText-103 or TinyStories
```python
# In Kaggle: add dataset via "Add Data" → search "wikitext-103"
# Path will be: /kaggle/input/wikitext-103/wiki.train.tokens

# Tokenize once, save to working dir
def prepare_data(raw_path, vocab_size=8192, chunk_size=256):
    """Tokenize and chunk. Run once, save as .pt file."""
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train(files=[raw_path], vocab_size=vocab_size, min_frequency=2)
    # ... chunk into (N_chunks, T) int tensor
    # Save: torch.save(chunks, '/kaggle/working/data_chunks.pt')
```

### For ablations (smaller scale): use TinyShakespeare
```python
# Small enough to tokenize inline, no separate dataset needed
# ~1M tokens → fast iteration
```

---

## 6. TRAINING LOOP TEMPLATE (auto-checkpoint)

```python
CHECKPOINT_EVERY = 500   # steps
SESSION_BUDGET   = 8.5   # hours (leave 30min margin before session dies)
import time

def training_loop(model, opts, si, data, cfg, start_step=0, stage='stage0'):
    t_start = time.time()
    loss_history = []
    
    for step in range(start_step, cfg['n_steps']):
        
        # ── Graceful session-end: save and exit if near limit ──────────
        elapsed = (time.time() - t_start) / 3600
        if elapsed > SESSION_BUDGET:
            print(f"Session budget reached at step {step}. Saving...")
            save_checkpoint(model, opts, si, step, stage,
                           f'/kaggle/working/ckpt_{stage}_{step}.pt')
            break
        
        # ── Training step ──────────────────────────────────────────────
        batch = sample_batch(data, cfg['batch_size'], cfg['T'])
        
        if stage == 'stage0':
            loss, info = train_step_v605(batch, model, opts, si, phase=1, cfg=cfg)
        elif stage == 'stage1':
            loss, info = psc_train_step(batch, model, opts, si, cfg=cfg)
        elif stage == 'stage3':
            loss, info = sft_train_step_ctp(batch, model, opts, si, cfg=cfg)
        elif stage == 'stage4':
            loss, info = grpo_train_step(batch, model, opts, cfg=cfg)
        
        loss_history.append(float(loss))
        
        # ── Logging ────────────────────────────────────────────────────
        if step % 100 == 0:
            avg = sum(loss_history[-100:]) / min(len(loss_history), 100)
            print(f"[{stage}] step={step:6d} loss={avg:.4f} "
                  f"U_epi={info.get('U_epistemic',0):.3f} "
                  f"t={elapsed:.1f}h")
        
        # ── Periodic checkpoint ─────────────────────────────────────────
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            ckpt_path = f'/kaggle/working/ckpt_{stage}_{step:06d}.pt'
            save_checkpoint(model, opts, si, step, stage, ckpt_path)
    
    return loss_history
```

---

## 7. STAGE-BY-STAGE EXECUTION PLAN

### NOTEBOOK 1: Stage 0 — LM Warmup (`cfln_stage0_lm_warmup.ipynb`)
**Purpose:** Train base language model. Establishes CNEP routing and basic representations.
**Config:** CFG_ABLATION_605 (default)
```python
cfg = {**CFG_ABLATION_605, 'n_steps': 10_000, 'batch_size': 8, 'T': 256,
       'lr_local': 3e-4, 'lr_persist': 1e-6, 'lr_unit': 1e-3}
model = CFLNModel(cfg)
opts  = build_optimizers_v605(model, cfg)
si    = SynapticIntelligence(model, c_SI=0.5, rho=0.999)
# Load data, run training_loop with stage='stage0'
# Save final: ckpt_stage0_final.pt
```
**Expected time:** 1.5-2h on T4
**Stop criterion:** val perplexity plateaus (usually by 8K steps at this scale)

---

### NOTEBOOK 2: Stage 1 — PSC Pretraining (`cfln_stage1_psc.ipynb`)
**Purpose:** Teach the r_lista chain HOW to use thinking tokens before doing STaR.
**Input:** Load `ckpt_stage0_final.pt`
```python
cfg = {**CFG_PSC, 'n_steps': 5_000}
# psc_train_step uses L_LM + L_improve + L_economy + L_predictive
# Loss breakdown: ~1.5% overhead vs Stage 0
```
**Expected time:** 1-1.5h on T4
**Key metric:** L_improve trending down (thinking chain getting useful)
**Stop criterion:** L_improve / L_LM < 0.05 (thinking adds <5% loss)

---

### NOTEBOOK 3: Stage 2 — RPP-STaR Trace Generation (`cfln_stage2_rpp.ipynb`)
**Purpose:** Generate high-quality training traces for SFT. 70-90% acceptance rate.
**Input:** Load `ckpt_stage1_final.pt`
```python
# Generate traces — no gradient, pure inference
traces = star_generate_traces_rpp(
    model, prompts,
    n_traces_target=5_000,
    n_think=8,             # thinking tokens per step
    cfg=CFG_PSC
)
# Save traces as a separate file
torch.save(traces, '/kaggle/working/rpp_traces.pt')
```
**Expected time:** 30-45min (inference only, faster than training)
**Key metric:** acceptance_rate (target >70%; retry with more RPP steps if <50%)

---

### NOTEBOOK 4: Stage 3 — SFT on RPP Traces (`cfln_stage3_sft.ipynb`)
**Purpose:** Teach WHAT to write in thinking tokens.
**Input:** Load `ckpt_stage1_final.pt` + `rpp_traces.pt`
```python
cfg = {**CFG_PSC, 'n_steps': 2_000, 'T': 512}  # longer T for think+output
# sft_train_step_ctp uses compute_ctp_loss(tau_think=0.5)
# tau_think=0.5: thinking tokens weighted at 50% vs output tokens
```
**Expected time:** 30-45min on T4
**Key metric:** CTP CE on thinking tokens (should drop significantly vs Stage 0 baseline)

---

### NOTEBOOK 5: Stage 4 — GRPO Finetuning (`cfln_stage4_grpo.ipynb`)
**Purpose:** Reinforce reasoning quality via intrinsic perplexity-reduction reward.
**Input:** Load `ckpt_stage3_final.pt`
```python
cfg = {**CFG_GRPO, 'n_steps': 1_000}
# G=8 rollouts per input → 8× inference per step
# R_norm = clip((R - mu_R) / sigma_R, -5, 5)
# Intrinsic reward: perplexity reduction vs no-thinking baseline
```
**Expected time:** 1.5-2h on T4 (G=8 rollouts is expensive)
**Key metric:** mean reward trending positive, KL vs reference model stable

---

### NOTEBOOK 6: Ablations (`cfln_ablations.ipynb`)
**Purpose:** Validate key architectural decisions and compare variants.

Priority order (most informative first):

| Ablation | What it tests | Config change | Est. time |
|---|---|---|---|
| **A93** | 2-tier vs 3-tier CFLN | restore n_g=64, compare | 1h each |
| **A89** | PSC+SFT+GRPO vs PSC+SFT | skip Stage 4 | already have from Stage 3 |
| **A88** | PSC+SFT vs PSC only | skip Stage 3 | ~45min |
| **A87** | Full PSC vs partial PSC | vary L_economy, L_predictive | 2× 1h |
| **A85/86** | PSC ablation components | vary PSC loss terms | 2× 1h |
| **A90** | Cold STaR baseline | skip Stages 1+2, random STaR | 1.5h |

**For A93 specifically:**
```python
# Baseline: current v6.0.9 (no global tier)  ← already trained
# Variant: restore global tier
cfg_3tier = {**CFG_ABLATION_605, 'n_g': 64, 'd_e_g': 32}
# Train 5K steps from random init on same data
# Compare: perplexity, CL benchmark (domain shift test), per-token time
```

---

## 8. KAGGLE-SPECIFIC TIPS

### GPU memory: avoid OOM
```python
# Use gradient checkpointing for long sequences
model.enable_gradient_checkpointing = True  # if implemented

# Reduce T if OOM
cfg['T'] = 128  # instead of 256

# Clear cache between stages
import gc; gc.collect(); torch.cuda.empty_cache()

# Monitor memory
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
```

### Internet on vs off
```python
# Turn internet OFF after downloading data/code
# Session limit increases: 9h → 12h with internet ON, but use carefully
# Download everything in first cell, then can turn off for main training
```

### Dataset persistence between sessions
```python
# BEFORE session ends:
!cp /kaggle/working/ckpt_stage1_final.pt /kaggle/working/to_upload/

# Then "Save & Run All" → go to Output tab → "New Dataset Version"
# Or use kaggle API: !kaggle datasets version -p /kaggle/working/to_upload -m "stage1"

# NEXT session:
# Add the dataset in "Add Data" → your private dataset
# Files appear at: /kaggle/input/cfln-checkpoints/ckpt_stage1_final.pt
ckpt = torch.load('/kaggle/input/cfln-checkpoints/ckpt_stage1_final.pt')
```

### Wandb logging (optional but recommended)
```python
import wandb
wandb.init(project="cfln", name=f"stage{stage}_run1",
           config=cfg, mode="online")  # or mode="offline" then sync later
wandb.log({'loss': loss, 'U_epistemic': u_epi, 'step': step})
```

---

## 9. RECOMMENDED RUN ORDER (minimal, gets you trained model)

```
Session 1 (12h internet ON):
  Download data, tokenize, save to dataset
  Run Stage 0 full (10K steps) → save checkpoint
  Start Stage 1 → save checkpoint at session end

Session 2 (9h internet OFF):
  Load Stage 1 checkpoint → finish Stage 1
  Run Stage 2 (RPP trace gen) → save traces
  Run Stage 3 (SFT) → save checkpoint

Session 3 (9h internet OFF):
  Load Stage 3 checkpoint
  Run Stage 4 (GRPO) → save final model
  Run Ablation A93 (2-tier vs 3-tier verification)

Session 4 (optional, ablations):
  A85-A90 ablation series
```

---

## 10. QUICK SANITY CHECKS (run before full training)

```python
# 1. Model instantiates and forward passes cleanly
model = CFLNModel(CFG_VERIFY_605)
model = model.cuda()
batch = torch.randint(0, 32, (2, 64)).cuda()
with torch.no_grad():
    out, *_ = model(batch)
print(f"Forward OK: {out.shape}")  # should be (2, 64, vocab_size)

# 2. One training step doesn't NaN
loss, info = train_step_v605(batch, model, opts, si, phase=1, cfg=CFG_VERIFY_605)
assert not torch.isnan(torch.tensor(loss)), "NaN in loss!"
print(f"Train step OK: loss={loss:.4f}, U_epi={info['U_epistemic']:.3f}")

# 3. All 68 tests pass
exec(open('cfln_tests.py').read())  # if you extract test functions to a file

# 4. Memory check
torch.cuda.empty_cache()
print(f"Model memory: {sum(p.numel()*p.element_size() for p in model.parameters())/1e6:.1f}MB")
print(f"GPU used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

---

## 11. EXPECTED OUTCOMES BY STAGE

| After | Perplexity (WikiText) | CTP firing rate | U_epi calibration |
|---|---|---|---|
| Stage 0 | ~80-120 (depends on scale) | N/A (untrained) | ~0.5 (calibrated) |
| Stage 1 | similar (PSC adds <5%) | Think chain coherent | better |
| Stage 3 | ~60-90 (SFT improves) | >10% tokens trigger | good |
| Stage 4 | ~50-80 (GRPO improves) | consistent patterns | good |

**Key signal to watch:** if L_improve in Stage 1 never drops → PSC pretraining not working → increase `psc_alpha`, decrease `u_epi_psc_threshold`
