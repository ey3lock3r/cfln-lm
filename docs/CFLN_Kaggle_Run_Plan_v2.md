# CFLN Kaggle Run Plan v2
# Expert-reviewed · all 7 ablations · 5 OQ probes · 12 gaps closed
# v6.0.9 · TinyShakespeare (smoke) → WikiText-103 (real experiments)

---

## EXPERT REVIEW FINDINGS (before reading this plan)

| Area | v1 status | v2 fixes |
|---|---|---|
| Dataset | Wrong — TinyShakespeare alone can't test CL | Two-phase: TS smoke → WikiText-103 real |
| Ablations | A85, A86 missing — PSC series uninterpretable without them | All 7 added, priority reordered |
| OQs | Not addressed | 5 OQ probes added to standard training loop |
| Tokenization | "tokenize once" with no code | Full BPE + special token setup provided |
| GRPO reference | Missing ref_model copy | `copy.deepcopy` before Stage 4 |
| Gradient clip | Missing for cfloat params | Clip pattern for complex gradients |
| Stage transitions | No guidance | Explicit: keep SI + optimizer state |
| Kaggle API | Vague | Exact commands provided |

---

## 0. ENVIRONMENT SETUP (every session, Cell 1)

```python
# ── Install (internet ON only — first session) ─────────────────────────
!pip install -q tokenizers wandb

# ── Imports ────────────────────────────────────────────────────────────
import torch, copy, time, os, gc
import torch.nn.functional as F
from pathlib import Path
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'N/A'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if DEVICE=='cuda' else "")

# ── Paths ───────────────────────────────────────────────────────────────
WORK   = Path('/kaggle/working')
INPUT  = Path('/kaggle/input')
CKPT_DIR = WORK / 'checkpoints'; CKPT_DIR.mkdir(exist_ok=True)

# ── Load CFLN spec (paste your cfln.py here or upload as dataset) ───────
# exec(open('/kaggle/input/cfln-code/cfln_v609.py').read())
```

---

## 1. DATASET SETUP — TWO PHASES

### Phase 1: TinyShakespeare (smoke test only, no download)
```python
def get_tinyshakespeare():
    """Downloads ~1MB inline. Use ONLY for pipeline smoke test."""
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data, _ = urllib.request.urlretrieve(url, WORK/'shakespeare.txt')
    return open(WORK/'shakespeare.txt').read()

# TinyShakespeare gives you: 488 steps/epoch at B=8,T=256
# Expect overfit after ~3 epochs. DO NOT use for ablations or CL tests.
# Use ONLY to verify: no crashes, loss goes down, checkpoints save/load.
```

### Phase 2: WikiText-103 (real experiments — add via Kaggle UI)
```
Kaggle UI → "Add Data" → search "wikitext103" → add "WikiText-103 Language Model Dataset"
Files appear at: /kaggle/input/wikitext-103-raw-v1/
```
```python
WIKITEXT_TRAIN = INPUT / 'wikitext-103-raw-v1/wiki.train.raw'
WIKITEXT_VALID = INPUT / 'wikitext-103-raw-v1/wiki.valid.raw'
# 103M train tokens. 267K val tokens.
# At B=8, T=256: 50,293 steps/epoch. Run 10K steps = see 20% of data.
```

---

## 2. TOKENIZER SETUP (run once, save — CRITICAL)

```python
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 8192  # matches CFG_ABLATION_605

def build_tokenizer(corpus_path, vocab_size=VOCAB_SIZE, save_dir=WORK/'tokenizer'):
    """Build BPE tokenizer. Run ONCE, save, reload in later sessions."""
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)
    
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>']
    )
    tok.save_model(str(save_dir))
    print(f"Tokenizer saved to {save_dir}. Vocab: {tok.get_vocab_size()}")
    return tok

def load_tokenizer(save_dir=WORK/'tokenizer'):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(
        str(save_dir/'vocab.json'),
        str(save_dir/'merges.txt')
    )
    return tok

# ── CRITICAL: Add CTP special tokens BEFORE model init ─────────────────
def extend_tokenizer_for_ctp(tok):
    """Add THINK_START and THINK_END tokens. Must match model vocab."""
    tok.add_special_tokens(['<think>', '</think>'])
    think_start_id = tok.token_to_id('<think>')
    think_end_id   = tok.token_to_id('</think>')
    print(f"<think>={think_start_id}, </think>={think_end_id}")
    return tok, think_start_id, think_end_id

# Build on first session:
# tok = build_tokenizer(WIKITEXT_TRAIN)
# tok, THINK_START_ID, THINK_END_ID = extend_tokenizer_for_ctp(tok)
# torch.save({'think_start': THINK_START_ID, 'think_end': THINK_END_ID},
#             WORK/'token_ids.pt')

# Load in subsequent sessions:
# tok = load_tokenizer()
# ids = torch.load(WORK/'token_ids.pt')  (or from dataset)
# THINK_START_ID, THINK_END_ID = ids['think_start'], ids['think_end']
```

---

## 3. DATA LOADING

```python
def tokenize_and_chunk(text_path, tok, chunk_size=256, stride=128):
    """Tokenize corpus → overlapping chunks → (N, T) int32 tensor."""
    text = open(text_path, encoding='utf-8').read()
    encoded = tok.encode(text)
    ids = torch.tensor(encoded.ids, dtype=torch.int32)
    
    chunks = []
    for i in range(0, len(ids) - chunk_size, stride):
        chunks.append(ids[i:i+chunk_size])
    
    data = torch.stack(chunks)  # (N, T)
    print(f"Data: {len(data):,} chunks × {chunk_size} tokens = {len(data)*chunk_size/1e6:.1f}M tokens")
    return data

def sample_batch(data, batch_size, T, device=DEVICE):
    """Random batch of (B, T) long tensors."""
    idx = torch.randint(0, len(data), (batch_size,))
    return data[idx].long().to(device)

# Save tokenized data once per dataset (expensive):
# data = tokenize_and_chunk(WIKITEXT_TRAIN, tok)
# torch.save(data, WORK/'wikitext_train_chunks.pt')
# data = torch.load(WORK/'wikitext_train_chunks.pt')  # subsequent sessions
```

---

## 4. MODEL + OPTIMIZER SETUP

```python
def setup_model(cfg, think_start_id, think_end_id):
    """Init model, move to GPU, extend vocab for CTP, compile."""
    cfg = {**cfg, 'think_start_id': think_start_id, 'think_end_id': think_end_id}
    
    model = CFLNModel(cfg).to(DEVICE)
    
    # ── CTP vocabulary extension (adds THINK tokens to embed + head) ───
    model = expand_vocabulary(model, new_vocab_size=cfg['vocab_size'])
    
    # ── Optional: torch.compile for 2-3× speedup ───────────────────────
    # Disable if unit growth causes recompilation issues
    # model = torch.compile(model, dynamic=True)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params | "
          f"Memory: {sum(p.numel()*p.element_size() for p in model.parameters())/1e6:.1f}MB")
    return model

def setup_training(model, cfg):
    """Build optimizers, SI, lr scheduler. Returns opts dict + si."""
    opts = build_optimizers_v605(model, cfg)
    # opts = {'opt_g': AdamW, 'opt_u': AdamW, 'muon': Muon}
    
    si = SynapticIntelligence(model, c_SI=cfg.get('c_SI', 0.5),
                               rho=cfg.get('rho_SI', 0.999))
    si.record_theta_0(model)  # snapshot initial params for SI omega
    
    # ── LR warmup scheduler ─────────────────────────────────────────────
    warmup_steps = cfg.get('warmup_steps', 500)
    def lr_lambda(step):
        return min(1.0, step / warmup_steps)
    schedulers = {
        'sched_g': torch.optim.lr_scheduler.LambdaLR(opts['opt_g'], lr_lambda),
        'sched_u': torch.optim.lr_scheduler.LambdaLR(opts['opt_u'], lr_lambda),
    }
    return opts, si, schedulers
```

---

## 5. CHECKPOINT SAVE / LOAD (complete state)

```python
def save_checkpoint(model, opts, si, schedulers, step, stage, path):
    ckpt = {
        'step': step, 'stage': stage,
        'model_state':   model.state_dict(),
        'opt_g_state':   opts['opt_g'].state_dict(),
        'opt_u_state':   opts['opt_u'].state_dict(),
        'muon_state':    opts['muon'].state_dict(),
        'sched_g_state': schedulers['sched_g'].state_dict(),
        'sched_u_state': schedulers['sched_u'].state_dict(),
        # SI — NEVER reset between stages (CL memory)
        'si_omega':   {n: p.cpu().clone() for n, p in si.omega.items()},
        'si_theta_0': {n: p.cpu().clone() for n, p in si.theta_0.items()},
        # Non-param session state
        'titans_M':         model.encoder.titans.M.cpu().clone(),
        'bank_x_c_prev':    model.bank._x_c_prev_bank.cpu().clone(),
        'bank_ema_delta':   model.bank._ema_delta_bank.cpu().clone(),
        'bank_u_epi_mu':    model.bank._u_epi_mu.cpu().clone(),
        'bank_u_epi_var':   model.bank._u_epi_var.cpu().clone(),
        'cfg':              model.cfg,
    }
    torch.save(ckpt, path)
    print(f"✓ Saved: {path.name}  (step={step}, stage={stage})")

def load_checkpoint(path, model, opts, si, schedulers):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    opts['opt_g'].load_state_dict(ckpt['opt_g_state'])
    opts['opt_u'].load_state_dict(ckpt['opt_u_state'])
    opts['muon'].load_state_dict(ckpt['muon_state'])
    schedulers['sched_g'].load_state_dict(ckpt['sched_g_state'])
    schedulers['sched_u'].load_state_dict(ckpt['sched_u_state'])
    for n, p in ckpt['si_omega'].items():
        si.omega[n] = p.to(DEVICE)
    for n, p in ckpt['si_theta_0'].items():
        si.theta_0[n] = p.to(DEVICE)
    model.encoder.titans.M.copy_(ckpt['titans_M'].to(DEVICE))
    model.bank._x_c_prev_bank.copy_(ckpt['bank_x_c_prev'].to(DEVICE))
    model.bank._ema_delta_bank.copy_(ckpt['bank_ema_delta'].to(DEVICE))
    model.bank._u_epi_mu.copy_(ckpt['bank_u_epi_mu'].to(DEVICE))
    model.bank._u_epi_var.copy_(ckpt['bank_u_epi_var'].to(DEVICE))
    print(f"✓ Loaded: step={ckpt['step']}, stage={ckpt['stage']}")
    return ckpt['step'], ckpt['stage']
```

---

## 6. KAGGLE DATASET PERSISTENCE (exact commands)

```python
# ── First-time setup (internet ON, once per account) ───────────────────
# !pip install kaggle
# Upload kaggle.json to /root/.kaggle/ via Kaggle secrets:
import json, os
os.makedirs('/root/.kaggle', exist_ok=True)
# In Kaggle: Add-ons → Secrets → add KAGGLE_KEY with your kaggle.json content
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient().get_secret("KAGGLE_KEY")
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(secret)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# ── After training: upload checkpoint as dataset version ───────────────
def upload_checkpoint(local_ckpt_path, dataset_title='cfln-checkpoints'):
    """Upload checkpoint to Kaggle Dataset. Run before session ends."""
    upload_dir = WORK / 'upload'
    upload_dir.mkdir(exist_ok=True)
    
    # Copy checkpoint(s) to upload dir
    import shutil
    shutil.copy(local_ckpt_path, upload_dir)
    
    # Create dataset metadata (first time only)
    meta = {'title': dataset_title, 'id': f'{os.environ["KAGGLE_USERNAME"]}/{dataset_title}',
            'licenses': [{'name': 'CC0-1.0'}]}
    with open(upload_dir / 'dataset-metadata.json', 'w') as f:
        json.dump(meta, f)
    
    # Push new version
    !kaggle datasets version -p {upload_dir} -m "step {step}"
    print(f"✓ Uploaded to kaggle dataset: {dataset_title}")

# ── Next session: add dataset via Kaggle UI → Add Data → your dataset ──
# Files appear at: /kaggle/input/cfln-checkpoints/ckpt_stage0_final.pt
```

---

## 7. TRAINING LOOP WITH OQ PROBES + AUTO-CHECKPOINT

```python
SESSION_BUDGET_H = 8.5   # save and stop before Kaggle kills session
CKPT_EVERY       = 500   # steps between checkpoints
LOG_EVERY        = 100   # steps between console/wandb logs

# ── OQ Probe metrics (logged every step) ────────────────────────────────
class OQProbes:
    """Instruments for Open Questions — zero overhead when not measured."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.escape_count = 0        # OQ-v598-4: escape frequency
        self.rule_writes  = 0        # OQ-v598-4: rule cache writes
        self.rule_reads   = 0        # OQ-v598-4: rule cache reads
        self.domain_shifts= 0        # OQ-v596-3: domain detection events
        self.u_epi_vals   = []       # OQ-v598-2: U_epi calibration
        self.ce_vals      = []       # OQ-v598-2: actual CE for correlation
        self.n_active_units = []     # track unit growth
    def log(self, info, loss):
        u = info.get('U_epistemic', 0.5)
        self.u_epi_vals.append(u)
        self.ce_vals.append(float(loss))
        self.n_active_units.append(info.get('n_l', 0))
        if info.get('escape_fired', False):  self.escape_count += 1
        if info.get('rule_write',  False):   self.rule_writes  += 1
        if info.get('rule_read',   False):   self.rule_reads   += 1
        if info.get('domain_shift',False):   self.domain_shifts+= 1
    def summary(self, window=500):
        n = min(window, len(self.u_epi_vals))
        if n < 10: return {}
        u = self.u_epi_vals[-n:]; ce = self.ce_vals[-n:]
        # Pearson correlation between U_epi and CE (OQ-v598-2)
        import statistics
        try:
            corr = (sum((a-statistics.mean(u))*(b-statistics.mean(ce))
                    for a,b in zip(u,ce)) /
                    (n * statistics.stdev(u) * statistics.stdev(ce) + 1e-8))
        except: corr = 0.0
        return {
            'u_epi_ce_corr': corr,           # OQ-v598-2: target >0.4
            'escape_rate':   self.escape_count / max(n,1),
            'rule_write_rate': self.rule_writes / max(n,1),
            'rule_read_rate':  self.rule_reads  / max(n,1),
            'domain_shifts': self.domain_shifts,
            'n_active_units': self.n_active_units[-1] if self.n_active_units else 0,
        }

def training_loop(model, opts, si, schedulers, data, cfg,
                  stage='stage0', start_step=0,
                  train_fn=None,  # e.g. train_step_v605
                  n_steps=None):
    
    n_steps = n_steps or cfg.get('n_steps', 10_000)
    t0 = time.time()
    probes = OQProbes()
    loss_hist = []
    
    if train_fn is None:
        train_fn = lambda batch, step: train_step_v605(
            batch, model, opts, si, phase=2 if step > cfg.get('phase2_start', 5000) else 1,
            cfg=cfg)
    
    for step in range(start_step, start_step + n_steps):
        
        # ── Session budget guard ────────────────────────────────────────
        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= SESSION_BUDGET_H:
            ckpt_path = CKPT_DIR / f'ckpt_{stage}_{step:06d}_autosave.pt'
            save_checkpoint(model, opts, si, schedulers, step, stage, ckpt_path)
            print(f"Session budget reached. Saved to {ckpt_path.name}. Stopping.")
            break
        
        # ── Gradient clip helper (cfloat-safe) ─────────────────────────
        def clip_grads(max_norm=1.0):
            # PyTorch clip_grad_norm_ works on cfloat params correctly
            # (treats real+imag separately under the hood via view_as_real)
            all_params = [p for group in opts['opt_g'].param_groups
                         for p in group['params'] if p.grad is not None]
            all_params += [p for group in opts['opt_u'].param_groups
                          for p in group['params'] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        
        # ── Training step ───────────────────────────────────────────────
        batch = sample_batch(data, cfg['batch_size'], cfg.get('T', 256))
        loss, info = train_fn(batch, step)
        
        # ── Gradient clip ───────────────────────────────────────────────
        clip_grads(max_norm=1.0)
        
        # ── Schedulers ──────────────────────────────────────────────────
        schedulers['sched_g'].step()
        schedulers['sched_u'].step()
        
        # ── OQ probes ───────────────────────────────────────────────────
        probes.log(info, loss)
        loss_hist.append(float(loss))
        
        # ── Logging ─────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            avg = sum(loss_hist[-LOG_EVERY:]) / min(len(loss_hist), LOG_EVERY)
            oq  = probes.summary()
            print(f"[{stage}] {step:6d} | loss={avg:.4f} | "
                  f"U_epi={info.get('U_epistemic',0):.3f} | "
                  f"escape={oq.get('escape_rate',0):.3f} | "
                  f"rule_w={oq.get('rule_write_rate',0):.3f} | "
                  f"n_l={oq.get('n_active_units',0)} | "
                  f"t={elapsed_h:.2f}h")
            wandb.log({'loss': avg, 'stage': stage, 'step': step, **oq})
        
        # ── Periodic checkpoint ─────────────────────────────────────────
        if step % CKPT_EVERY == 0 and step > start_step:
            ckpt_path = CKPT_DIR / f'ckpt_{stage}_{step:06d}.pt'
            save_checkpoint(model, opts, si, schedulers, step, stage, ckpt_path)
    
    return loss_hist, probes
```

---

## 8. EVALUATION LOOP

```python
@torch.no_grad()
def evaluate(model, val_data, cfg, n_batches=100):
    """Perplexity on validation set. Freezes unit growth during eval."""
    model.eval()
    model.reset_for_inference()
    
    # Freeze unit growth (fair comparison across checkpoints)
    original_growth = getattr(model.bank, '_allow_growth', True)
    model.bank._allow_growth = False
    
    total_ce = 0.0
    for _ in range(n_batches):
        batch = sample_batch(val_data, cfg['batch_size'], cfg.get('T', 256))
        out, *_ = model(batch)
        # out: (B, T, vocab). Predict token t+1 from token t
        ce = F.cross_entropy(
            out[:, :-1].reshape(-1, cfg['vocab_size']),
            batch[:, 1:].reshape(-1)
        )
        total_ce += float(ce.item())
    
    model.bank._allow_growth = original_growth
    model.train()
    
    ppl = (total_ce / n_batches)  # mean CE
    return {'val_ce': ppl, 'val_ppl': torch.exp(torch.tensor(ppl)).item()}
```

---

## 9. STAGE-BY-STAGE EXECUTION

### STAGE 0: LM Warmup
```python
# ── Run this notebook: cfln_stage0_lm_warmup.ipynb ─────────────────────
CFG_S0 = {**CFG_ABLATION_605,
           'n_steps': 10_000, 'batch_size': 8, 'T': 256,
           'lr_local': 3e-4, 'lr_persist': 1e-6, 'lr_unit': 1e-3,
           'warmup_steps': 500}

wandb.init(project='cfln', name='stage0_lm_warmup', config=CFG_S0)
model = setup_model(CFG_S0, THINK_START_ID, THINK_END_ID)
opts, si, scheds = setup_training(model, CFG_S0)

# Resume if checkpoint exists:
# start_step, _ = load_checkpoint(CKPT_DIR/'ckpt_stage0_005000.pt', model, opts, si, scheds)

loss_hist, probes = training_loop(model, opts, si, scheds, train_data, CFG_S0,
                                   stage='stage0', start_step=0)
# Evaluate
print(evaluate(model, val_data, CFG_S0))
save_checkpoint(model, opts, si, scheds, 10_000, 'stage0',
                CKPT_DIR/'ckpt_stage0_final.pt')
# Upload to dataset before session ends
```

**Stop signal:** val_ppl stable for 1K steps (usually ~8-10K steps at this scale)

---

### STAGE 1: PSC Pretraining
```python
# Load Stage 0 final — keep optimizer + SI (DO NOT reset)
start_step, _ = load_checkpoint(INPUT/'cfln-checkpoints/ckpt_stage0_final.pt',
                                 model, opts, si, scheds)
CFG_S1 = {**CFG_PSC, 'n_steps': 5_000}
wandb.init(project='cfln', name='stage1_psc', config=CFG_S1)

loss_hist, probes = training_loop(model, opts, si, scheds, train_data, CFG_S1,
                                   stage='stage1', start_step=0,
                                   train_fn=lambda b, s: psc_train_step(b, model, opts, si, cfg=CFG_S1))
save_checkpoint(model, opts, si, scheds, 5_000, 'stage1',
                CKPT_DIR/'ckpt_stage1_final.pt')
```

**Stop signal:** `L_improve / L_LM < 0.05` in wandb — thinking chain working.
**If stuck:** increase `psc_alpha` (1.0 → 2.0) or lower `u_epi_psc_threshold` (0.5 → 0.3)

---

### STAGE 2: RPP-STaR Trace Generation (inference only, fast)
```python
# Load Stage 1. No optimizer needed — pure inference.
model.eval(); model.reset_for_inference()

# Sample prompts from validation set
prompts = [val_data[i][:64] for i in range(0, 1000, 1)]  # 1K prompts

traces = star_generate_traces_rpp(model, prompts, n_traces_target=5_000,
                                    n_think=8, cfg=CFG_PSC)
# traces is a list of dicts:
# {'prompt_ids': Tensor, 'think_ids': Tensor, 'completion_ids': Tensor,
#  'reward': float, 'accepted': bool}

accepted = [t for t in traces if t['accepted']]
print(f"Acceptance rate: {len(accepted)/len(traces):.1%}  (target: >70%)")
# If <50%: model PSC didn't converge. Re-run Stage 1 with more steps.

torch.save(traces, CKPT_DIR/'rpp_traces.pt')
```

---

### STAGE 3: SFT on RPP Traces
```python
traces = torch.load(INPUT/'cfln-checkpoints/rpp_traces.pt')
accepted_traces = [t for t in traces if t['accepted']]
CFG_S3 = {**CFG_PSC, 'n_steps': 2_000, 'T': 512}

loss_hist, _ = training_loop(model, opts, si, scheds, accepted_traces, CFG_S3,
                               stage='stage3', start_step=0,
                               train_fn=lambda b, s: sft_train_step_ctp(
                                   b, model, opts, si, cfg=CFG_S3))
save_checkpoint(model, opts, si, scheds, 2_000, 'stage3',
                CKPT_DIR/'ckpt_stage3_final.pt')
```

---

### STAGE 4: GRPO Finetuning (CRITICAL setup — ref_model)
```python
# BEFORE Stage 4: make a frozen copy of Stage 3 model as KL reference
load_checkpoint(INPUT/'cfln-checkpoints/ckpt_stage3_final.pt',
                model, opts, si, scheds)

ref_model = copy.deepcopy(model)  # ← CRITICAL: freeze reference
ref_model.eval()
for p in ref_model.parameters(): p.requires_grad_(False)

CFG_S4 = {**CFG_GRPO, 'n_steps': 1_000}

def grpo_step(batch, step):
    return grpo_train_step(batch, model, ref_model, opts, cfg=CFG_S4)

loss_hist, probes = training_loop(model, opts, si, scheds, train_data, CFG_S4,
                                   stage='stage4', start_step=0,
                                   train_fn=grpo_step)
save_checkpoint(model, opts, si, scheds, 1_000, 'stage4',
                CKPT_DIR/'ckpt_stage4_final.pt')
```

**Stop signal:** mean GRPO reward positive and stable for 500 steps.
**KL blowup warning:** if `grpo_beta_kl` is too low, KL grows unbounded. Increase to 0.3.

---

## 10. ABLATION SERIES (priority order from DA review)

**Run all from SAME Stage 0 checkpoint. Keep SI, reset optimizer state for fairness.**

```python
def ablation_reset_optimizer(model, cfg):
    """Reset optimizer state between ablations (fair comparison)."""
    opts = build_optimizers_v605(model, cfg)
    scheds = {
        'sched_g': torch.optim.lr_scheduler.LambdaLR(opts['opt_g'],
                   lambda s: min(1.0, s/cfg.get('warmup_steps',500))),
        'sched_u': torch.optim.lr_scheduler.LambdaLR(opts['opt_u'],
                   lambda s: min(1.0, s/cfg.get('warmup_steps',500))),
    }
    return opts, scheds
```

### A85: PSC — L_improve only (baseline)
```python
load_checkpoint(INPUT/'cfln-checkpoints/ckpt_stage0_final.pt', model, opts, si, scheds)
opts, scheds = ablation_reset_optimizer(model, CFG_PSC)
CFG_A85 = {**CFG_PSC, 'psc_beta_max': 0.0, 'psc_gamma': 0.0, 'n_steps': 5_000}
# psc_beta_max=0 disables L_economy. psc_gamma=0 disables L_predictive.
wandb.init(project='cfln', name='A85_psc_improve_only', config=CFG_A85)
```

### A86: PSC — L_improve + L_economy (two-term)
```python
load_checkpoint(INPUT/'cfln-checkpoints/ckpt_stage0_final.pt', model, opts, si, scheds)
opts, scheds = ablation_reset_optimizer(model, CFG_PSC)
CFG_A86 = {**CFG_PSC, 'psc_gamma': 0.0, 'n_steps': 5_000}
# psc_gamma=0 disables L_predictive only.
wandb.init(project='cfln', name='A86_psc_improve_economy', config=CFG_A86)
```

### A87: Full PSC (all three terms) — already done as Stage 1
```python
# This IS Stage 1. No extra run needed. Use Stage 1 wandb run as A87.
```

### A88: PSC + SFT (no GRPO) — already done after Stage 3
```python
# This IS Stage 3 final. Use Stage 3 final checkpoint eval as A88.
```

### A89: PSC + SFT + GRPO (full pipeline) — already done after Stage 4
```python
# This IS Stage 4 final. Use Stage 4 final checkpoint eval as A89.
```

### A90: Cold STaR + GRPO (no PSC) — baseline comparison
```python
load_checkpoint(INPUT/'cfln-checkpoints/ckpt_stage0_final.pt', model, opts, si, scheds)
opts, scheds = ablation_reset_optimizer(model, CFG_GRPO)
# Skip PSC entirely: random STaR traces (acceptance ~15% vs PSC's ~70-90%)
cold_traces = star_generate_traces_rpp(model, prompts, n_traces_target=5_000,
                                        use_psc=False, n_think=8, cfg=CFG_GRPO)
wandb.init(project='cfln', name='A90_cold_star_grpo', config=CFG_GRPO)
```

### A93: 2-tier (current) vs 3-tier CFLN
```python
# CURRENT: 2-tier — already trained in Stages 0-4 ✓

# VARIANT: restore global tier for comparison
CFG_3TIER = {**CFG_ABLATION_605, 'n_g': 64, 'd_e_g': 32, 'n_l': 256}
# Freeze n_l growth for fair per-token flop comparison:
# measure actual tokens/sec with torch.cuda.Event() timer
model_3tier = setup_model(CFG_3TIER, THINK_START_ID, THINK_END_ID)
# Train 5K steps from scratch on same data, same seed

# Measure per-token time:
def measure_tokens_per_sec(model, batch, n_warmup=50, n_measure=200):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup): model(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_measure): model(batch)
        torch.cuda.synchronize()
    return (n_measure * batch.shape[0] * batch.shape[1]) / (time.perf_counter() - t0)

wandb.init(project='cfln', name='A93_2tier_vs_3tier')
```

---

## 11. SESSION PLAN (3 sessions to trained model)

```
SESSION 1 (internet ON, 12h max):
  Cell 1:  Setup, install, secrets
  Cell 2:  Build tokenizer (WikiText), save to working dir
  Cell 3:  Tokenize + chunk train/val, save .pt files
  Cell 4:  Sanity check (CFG_VERIFY_605, 10 steps, eval)
  Cell 5:  Stage 0 — LM warmup (10K steps, ~2h)
  Cell 6:  Stage 1 — PSC pretrain (5K steps, ~1.5h)
  → Save: upload stage0_final + stage1_final to dataset before session ends

SESSION 2 (internet OFF, 9h):
  Cell 1:  Load stage1_final from /kaggle/input/cfln-checkpoints/
  Cell 2:  Stage 2 — RPP-STaR traces (~45min)
  Cell 3:  Stage 3 — SFT (~45min)
  Cell 4:  Stage 4 — GRPO (~2h)  [ref_model copy is CRITICAL]
  Cell 5:  Ablations A85, A86 (~2h total, from stage0 checkpoint)
  → Save: upload stage3_final, stage4_final, A85, A86 logs

SESSION 3 (internet OFF, 9h):
  Cell 1:  Load checkpoints
  Cell 2:  Ablation A90 — Cold STaR baseline (~2h)
  Cell 3:  Ablation A93 — 3-tier variant training (~1.5h)
  Cell 4:  Comparative evaluation: A87/A88/A89/A90 vs A85/A86
  Cell 5:  A93: per-token timing measurement
  → Final results and wandb summary
```

---

## 12. OQ MEASUREMENT SUMMARY (collected automatically in training loop)

| OQ | Metric logged | Target |
|---|---|---|
| OQ-v598-2: U_epi calibration | `u_epi_ce_corr` (Pearson U_epi ↔ CE) | >0.4 |
| OQ-v598-4: rule cache escape | `escape_rate`, `rule_write_rate` | escape↓ over training |
| OQ-v600-1: thinking coherence | thinking token CE (in sft_train_step) | ↓ over SFT |
| OQ-v596-3: domain drift events | `domain_shifts` count | fires on doc boundaries |
| OQ-CONSOL-1: unit growth rate | `n_active_units` | grows then stabilises |

---

## 13. QUICK SANITY CHECKS (run before anything else)

```python
# Must all pass before starting Stage 0:

def sanity_check(model, cfg):
    model.eval()
    batch = torch.randint(0, cfg['vocab_size'], (2, 16)).to(DEVICE)
    
    # 1. Forward pass doesn't crash
    with torch.no_grad():
        out, z, u, info = model(batch)
    assert out.shape == (2, 16, cfg['vocab_size']), f"Bad output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    print(f"✓ Forward: {out.shape}, U_epi={info.get('U_epistemic',0):.3f}")
    
    # 2. One train step doesn't NaN
    model.train()
    opts_test = build_optimizers_v605(model, cfg)
    si_test = SynapticIntelligence(model, c_SI=0.5, rho=0.999)
    si_test.record_theta_0(model)
    loss, info = train_step_v605(batch, model, opts_test, si_test, phase=1, cfg=cfg)
    assert not torch.isnan(torch.tensor(loss)), f"NaN loss: {loss}"
    print(f"✓ Train step: loss={loss:.4f}")
    
    # 3. Checkpoint round-trip
    path = WORK/'sanity_ckpt.pt'
    scheds_test = {'sched_g': torch.optim.lr_scheduler.LambdaLR(opts_test['opt_g'], lambda s: 1.0),
                   'sched_u': torch.optim.lr_scheduler.LambdaLR(opts_test['opt_u'], lambda s: 1.0)}
    save_checkpoint(model, opts_test, si_test, scheds_test, 0, 'sanity', path)
    load_checkpoint(path, model, opts_test, si_test, scheds_test)
    print("✓ Checkpoint save/load round-trip")
    
    # 4. GPU memory
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"✓ GPU used: {mem:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    print("\n✓ All sanity checks passed. Ready to train.")

sanity_check(CFLNModel(CFG_VERIFY_605).to(DEVICE), CFG_VERIFY_605)
```
