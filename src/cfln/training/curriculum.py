"""curriculum.py — Three-domain CL training protocol and curriculum sampler (§3).

The spec (§3 / §1.29.1) describes a 4-stage PSC-RPP-RL pipeline:
  Stage 0: Standard LM pre-training (train_step_v605)
  Stage 1: PSC pre-training (10% of budget, psc_train_step)
  Stage 2: RPP-STaR trace generation (offline)
  Stage 3: SFT on RPP traces (sft_train_step_ctp)
  Stage 4: GRPO fine-tuning (optional)

The three-domain CL protocol (§5 Phase 3) cycles through domains:
  LANG (natural language), MATH (arithmetic/logic), CODE (programming).
"""
import random
from typing import List, Dict, Any, Optional


DOMAINS = ['LANG', 'MATH', 'CODE']

# Stages as described in §1.29.1
STAGE_NAMES = {
    0: 'pretrain',
    1: 'psc',
    2: 'rpp_trace',
    3: 'sft_ctp',
    4: 'grpo',
}


class CurriculumSampler:
    """Three-domain curriculum sampler for continual-learning evaluation.

    Cycles through LANG → MATH → CODE domains, with configurable steps per
    domain. During evaluation uses a held-out interleaved stream.

    Args:
        domain_datasets: dict mapping domain name to list of batch-dicts.
        steps_per_domain: number of gradient steps per domain per cycle.
        shuffle: whether to shuffle within each domain epoch.
        seed: random seed for reproducibility.
    """

    def __init__(self, domain_datasets: Dict[str, List[Any]],
                 steps_per_domain: int = 1000,
                 shuffle: bool = True,
                 seed: int = 42):
        self.domain_datasets = domain_datasets
        self.steps_per_domain = steps_per_domain
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._domain_order = list(DOMAINS)
        self._reset_iterators()

    def _reset_iterators(self):
        self._iters: Dict[str, Any] = {}
        for domain in self._domain_order:
            if domain in self.domain_datasets:
                data = list(self.domain_datasets[domain])
                if self.shuffle:
                    self._rng.shuffle(data)
                self._iters[domain] = iter(data)

    def _next_from_domain(self, domain: str):
        """Advance iterator for domain; restart on exhaustion."""
        try:
            return next(self._iters[domain])
        except StopIteration:
            data = list(self.domain_datasets[domain])
            if self.shuffle:
                self._rng.shuffle(data)
            self._iters[domain] = iter(data)
            return next(self._iters[domain])

    def generate(self, total_steps: int):
        """Yield (step, domain, batch) for total_steps gradient steps.

        Domain cycles: LANG for steps_per_domain, then MATH, then CODE, repeat.
        """
        step = 0
        domain_idx = 0
        domain_steps = 0
        domains_available = [d for d in self._domain_order
                              if d in self.domain_datasets]
        if not domains_available:
            raise ValueError('No domain datasets provided.')
        while step < total_steps:
            domain = domains_available[domain_idx % len(domains_available)]
            batch = self._next_from_domain(domain)
            yield step, domain, batch
            step += 1
            domain_steps += 1
            if domain_steps >= self.steps_per_domain:
                domain_idx += 1
                domain_steps = 0

    def eval_stream(self, n_samples_per_domain: int = 100):
        """Yield (domain, batch) for evaluation — one pass each domain."""
        for domain in self._domain_order:
            if domain not in self.domain_datasets:
                continue
            data = list(self.domain_datasets[domain])
            for batch in data[:n_samples_per_domain]:
                yield domain, batch


class CFLNCurriculumTrainer:
    """4-stage PSC-RPP-RL curriculum trainer wrapping train_step functions.

    Stage 0 (pretrain):  standard LM pre-training via train_step_v605.
    Stage 1 (psc):       PSC pre-training via psc_train_step.
    Stage 2 (rpp_trace): offline RPP trace generation (call generate_traces()).
    Stage 3 (sft_ctp):   SFT on RPP traces via sft_train_step_ctp.
    Stage 4 (grpo):      GRPO fine-tuning via grpo_train_step (optional).
    """

    # Fraction of total budget per stage (Stage 2 is offline, no step budget)
    STAGE_FRACTIONS = {0: 0.80, 1: 0.10, 2: 0.00, 3: 0.08, 4: 0.02}

    def __init__(self, model, opts, si, cfg: dict,
                 sampler: Optional[CurriculumSampler] = None):
        self.model = model
        self.opts = opts
        self.si = si
        self.cfg = cfg
        self.sampler = sampler
        self._step = 0
        self._stage = 0
        self._rpp_traces: List[Any] = []

    def _get_stage(self, step: int, total_steps: int) -> int:
        """Determine training stage from step fraction."""
        frac = step / max(total_steps, 1)
        if frac < self.STAGE_FRACTIONS[0]:
            return 0
        elif frac < self.STAGE_FRACTIONS[0] + self.STAGE_FRACTIONS[1]:
            return 1
        elif frac < (self.STAGE_FRACTIONS[0]
                     + self.STAGE_FRACTIONS[1]
                     + self.STAGE_FRACTIONS[3]):
            return 3
        return 4

    def step(self, batch: dict, total_steps: int,
             doc_ctx=None, psc_loss_fn=None, ref_model=None) -> dict:
        """Execute one training step at the current curriculum stage."""
        from cfln.training.train_step import train_step_v605, psc_train_step

        stage = self._get_stage(self._step, total_steps)
        self._stage = stage

        if stage == 0:
            info = train_step_v605(
                batch, self.model, self.opts, self.si,
                phase='pretrain', step=self._step,
                total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)
        elif stage == 1:
            if psc_loss_fn is None:
                # Fall back to standard step if PSCLoss not provided
                info = train_step_v605(
                    batch, self.model, self.opts, self.si,
                    phase='psc', step=self._step,
                    total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)
            else:
                info = psc_train_step(
                    batch, self.model, psc_loss_fn, self.opts, self.si,
                    phase='psc', step=self._step,
                    total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)
        elif stage == 3:
            if self._rpp_traces:
                from cfln.training.train_step import sft_train_step_ctp
                # Use a small local import to avoid circular dep; defined in spec §3.2b
                info = sft_train_step_ctp(
                    self._rpp_traces[:self.cfg.get('sft_batch_size', 4)],
                    self.model, self.opts, self.si, self.cfg)
            else:
                info = train_step_v605(
                    batch, self.model, self.opts, self.si,
                    phase='sft', step=self._step,
                    total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)
        elif stage == 4:
            if ref_model is not None:
                from cfln.training.train_step import grpo_train_step
                info = grpo_train_step(
                    batch, self.model, ref_model, self.opts, self.cfg)
            else:
                info = train_step_v605(
                    batch, self.model, self.opts, self.si,
                    phase='grpo', step=self._step,
                    total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)
        else:
            info = train_step_v605(
                batch, self.model, self.opts, self.si,
                phase='pretrain', step=self._step,
                total_steps=total_steps, cfg=self.cfg, doc_ctx=doc_ctx)

        info['stage'] = stage
        info['stage_name'] = STAGE_NAMES.get(stage, 'unknown')
        self._step += 1
        return info

    def generate_traces(self, dataset_items: list,
                         n_think: int = 8, n_opt: int = 10,
                         max_traces: int = 10000) -> list:
        """Stage 2: generate RPP-STaR traces offline.

        Stores traces internally for use in Stage 3 SFT.
        """
        from cfln.training.rpp import star_generate_traces_rpp
        self._rpp_traces = star_generate_traces_rpp(
            self.model, dataset_items,
            n_think=n_think, n_opt=n_opt, max_traces=max_traces)
        return self._rpp_traces

    def eval_domains(self, domain_batches: Dict[str, List[dict]],
                     device='cpu') -> Dict[str, float]:
        """3-domain eval: compute mean cross-entropy per domain."""
        import torch
        import torch.nn.functional as F
        results: Dict[str, float] = {}
        self.model.eval()
        for domain, batches in domain_batches.items():
            losses = []
            for batch in batches:
                input_ids = batch['input_ids'].to(device)
                with torch.no_grad():
                    logits, _, _ = self.model(input_ids, training=False)
                targets = input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='mean',
                    ignore_index=self.cfg.get('pad_id', -100))
                losses.append(float(ce))
            results[domain] = sum(losses) / max(len(losses), 1)
        self.model.train()
        return results
