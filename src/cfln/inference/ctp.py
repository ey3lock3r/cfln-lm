"""ctp.py — CTP inference: generate_cfln_ctp, _mark_think_positions, compute_ctp_loss (§4)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _mark_think_positions(targets: torch.Tensor, start_id: int, end_id: int
                          ) -> torch.Tensor:
    """v6.0 CTP: Return bool tensor marking positions BETWEEN <think> and </think>."""
    device = targets.device
    T = targets.shape[-1]
    is_think = torch.zeros(T, dtype=torch.bool, device=device)
    in_think = False
    for t in range(T):
        tok = int(targets.flat[t]) if targets.dim() == 1 else int(targets[0, t].item())
        if tok == start_id:
            in_think = True
        is_think[t] = in_think
        if tok == end_id:
            in_think = False
    return is_think


def compute_ctp_loss(logits: torch.Tensor, targets: torch.Tensor,
                      think_start_id: int, think_end_id: int,
                      tau_think: float = 0.5) -> torch.Tensor:
    """v6.0 CTP / v6.0.1 C3 fix: Cross-entropy with per-batch-item thinking weights.
    logits: (B, T, V), targets: (B, T).
    tau_think=0.5 for STaR/SFT; tau_think=0 (+ KL) for GRPO phase.
    Weight for <think>, interior thinking, AND </think> = tau_think.
    All other positions weight = 1.0.
    FIXED: per-batch-item weights (v6.0 used targets[0] for all B sequences).
    """
    B, T, V = logits.shape
    # Build per-batch-item (B, T) weight tensor — each sequence tracked independently
    weights_bt = torch.ones(B, T, device=logits.device, dtype=logits.dtype)
    tgt_1d = targets if targets.dim() == 1 else None
    for b in range(B):
        seq = tgt_1d if (tgt_1d is not None) else targets[b]   # (T,)
        is_think_b = _mark_think_positions(seq, think_start_id, think_end_id)
        weights_bt[b] = torch.where(is_think_b,
            torch.full((T,), tau_think, device=logits.device, dtype=logits.dtype),
            torch.ones(T, device=logits.device, dtype=logits.dtype))
        if tgt_1d is not None:
            break   # 1D input: only one sequence
    loss_per_tok = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T),
                                    reduction='none')    # (B*T,)
    return (loss_per_tok * weights_bt.reshape(B * T)).mean()


def _sample_block(logits, temperature=1.0, top_k=50):
    """v5.9.9 DCG+: Sample or argmax from logits block (B, M, V)."""
    B, M, V = logits.shape
    lg = logits / max(float(temperature), 1e-8)
    if top_k > 0:
        tv = torch.topk(lg, min(top_k, V), dim=-1)
        lg = lg.masked_fill(lg < tv.values[..., -1:], float('-inf'))
    probs = torch.softmax(lg, dim=-1)
    tokens = torch.multinomial(probs.reshape(B * M, V), 1).reshape(B, M)
    return tokens, probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def generate_cfln_ctp(model, prompt_ids,
                       max_new_tokens: int = 100,
                       max_think_tokens: int = 64,
                       think_threshold: float = 0.5,
                       temperature: float = 1.0,
                       top_k: int = 50,
                       use_refinement: bool = True,
                       show_thinking: bool = False):
    """v6.0 CTP: CFLN Think Protocol generation.
    Generates thinking tokens (gated by U_epistemic) before each output token.
    Thinking tokens update r_lista/rho_l/H_seq but NOT h_cache/rule/Titans M/Telescoping.
    The r_lista reasoning chain across thinking tokens is the core CoT mechanism.
    REQUIRES: model.expand_vocabulary() called first (THINK_START_ID >= 0).
    show_thinking=True: return full sequence including thinking tokens.
    show_thinking=False: return only committed output tokens (default).
    Note: returned sequence INCLUDES prompt tokens (slice [:, prompt_len:] for output only).
    """
    # v6.0.1 H5: validate both thinking token IDs before generation
    assert model.THINK_START_ID >= 0 and model.THINK_END_ID >= 0, (
        'Call model.expand_vocabulary() before generate_cfln_ctp(). '
        f'Got THINK_START_ID={model.THINK_START_ID}, THINK_END_ID={model.THINK_END_ID}')
    assert model.THINK_END_ID == model.THINK_START_ID + 1, (
        f'THINK_END_ID must be THINK_START_ID+1 (got {model.THINK_END_ID} vs {model.THINK_START_ID}+1)')
    THINK_START = model.THINK_START_ID
    THINK_END = model.THINK_END_ID
    model.training = False   # avoid recursion from shared submodule refs in model.eval()
    device = prompt_ids.device
    model.reset_for_inference()
    generated = prompt_ids.clone()         # full sequence (thinking + output)
    output_only = prompt_ids.clone()       # display sequence (output only)
    B = generated.shape[0]

    def _set_thinking(flag: bool):
        model._in_thinking_mode = flag
        model.diff_aux.cun._in_thinking_mode = flag
        if hasattr(model.encoder, 'titans'):
            model.encoder.titans._in_thinking_mode = flag
            if not flag:
                # v6.0.1 H4/M9: clear chunk_accum on exit — prevents thinking-token
                # embeddings from contaminating the next real-token Titans M update
                model.encoder.titans._chunk_accum = []

    while generated.shape[1] - prompt_ids.shape[1] < max_new_tokens:
        # ── Assess whether to think ──────────────────────────────────────────
        with torch.no_grad():
            _, _, aux = model(generated[:, -1:], training=False,
                               use_refinement=use_refinement)
        U_epi = float(model.bank._u_epistemic_last)

        if U_epi > think_threshold and THINK_START >= 0:
            # ── Inject <think> token ─────────────────────────────────────────
            ts_tok = torch.full((B, 1), THINK_START, dtype=torch.long, device=device)
            _set_thinking(True)
            with torch.no_grad():
                model(ts_tok, training=False, use_refinement=False)
            generated = torch.cat([generated, ts_tok], dim=1)

            # ── Think loop ───────────────────────────────────────────────────
            n_think = 0
            while n_think < max_think_tokens:
                with torch.no_grad():
                    lg, _, aux_t = model(generated[:, -1:], training=False,
                                          use_refinement=use_refinement)
                # Stop when confident OR model generates </think>
                U_now = float(model.bank._u_epistemic_last)
                if U_now < think_threshold * 0.5:
                    break   # confidence restored

                lg_t = aux_t['logits'][:, -1, :]
                lg_t[:, THINK_START] = float('-inf')   # no nested thinking
                think_tok, _ = _sample_block(lg_t.unsqueeze(1), temperature, top_k)
                think_tok = think_tok.squeeze(1)      # (B,)

                if (think_tok == THINK_END).all():
                    break
                # v6.0.1 C2: removed redundant model(think_tok) call.
                # Next iter's model(generated[:,-1:]) processes think_tok as new last token.
                # Saves n_think forward passes (was 2× per thinking token → now 1×).
                generated = torch.cat([generated, think_tok.unsqueeze(1)], dim=1)
                n_think += 1

            # ── Inject </think> token ─────────────────────────────────────────
            te_tok = torch.full((B, 1), THINK_END, dtype=torch.long, device=device)
            # v6.0.2 H1: do NOT call model(te_tok) here — </think> will be processed
            # exactly ONCE by the output-token assessment: model(generated[:,-1:]) below.
            # (Pre-fix: te_tok was processed here AND again as generated[-1] → double-processing)
            generated = torch.cat([generated, te_tok], dim=1)
            _set_thinking(False)   # exit thinking mode before output token

        # ── Generate output token (thinking mode OFF) ────────────────────────
        with torch.no_grad():
            lg_o, _, aux_o = model(generated[:, -1:], training=False,
                                    use_refinement=use_refinement)
        lg_last = aux_o['logits'][:, -1, :]
        # Mask thinking tokens from output positions
        lg_last[:, THINK_START] = float('-inf')
        lg_last[:, THINK_END] = float('-inf')
        out_tok, _ = _sample_block(lg_last.unsqueeze(1), temperature, top_k)
        out_tok = out_tok.squeeze(1)

        generated = torch.cat([generated, out_tok.unsqueeze(1)], dim=1)
        output_only = torch.cat([output_only, out_tok.unsqueeze(1)], dim=1)

    _set_thinking(False)   # ensure thinking mode reset on exit
    return generated if show_thinking else output_only
