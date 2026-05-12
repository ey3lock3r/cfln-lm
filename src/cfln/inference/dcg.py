"""dcg.py — DCG+ deferred-commitment generation (§4 of CFLN v6.0.9 spec)."""
import torch


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


def generate_cfln_dcg_plus(model, prompt_ids, max_new_tokens=100,
                             block_size=8, max_revise_rounds=2,
                             commit_threshold=None, temperature=1.0,
                             top_k=50, self_consistency_K=1,
                             deep_lista_iters=16, use_refinement=True):
    """v5.9.9/v6.0.2 DCG+: Deferred-Commitment Generation.
    Three-phase protocol: Draft → Reflect → Selective Revision → Commit.
    Zero new training. Synthesises WM2 (block-parallel), Z_val gate, U_epistemic,
    Hopfield confidence, deep LISTA scratchpad, and self-consistency voting.
    Compute advantage: (R+1)×(T+M) vs M×T for standard AR (≈2.6× faster at M=8, R=2).
    """
    model.eval()
    device = prompt_ids.device
    model.reset_for_inference()
    generated = prompt_ids.clone()

    while generated.shape[1] - prompt_ids.shape[1] < max_new_tokens:
        M = min(block_size, max_new_tokens - (generated.shape[1] - prompt_ids.shape[1]))
        if M <= 0:
            break

        # ── PHASE 1: DRAFT ───────────────────────────────────────────────────
        with torch.no_grad():
            logits, _, aux = model(generated, training=False,
                                    use_refinement=use_refinement)
        # v6.0.2 C1: save pos_offset after Phase 1 — revision passes must NOT advance it further
        _pos_after_draft = model._pos_offset
        # §1.74: adaptive commit threshold — sampled once per block from calibrated U_epi
        if commit_threshold is None:
            _u_epi_cal = float(getattr(model.bank, '_u_epistemic_last', 0.5))
            _commit_thr = max(0.1, 1.0 - _u_epi_cal)
        else:
            _commit_thr = float(commit_threshold)

        all_infos_last = aux['all_infos'][-1]   # last CFL layer, all T positions
        T_ctx = len(all_infos_last)
        u_epi = torch.tensor([i.get('U_epistemic', 0.5) for i in all_infos_last],
                              device=device)
        z_val = torch.tensor([i.get('Z_val', 0.5) for i in all_infos_last],
                              device=device)
        u_hop = torch.tensor(aux.get('U_hopfield_per_pos', [0.0] * T_ctx),
                              device=device)

        # Commitment score per position (Dr. K calibration formula)
        z_contrib = 1.0 / (1.0 + z_val.clamp(0.01))    # routing concentration ∈ (0,1]
        w = torch.sigmoid(model.w_commit)           # 3 learned calibration scalars
        commit_full = torch.sigmoid(
            w[0] * (1.0 - u_epi) + w[1] * z_contrib + w[2] * u_hop.clamp(0, 1))
        commit_score = commit_full[-M:].clone()     # last M positions

        # Sample draft tokens from last M positions
        full_logits = aux.get('logits', logits)      # (B,T,V)
        draft_logits = full_logits[:, -M:, :]          # (B,M,V)
        draft_tokens, _ = _sample_block(draft_logits, temperature, top_k)

        # ── PHASE 2: REFLECT — optional self-consistency voting ──────────────
        if self_consistency_K > 1:
            uncertain = (commit_score < _commit_thr).nonzero(as_tuple=True)[0]
            if len(uncertain) > 0:
                alt = [draft_tokens.clone()]
                for _ in range(self_consistency_K - 1):
                    a, _ = _sample_block(draft_logits, temperature * 1.2, top_k)
                    alt.append(a)
                for pos in uncertain.tolist():
                    votes = torch.stack([a[:, pos] for a in alt], dim=0)  # (K,B)
                    draft_tokens[:, pos] = votes.mode(dim=0).values

        # ── PHASE 3: SELECTIVE REVISION ──────────────────────────────────────
        for round_i in range(max_revise_rounds):
            revise_pos = (commit_score < _commit_thr).nonzero(as_tuple=True)[0]
            if len(revise_pos) == 0:
                break

            # v6.0.3 C1: set pos=0 before revision pass (corrected fix)
            # Context tokens[0..T-1] need CRoPE pos 0..T-1 (not T..2T-1)
            # Draft tokens[0..M-1] need CRoPE pos T..T+M-1 (correct with base=0)
            # Titans Q_t uses absolute position → must be correct for memory retrieval
            model._pos_offset = 0   # CORRECTED from v6.0.2 which used _pos_after_draft=T

            # Full context: committed + draft block (all M tokens attend to each other)
            rev_ctx = torch.cat([generated, draft_tokens], dim=1)
            with torch.no_grad():
                logits_r, _, aux_r = model(rev_ctx, training=False,
                                            use_refinement=use_refinement)

            all_infos_r = aux_r['all_infos'][-1][-M:]
            u_epi_r = torch.tensor([i.get('U_epistemic', 0.5) for i in all_infos_r],
                                    device=device)
            z_r = torch.tensor([i.get('Z_val', 0.5) for i in all_infos_r],
                                device=device)
            u_hop_r = torch.tensor(aux_r.get('U_hopfield_per_pos', [0.0] * M)[-M:],
                                    device=device)
            z_c_r = 1.0 / (1.0 + z_r.clamp(0.01))
            commit_r = torch.sigmoid(
                w[0] * (1.0 - u_epi_r) + w[1] * z_c_r + w[2] * u_hop_r.clamp(0, 1))

            new_logits = aux_r.get('logits', logits_r)[:, -M:, :]
            new_tokens, _ = _sample_block(new_logits, temperature, top_k)

            # Dr. V monotonicity: accept only if individual AND block-min improve
            block_min_before = float(commit_score.min().item())
            updated = False
            for pos in revise_pos.tolist():
                if (float(commit_r[pos].item()) > float(commit_score[pos].item()) and
                        float(commit_r[pos].item()) >= block_min_before):
                    draft_tokens[:, pos] = new_tokens[:, pos]
                    commit_score[pos] = commit_r[pos]
                    updated = True
            if not updated:
                break    # no improvement → stop early

        # ── DEEP LISTA SCRATCHPAD (Dr. L/D) ──────────────────────────────────
        # For positions still uncertain: run extra LISTA iterations as implicit thinking
        # These update r_lista WITHOUT generating tokens — continuous scratchpad writes
        still_uncertain = (commit_score < _commit_thr).nonzero(as_tuple=True)[0]
        if len(still_uncertain) > 0 and deep_lista_iters > 0:
            last_pos = int(still_uncertain[-1].item())
            re_ctx = torch.cat([generated, draft_tokens[:, :last_pos + 1]], dim=1)
            with torch.no_grad():
                # Run with extra LISTA depth — writes deeper h_N to r_lista
                try:   # v6.0.2 M3: always restore even if model() raises
                    model.diff_aux.cun.N_iter_override = deep_lista_iters
                    model(re_ctx, training=False, use_refinement=True)
                finally:
                    model.diff_aux.cun.N_iter_override = None   # restored on exception too

        # ── COMMIT BLOCK ─────────────────────────────────────────────────────
        # v6.0.3 C1+M5: set _pos_offset = new context length (AFTER cat)
        # This is exact: prompt_len + K*M after K committed blocks
        generated = torch.cat([generated, draft_tokens], dim=1)
        model._pos_offset = generated.shape[1]   # exact context length

    return generated
