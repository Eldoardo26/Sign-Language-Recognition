"""
decoding.py — Algoritmi di decodifica CTC con prior scaling (Deep Sign Eq.13).

Funzioni:
    beam_search_ctc_optimized      — Beam search CTC con prior scaling
    greedy_decode_with_prior       — Greedy decode con prior scaling (veloce)
    greedy_decode_with_bigram      — Greedy + bigram LM rescoring
    create_scheduler_warmup_cosine — LR scheduler warmup + cosine
"""

import numpy as np
import torch
from scipy.special import logsumexp

from utils import collapse_ctc


# ============================================================
# BEAM SEARCH CTC
# ============================================================

def beam_search_ctc_optimized(
    log_probs_TBC: torch.Tensor,
    beam_width: int = 10,
    beta: float = 0.3,
    log_prior: torch.Tensor = None,
) -> list:
    """
    Beam search CTC con prior scaling (Deep Sign Eq.13).

    Lo scaling del prior sottrae beta * log_prior dai log-prob acustici
    e li rinormalizza prima della ricerca, riducendo il bias verso classi
    frequenti come <blank>.

    Args:
        log_probs_TBC : Tensor (T, B, C) — log-probabilità acustiche
        beam_width    : ampiezza del beam
        beta          : peso del prior scaling (0 = disabilitato)
        log_prior     : Tensor (C,) — log prior delle classi (o None)

    Returns:
        list di B sequenze di indici (senza blank, senza duplicati adiacenti)
    """
    T, B, C = log_probs_TBC.shape
    results = []

    for b in range(B):
        lp = log_probs_TBC[:, b, :].cpu().numpy()

        # Prior scaling
        if beta > 0.0 and log_prior is not None:
            lp_prior = log_prior.cpu().numpy()
            lp       = lp - beta * lp_prior.reshape(1, -1)
            lp       = lp - logsumexp(lp, axis=1, keepdims=True)

        # Beam search
        beams = {((), None): 0.0}

        for t in range(T):
            new_beams = {}
            for (prefix, last_char), beam_lp in beams.items():
                for c in range(C):
                    new_lp = beam_lp + lp[t, c]
                    if c == 0:
                        key = (prefix, last_char)
                    elif c == last_char:
                        continue
                    else:
                        key = (prefix + (c,), c)
                    if key not in new_beams or new_beams[key] < new_lp:
                        new_beams[key] = new_lp

            sorted_beams = sorted(new_beams.items(), key=lambda x: -x[1])
            beams        = dict(sorted_beams[:beam_width])

        best_key = max(beams.keys(),
                       key=lambda k: beams[k] / max(len(k[0]), 1) ** 0.7)
        results.append(list(best_key[0]))

    return results


# ============================================================
# GREEDY DECODE
# ============================================================

def greedy_decode_with_prior(
    log_probs_TBC: torch.Tensor,
    beta: float = 0.0,
    log_prior: torch.Tensor = None,
) -> list:
    """
    Greedy CTC decode con prior scaling opzionale (più veloce del beam).

    Args:
        log_probs_TBC : Tensor (T, B, C)
        beta          : peso del prior (0 = disabilitato)
        log_prior     : Tensor (C,) o None

    Returns:
        list di B sequenze di indici
    """
    T, B, C = log_probs_TBC.shape
    results = []
    for b in range(B):
        lp = log_probs_TBC[:, b, :].clone()
        if beta > 0.0 and log_prior is not None:
            lp = lp - beta * log_prior.unsqueeze(0)
        best_ids = lp.argmax(dim=-1).tolist()
        results.append(collapse_ctc(best_ids))
    return results


# ============================================================
# GREEDY + BIGRAM LM RESCORING
# ============================================================

def greedy_decode_with_bigram(
    log_probs_TBC: torch.Tensor,
    beta: float = 0.0,
    log_prior: torch.Tensor = None,
    log_bigram: dict = None,
    alpha: float = 0.3,
) -> list:
    """
    Greedy decode con bigram rescoring: per ogni frame combina
    lo score acustico con il log-bigram P(c | prev).

    Args:
        log_probs_TBC : Tensor (T, B, C)
        beta          : prior scaling weight
        log_prior     : Tensor (C,) o None
        log_bigram    : {prev_idx: {next_idx: log_prob}}
        alpha         : peso del bigram LM

    Returns:
        list di B sequenze di indici
    """
    T, B, C = log_probs_TBC.shape
    results = []

    for b in range(B):
        lp = log_probs_TBC[:, b, :].clone()
        if beta > 0.0 and log_prior is not None:
            lp = lp - beta * log_prior.unsqueeze(0)

        seq  = []
        prev = None

        for t in range(T):
            if log_bigram and alpha > 0 and prev is not None and prev in log_bigram:
                bigram_scores = torch.tensor(
                    [log_bigram[prev].get(c, -10.0) for c in range(C)],
                    dtype=torch.float32,
                )
                combined = lp[t] + alpha * bigram_scores
                best     = combined.argmax().item()
            else:
                best = lp[t].argmax().item()

            seq.append(best)
            if best != 0:
                prev = best

        results.append(collapse_ctc(seq))

    return results


# ============================================================
# LR SCHEDULER
# ============================================================

def create_scheduler_warmup_cosine(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Scheduler con warmup lineare + cosine annealing.

    Args:
        optimizer     : istanza di torch.optim
        total_steps   : numero totale di step
        warmup_steps  : numero di step di warmup
        min_lr_ratio  : fattore minimo di LR al termine del cosine

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress     = float(current_step - warmup_steps) / \
                       float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
