# coding: utf-8
"""
vocab_utils.py — canonical gloss-merge and shared-vocabulary construction.

Single source of truth for the deterministic gloss-merge policy used by the
cross-modal distillation modules. The teacher notebook (csrl_skeleton/
tssi75_cslr.ipynb) implements the same policy; keep them in sync (same default
threshold 0.85, same hyphen-compaction rule).
"""
from difflib import SequenceMatcher

import torch


def build_merge_map(tokens, threshold: float = 0.85) -> dict:
    """Deterministic gloss-merge policy.

    Orthographically similar glosses (SequenceMatcher ratio >= threshold) are
    merged towards the alphabetically first form; hyphenated forms merge into
    their compact variant when both exist. Merge chains (a->b, b->c => a->c)
    are resolved; on cycles the lexicographically smallest member is kept as the
    canonical form.

    Returns a dict {token: canonical_token} for the tokens that get merged.
    """
    toks = sorted(tokens)
    mm = {}
    for i, g1 in enumerate(toks):
        if g1 in mm:
            continue
        for g2 in toks[i + 1:]:
            if g2 in mm:
                continue
            if SequenceMatcher(None, g1, g2).ratio() >= threshold:
                mm[g2] = g1
    for t in toks:
        compact = t.replace("-", "")
        if "-" in t and compact in tokens and compact != t and t not in mm:
            mm[t] = compact
    resolved = {}
    for k in mm:
        seen, cur = [k], mm[k]
        while cur in mm and cur not in seen:
            seen.append(cur)
            cur = mm[cur]
        if cur in seen:                      # cycle: canonicalise deterministically
            cur = min(seen)
        for s in seen:
            if s != cur:
                resolved[s] = cur
    return resolved


def build_shared_vocab(gls_vocab, threshold: float = 0.85):
    """Build the shared gloss vocabulary and a student-id -> shared-id lookup.

    Returns:
        vocab : list[str]   — ['<blank>', '<unk>', gloss_1, ...]
        lut   : LongTensor  — lut[student_id] = shared_id
    """
    itos = list(gls_vocab.itos)
    specials = {"<si>", "<unk>", "<pad>"}
    tokens = {t for t in itos if t not in specials}
    mm = build_merge_map(tokens, threshold)
    merged = sorted({mm.get(t, t) for t in tokens})
    vocab = ["<blank>", "<unk>"] + merged
    g2i = {g: i for i, g in enumerate(vocab)}
    lut = []
    for t in itos:
        if t in ("<si>", "<pad>"):
            lut.append(0)                       # blank / padding -> shared blank
        elif t == "<unk>":
            lut.append(1)
        else:
            lut.append(g2i.get(mm.get(t, t), 1))
    return vocab, torch.LongTensor(lut)
