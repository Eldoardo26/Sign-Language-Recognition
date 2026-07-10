"""Vocabulary construction with automatic gloss merge (SequenceMatcher + hyphen)."""

import math
from collections import Counter
from difflib import SequenceMatcher
import torch

PAD_IDX, UNK_IDX = 0, 1


def build_merge_map(glosses: set[str], similarity_threshold: float = 0.85) -> dict[str, str]:
    gl = sorted(glosses)
    mm = {}
    for i, g1 in enumerate(gl):
        if g1 in mm:
            continue
        for g2 in gl[i + 1:]:
            if g2 in mm:
                continue
            if SequenceMatcher(None, g1, g2).ratio() >= similarity_threshold:
                mm[g2] = g1
    return mm


def _apply_safe_hyphen_merges(glosses: set[str]) -> dict[str, str]:
    safe = {}
    for t in sorted(glosses):
        compact = t.replace("-", "")
        if "-" in t and compact in glosses and compact != t:
            safe[t] = compact
    return safe


def _toks(g: str) -> list[str]:
    return str(g).strip().upper().split()


def build_vocab_from_raw(train_raw: dict, dev_raw: dict, test_raw: dict,
                         sim_thr: float = 0.85) -> dict:
    allg = set()
    for raw in (train_raw, dev_raw, test_raw):
        for s in raw.values():
            allg.update(_toks(s["gloss"]))

    mm = build_merge_map(allg, sim_thr)
    n_hyphen = 0
    for k, v in _apply_safe_hyphen_merges(allg).items():
        if k not in mm:
            mm[k] = v
            n_hyphen += 1

    def mtoks(g):
        return [mm.get(t, t) for t in _toks(g)]

    traing = set()
    for s in train_raw.values():
        traing.update(mtoks(s["gloss"]))

    vocab = ["<blank>", "<unk>"] + sorted(traing)
    g2i = {g: i for i, g in enumerate(vocab)}
    i2g = {i: g for g, i in g2i.items()}

    def gloss_to_ids(g, is_train):
        out = []
        for t in mtoks(g):
            if t in g2i and t != "<blank>":
                out.append(g2i[t])
            elif not is_train:
                out.append(g2i["<unk>"])
        return out

    cnt = Counter()
    for s in train_raw.values():
        for t in mtoks(s["gloss"]):
            if t in g2i:
                cnt[g2i[t]] += 1

    prior = torch.zeros(len(vocab))
    for idx, c in cnt.items():
        prior[idx] = c
    prior[0] = prior[1:].sum() * 0.01
    prior = prior / prior.sum()
    log_prior = torch.log(prior + 1e-8)

    print(f"  merge map: {len(mm)} entries ({n_hyphen} hyphen-safe) | "
          f"vocab: {len(vocab)} classes (<blank>=0, <unk>=1, gloss_train={len(traing)})")

    return dict(
        vocab=vocab, g2i=g2i, i2g=i2g, num_classes=len(vocab),
        gloss_to_ids=gloss_to_ids, log_prior=log_prior, merge_map=mm,
    )
