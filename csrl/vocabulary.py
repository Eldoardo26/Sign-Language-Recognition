"""
vocabulary.py — Caricamento CSV PHOENIX, normalizzazione gloss,
                costruzione vocabolario, merge map e log-prior.

Funzioni principali:
    load_split          — carica CSV per split (train/dev/test)
    normalize_gloss_token — normalizza un singolo token gloss
    build_merge_map     — costruisce mappa di merge per gloss simili
    apply_merge_to_df   — applica merge map a un DataFrame
    collect_gloss_set   — raccoglie l'insieme di gloss da un DataFrame
    apply_unknown_to_df — sostituisce gloss OOV con '<unk>'
    build_vocabulary    — pipeline completa → vocab, g2i, i2g, LOG_PRIOR
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from difflib import SequenceMatcher

from config import ANNOTATIONS_DIR, CONFIG, RESULTS_DIR


# ============================================================
# CARICAMENTO CSV
# ============================================================

def load_split(split: str) -> pd.DataFrame:
    """Carica il CSV PHOENIX per lo split indicato (train / dev / test)."""
    path = os.path.join(ANNOTATIONS_DIR, f"PHOENIX-2014-T.{split}.corpus.csv")
    return pd.read_csv(path, sep="|")


# ============================================================
# NORMALIZZAZIONE TOKEN
# ============================================================

def normalize_gloss_token(token: str) -> str:
    """
    Normalizza un token gloss grezzo:
    - uppercase
    - unifica trattini (–, —, _) → -
    - rimuove caratteri non alfanumerici da inizio/fine
    - rimuove spazi interni
    """
    token = str(token).strip().upper()
    token = token.replace("–", "-").replace("—", "-").replace("_", "-")
    token = re.sub(r"^[^0-9A-ZÀ-ÖØ-öø-ÿÄÖÜß]+|[^0-9A-ZÀ-ÖØ-öø-ÿÄÖÜß]+$", "", token)
    token = re.sub(r"\s+", "", token)
    return token


# ============================================================
# MERGE MAP
# ============================================================

def build_merge_map(glosses, similarity_threshold: float = 0.85) -> dict:
    """
    Crea una mappa {gloss_raro → gloss_comune} per gloss ortograficamente
    simili (SequenceMatcher ratio >= similarity_threshold).
    Merge solo verso il termine che viene prima in ordine alfabetico.
    """
    glosses_sorted = sorted(glosses)
    merge_map = {}
    for i, g1 in enumerate(glosses_sorted):
        if g1 in merge_map:
            continue
        for g2 in glosses_sorted[i + 1:]:
            if g2 in merge_map:
                continue
            if SequenceMatcher(None, g1, g2).ratio() >= similarity_threshold:
                merge_map[g2] = g1
    return merge_map


def apply_merge_to_df(df: pd.DataFrame, merge_map: dict) -> pd.DataFrame:
    """Applica il merge_map alla colonna 'orth' di un DataFrame."""
    df = df.copy()
    new_orth = []
    for gloss_seq in df["orth"].fillna(""):
        tokens = str(gloss_seq).strip().upper().split()
        merged = [merge_map.get(t, t) for t in tokens]
        new_orth.append(" ".join(merged))
    df["orth"] = new_orth
    return df


def apply_safe_hyphen_merges(glosses: set) -> dict:
    """
    Merge automatico conservativo: TOKEN-CON-TRATTINO → TOKENCONTRATTINO
    solo se la forma compatta esiste già nel vocabolario.
    """
    safe_map = {}
    for token in sorted(glosses):
        compact = token.replace("-", "")
        if "-" in token and compact in glosses and compact != token:
            safe_map[token] = compact
    return safe_map


# ============================================================
# GLOSS SET / OOV MAPPING
# ============================================================

def collect_gloss_set(df: pd.DataFrame) -> set:
    """Raccoglie l'insieme di tutti i token gloss (uppercase) da un DataFrame."""
    gloss_set = set()
    for gloss_seq in df["orth"].fillna(""):
        gloss_set.update(str(gloss_seq).strip().upper().split())
    return gloss_set


def apply_unknown_to_df(df: pd.DataFrame, unknown_map: dict) -> pd.DataFrame:
    """Sostituisce i token presenti in unknown_map con '<unk>'."""
    df = df.copy()
    new_orth = []
    for gloss_seq in df["orth"].fillna(""):
        tokens  = str(gloss_seq).strip().upper().split()
        mapped  = [unknown_map.get(t, t) for t in tokens]
        new_orth.append(" ".join(mapped))
    df["orth"] = new_orth
    return df


# ============================================================
# PIPELINE COMPLETA
# ============================================================

def build_vocabulary(train_df, dev_df, test_df):
    """
    Pipeline completa:
      1. Merge gloss simili
      2. Mapping OOV → '<unk>'
      3. Costruzione vocabolario ['<blank>', '<unk>', ...gloss_train]
      4. Calcolo log-prior dal training set

    Returns:
        train_df_m, dev_df_m, test_df_m  — DataFrame con merge applicato
        vocab, g2i, i2g, num_classes
        LOG_PRIOR  — torch.Tensor shape (num_classes,)
        merge_map  — dict usato per merge
    """
    # -- Fase 1: merge gloss simili --
    all_glosses_raw = set()
    for df in [train_df, dev_df, test_df]:
        for gloss_seq in df["orth"].fillna(""):
            all_glosses_raw.update(str(gloss_seq).strip().upper().split())

    merge_map = build_merge_map(all_glosses_raw, similarity_threshold=0.85)

    # Carica merge map manuale se disponibile, poi estende con quella auto
    if CONFIG["use_gloss_merge"] and os.path.exists(CONFIG["merge_map_path"]):
        with open(CONFIG["merge_map_path"], "r", encoding="utf-8") as f:
            manual_map = json.load(f)
        merge_map.update(manual_map)
        print(f"✓ Merge map manuale caricata: {len(manual_map)} entry")

    # Aggiungi merge trattino-compatto conservativi
    safe_hyphen = apply_safe_hyphen_merges(all_glosses_raw)
    for k, v in safe_hyphen.items():
        if k not in merge_map:
            merge_map[k] = v
    print(f"✓ Merge map totale: {len(merge_map)} entry "
          f"({len(safe_hyphen)} hyphen-safe)")

    train_df_m = apply_merge_to_df(train_df, merge_map)
    dev_df_m   = apply_merge_to_df(dev_df,   merge_map)
    test_df_m  = apply_merge_to_df(test_df,  merge_map)

    # -- Fase 2: mapping OOV → '<unk>' --
    train_glosses = collect_gloss_set(train_df_m)
    dev_glosses   = collect_gloss_set(dev_df_m)
    test_glosses  = collect_gloss_set(test_df_m)

    unknown_map   = {g: "<unk>" for g in (dev_glosses | test_glosses) - train_glosses}
    dev_df_m      = apply_unknown_to_df(dev_df_m,  unknown_map)
    test_df_m     = apply_unknown_to_df(test_df_m, unknown_map)
    print(f"✓ OOV mapping: {len(unknown_map)} gloss → '<unk>'")

    # -- Fase 3: vocabolario --
    vocab       = ["<blank>", "<unk>"] + sorted(train_glosses)
    g2i         = {g: i for i, g in enumerate(vocab)}
    i2g         = {i: g for g, i in g2i.items()}
    num_classes = len(vocab)
    print(f"✓ Vocabolario: {num_classes} classi "
          f"(<blank>=0, <unk>=1, gloss_train={len(train_glosses)})")

    # -- Fase 4: log-prior --
    token_counts = Counter()
    for seq in train_df_m["orth"].fillna(""):
        for tok in str(seq).strip().upper().split():
            if tok in g2i:
                token_counts[g2i[tok]] += 1

    prior = torch.zeros(num_classes)
    for idx, cnt in token_counts.items():
        prior[idx] = cnt
    prior[0]  = prior[1:].sum() * 0.01      # blank basso ma > 0
    prior     = prior / prior.sum()
    LOG_PRIOR = torch.log(prior + 1e-8)
    print(f"✓ Log-prior calcolato (β={CONFIG['prior_beta']})")

    # Salva merge map se non esiste ancora
    if CONFIG["debug_save_reports"] and not os.path.exists(CONFIG["merge_map_path"]):
        os.makedirs(os.path.dirname(CONFIG["merge_map_path"]), exist_ok=True)
        with open(CONFIG["merge_map_path"], "w", encoding="utf-8") as f:
            json.dump(merge_map, f, indent=2, ensure_ascii=False)
        print(f"✓ Merge map salvata in: {CONFIG['merge_map_path']}")

    return train_df_m, dev_df_m, test_df_m, vocab, g2i, i2g, num_classes, LOG_PRIOR, merge_map
