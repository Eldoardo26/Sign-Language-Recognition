"""
utils.py — Funzioni di utilità generali condivise tra i moduli.
"""

import gc
import torch
from collections import Counter


# ============================================================
# CUDA
# ============================================================

def cleanup_cuda():
    """Libera memoria GPU e raccoglie garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# CTC DECODING UTILITIES
# ============================================================

def collapse_ctc(token_ids):
    """Rimuove blank (id=0) e duplicati consecutivi da una sequenza CTC."""
    result, prev = [], None
    for idx in token_ids:
        v = idx.item() if hasattr(idx, "item") else int(idx)
        if v != 0 and v != prev:
            result.append(v)
        prev = v
    return result


def ids_to_gloss_text(ids, i2g):
    """Converte una lista di indici in stringa di gloss leggibile."""
    tokens = []
    for idx in ids:
        token_id = int(idx.item() if hasattr(idx, "item") else idx)
        if token_id == 0:
            continue
        tokens.append(i2g.get(token_id, f"<unk:{token_id}>"))
    return " ".join(tokens) if tokens else "<blank>"


def format_counter(counter, i2g, top_k=10):
    """Formatta un Counter di indici come stringa leggibile con gloss label."""
    if not counter:
        return "<vuoto>"
    return ", ".join(
        f"{i2g.get(k, f'<unk:{k}>')}:{v}"
        for k, v in counter.most_common(top_k)
    )
