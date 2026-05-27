"""
metrics.py — Metriche di valutazione per CSLR.

Funzioni:
    compute_wer                      — WER con backtracking S/D/I
    compute_epoch_metrics            — metriche aggregate per epoch
    compute_word_recognition_metrics — precision/recall/F1 per gloss
    print_word_metrics               — stampa report leggibile
    summarize_label_distribution     — istogramma label nel dataset
"""

from collections import Counter, defaultdict


# ============================================================
# WORD ERROR RATE
# ============================================================

def compute_wer(refs: list, hyps: list) -> tuple:
    """
    Calcola WER tramite programmazione dinamica con backtracking.

    Returns:
        (wer, details)
        wer     : float  — (S+D+I) / N
        details : dict   — {'S': int, 'D': int, 'I': int, 'N': int}
    """
    total_s, total_d, total_i, total_n = 0, 0, 0, 0

    for ref, hyp in zip(refs, hyps):
        r, h = len(ref), len(hyp)
        total_n += r

        if r == 0:
            total_i += h
            continue

        # Matrice DP
        dp = [[0] * (h + 1) for _ in range(r + 1)]
        for i in range(1, r + 1):
            dp[i][0] = i
        for j in range(1, h + 1):
            dp[0][j] = j

        for i in range(1, r + 1):
            for j in range(1, h + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,   # sostituzione
                        dp[i - 1][j]     + 1,   # cancellazione
                        dp[i][j - 1]     + 1,   # inserzione
                    )

        # Backtracking
        i, j = r, h
        s, d, ins = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                s += 1; i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                d += 1; i -= 1
            else:
                ins += 1; j -= 1

        total_s += s
        total_d += d
        total_i += ins

    N   = max(total_n, 1)
    wer = (total_s + total_d + total_i) / N
    return wer, {"S": total_s, "D": total_d, "I": total_i, "N": N}


# ============================================================
# EPOCH METRICS
# ============================================================

def compute_epoch_metrics(
    all_refs: list,
    all_hyps: list,
    all_pred_counter: Counter,
    all_ref_counter: Counter,
    i2g: dict,
    top_k: int = 8,
) -> dict:
    """
    Metriche aggregate da visualizzare a fine epoch.

    Returns:
        dict con correct_classes, total_classes, class_accuracy,
             ref_dist, pred_dist, total_pred
    """
    total_ref_tokens  = sum(all_ref_counter.values())
    total_pred_tokens = sum(all_pred_counter.values())

    ref_dist  = {
        i2g.get(k, k): round(v / max(total_ref_tokens,  1) * 100, 2)
        for k, v in all_ref_counter.most_common(top_k)
    }
    pred_dist = {
        i2g.get(k, k): round(v / max(total_pred_tokens, 1) * 100, 2)
        for k, v in all_pred_counter.most_common(top_k)
    }

    ref_set  = set(all_ref_counter.keys())
    pred_set = set(all_pred_counter.keys())
    correct  = len(ref_set & pred_set)

    return {
        "correct_classes": correct,
        "total_classes":   len(ref_set),
        "class_accuracy":  correct / max(len(ref_set), 1) * 100,
        "ref_dist":        ref_dist,
        "pred_dist":       pred_dist,
        "total_pred":      total_pred_tokens,
    }


# ============================================================
# PER-WORD RECOGNITION METRICS
# ============================================================

def compute_word_recognition_metrics(
    all_refs: list,
    all_hyps: list,
    i2g: dict,
    min_support: int = 5,
    include_unk: bool = False,
) -> tuple:
    """
    Precision, recall e F1 per ogni gloss, basate su multiset-overlap.

    Args:
        all_refs     : lista di sequenze di indici (riferimento)
        all_hyps     : lista di sequenze di indici (ipotesi)
        i2g          : {indice: gloss}
        min_support  : soglia minima per classi "frequenti"
        include_unk  : se True include <unk> (id=1) come classe valida

    Returns:
        summary, per_class, never_predicted, low_recall
    """
    ref_count = defaultdict(int)
    hyp_count = defaultdict(int)
    tp_count  = defaultdict(int)

    for ref, hyp in zip(all_refs, all_hyps):
        ref_bag = defaultdict(int)
        hyp_bag = defaultdict(int)
        for r in ref: ref_bag[r] += 1
        for h in hyp: hyp_bag[h] += 1

        for k in set(ref_bag) | set(hyp_bag):
            ref_count[k] += ref_bag[k]
            hyp_count[k] += hyp_bag[k]
            tp_count[k]  += min(ref_bag[k], hyp_bag[k])

    # Escludi blank (0); <unk> (1) opzionale
    min_id    = 1 if include_unk else 2
    gloss_ids = [k for k in ref_count if k >= min_id]

    per_class = []
    for k in gloss_ids:
        tp        = tp_count[k]
        precision = tp / hyp_count[k]  if hyp_count[k] > 0 else 0.0
        recall    = tp / ref_count[k]  if ref_count[k]  > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) \
                    if (precision + recall) > 0 else 0.0
        per_class.append({
            "id":        k,
            "gloss":     i2g.get(k, f"[{k}]"),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   ref_count[k],
            "tp":        tp,
            "predicted": hyp_count[k],
        })

    per_class.sort(key=lambda x: -x["support"])

    total_support = sum(d["support"] for d in per_class)
    if total_support > 0:
        macro_f1        = sum(d["f1"]        * d["support"] for d in per_class) / total_support
        macro_precision = sum(d["precision"] * d["support"] for d in per_class) / total_support
        macro_recall    = sum(d["recall"]    * d["support"] for d in per_class) / total_support
    else:
        macro_f1 = macro_precision = macro_recall = 0.0

    never_predicted = [d for d in per_class if d["predicted"] == 0 and d["support"] >= min_support]
    low_recall      = [d for d in per_class if 0 < d["recall"] < 0.20  and d["support"] >= min_support]

    summary = {
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall":    round(macro_recall, 4),
        "n_classes":       len(per_class),
        "n_never_pred":    len(never_predicted),
        "n_low_recall":    len(low_recall),
        "include_unk":     include_unk,
    }
    return summary, per_class, never_predicted, low_recall


def print_word_metrics(
    summary, per_class, never_predicted, low_recall,
    summary_unk=None, per_class_unk=None,
    never_predicted_unk=None, low_recall_unk=None,
    top_k: int = 15,
    epoch=None,
    split: str = "VAL",
):
    """Stampa un report leggibile delle metriche per singola parola."""
    ep_str = f" Ep {epoch}" if epoch is not None else ""

    print(f"\n{'─'*65}")
    print(f"[WORD METRICS {split}{ep_str}]  (escluso <unk>)")
    print(f"  Macro-F1:        {summary['macro_f1']*100:.2f}%")
    print(f"  Macro-Precision: {summary['macro_precision']*100:.2f}%")
    print(f"  Macro-Recall:    {summary['macro_recall']*100:.2f}%")
    print(f"  Classi totali:   {summary['n_classes']}")
    print(f"  Mai predette:    {summary['n_never_pred']}  (support ≥ 5)")
    print(f"  Low recall<20%:  {summary['n_low_recall']}  (support ≥ 5)")

    print(f"\n  Top-{top_k} classi per support:")
    print(f"  {'Gloss':<22} {'Sup':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*50}")
    for d in per_class[:top_k]:
        print(f"  {d['gloss']:<22} {d['support']:>5} "
              f"{d['precision']*100:>5.1f}% {d['recall']*100:>5.1f}% {d['f1']*100:>5.1f}%")

    if never_predicted:
        print(f"\n  ⚠ Classi frequenti MAI predette (top-10):")
        for d in sorted(never_predicted, key=lambda x: -x["support"])[:10]:
            print(f"    {d['gloss']:<22} support={d['support']}")

    if summary_unk is not None:
        print(f"\n{'─'*65}")
        print(f"[WORD METRICS {split}{ep_str}]  (incluso <unk> come classe)")
        print(f"  Macro-F1:        {summary_unk['macro_f1']*100:.2f}%")
        print(f"  Macro-Recall:    {summary_unk['macro_recall']*100:.2f}%")
        delta_f1  = (summary_unk["macro_f1"]    - summary["macro_f1"])    * 100
        delta_rec = (summary_unk["macro_recall"] - summary["macro_recall"]) * 100
        print(f"  Δ Macro-F1 vs no-unk: {'+' if delta_f1 >= 0 else ''}{delta_f1:.2f}pp")
        print(f"  Δ Macro-Recall:       {'+' if delta_rec >= 0 else ''}{delta_rec:.2f}pp")
        if per_class_unk:
            for d in per_class_unk:
                if d["gloss"] == "<unk>":
                    print(f"\n  Riga <unk>: support={d['support']} | "
                          f"Prec={d['precision']*100:.1f}% | "
                          f"Rec={d['recall']*100:.1f}% | "
                          f"F1={d['f1']*100:.1f}%  (pred={d['predicted']}, tp={d['tp']})")

    print(f"{'─'*65}")


# ============================================================
# DATASET DIAGNOSTICS
# ============================================================

def summarize_label_distribution(dataset, i2g: dict, top_k: int = 20) -> Counter:
    """Stampa la distribuzione delle label nel dataset e restituisce il Counter."""
    counter = Counter()
    for sample in dataset.samples:
        counter.update(sample["labels"])
    print(f"Classi osservate nel dataset: {len(counter)}")
    print(f"Top-{top_k} label più frequenti:")
    for idx, count in counter.most_common(top_k):
        print(f"  {i2g.get(idx, idx):<20} {count}")
    return counter
