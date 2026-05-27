from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set all relevant random seeds.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dfs_order_phoenix_correct() -> np.ndarray:
    """Construct DFS order for MediaPipe skeleton (48 keypoints).

    Returns:
        DFS order as a numpy array of shape (135,).
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    edges_body = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
    edges_wrist_to_hand = [(3, 6), (5, 27)]
    finger_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    edges_lh = [(s + 6, e + 6) for s, e in finger_edges]
    edges_rh = [(s + 27, e + 27) for s, e in finger_edges]

    for u, v in edges_body + edges_wrist_to_hand + edges_lh + edges_rh:
        if u < 48 and v < 48:
            adj[u].append(v)
            adj[v].append(u)

    dfs_path: List[int] = []
    visited: set[int] = set()

    def dfs(node: int) -> None:
        dfs_path.append(node)
        visited.add(node)
        for n in sorted([x for x in adj[node] if x not in visited]):
            dfs(n)
            if len(dfs_path) < 135:
                dfs_path.append(node)

    dfs(0)
    while len(dfs_path) < 135:
        dfs_path.extend(dfs_path[-10:])

    return np.array(dfs_path[:135], dtype=np.int32)


class CTCLossWithEntropy(nn.Module):
    """CTC loss with entropy regularization.

    Args:
        blank: Blank index.
        entropy_weight: Weight for entropy regularization.
    """

    def __init__(self, blank: int = 0, entropy_weight: float = 0.05) -> None:
        super().__init__()
        self.blank = blank
        self.entropy_weight = entropy_weight
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)

    def forward(
        self,
        log_probs_tbc: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss with entropy regularization.

        Args:
            log_probs_tbc: Log probabilities with shape (T, B, C).
            targets: Concatenated targets.
            input_lengths: Input lengths per sample.
            target_lengths: Target lengths per sample.

        Returns:
            Loss value.
        """
        t, b, c = log_probs_tbc.shape
        ctc_loss = self.ctc(log_probs_tbc, targets, input_lengths, target_lengths)

        if self.entropy_weight <= 0:
            return ctc_loss

        probs = torch.exp(log_probs_tbc).clamp(min=1e-10)
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
        max_entropy = math.log(max(c, 1))
        entropy_reg = entropy / max(max_entropy, 1e-8)
        return ctc_loss + self.entropy_weight * entropy_reg


def beam_search_ctc_optimized(
    log_probs_tbc: torch.Tensor,
    beam_width: int = 10,
    beta: float = 0.3,
    log_prior: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Beam search CTC decoding with optional prior scaling.

    Args:
        log_probs_tbc: Log probabilities with shape (T, B, C).
        beam_width: Beam width.
        beta: Prior scaling factor.
        log_prior: Log prior of classes.

    Returns:
        List of decoded sequences per batch item.
    """
    from scipy.special import logsumexp

    t, b, c = log_probs_tbc.shape
    results: List[List[int]] = []

    for bi in range(b):
        lp = log_probs_tbc[:, bi, :].cpu().numpy()
        if beta > 0.0 and log_prior is not None:
            lp_prior = log_prior.cpu().numpy()
            lp = lp - beta * lp_prior.reshape(1, -1)
            lp = lp - logsumexp(lp, axis=1, keepdims=True)

        beams: Dict[Tuple[Tuple[int, ...], Optional[int]], float] = {((), None): 0.0}
        for ti in range(t):
            new_beams: Dict[Tuple[Tuple[int, ...], Optional[int]], float] = {}
            for (prefix, last_char), beam_lp in beams.items():
                for ci in range(c):
                    new_lp = beam_lp + lp[ti, ci]
                    if ci == 0:
                        key = (prefix, last_char)
                    elif ci == last_char:
                        continue
                    else:
                        key = (prefix + (ci,), ci)

                    if key not in new_beams or new_beams[key] < new_lp:
                        new_beams[key] = new_lp

            sorted_beams = sorted(new_beams.items(), key=lambda x: -x[1])
            beams = dict(sorted_beams[:beam_width])

        best_key = max(beams.keys(), key=lambda k: beams[k])
        results.append(list(best_key[0]))

    return results


def create_scheduler_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create linear warmup + cosine schedule.

    Args:
        optimizer: Optimizer instance.
        total_steps: Total steps.
        warmup_steps: Number of warmup steps.
        min_lr_ratio: Minimum learning rate ratio.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CheckpointManager:
    """Checkpoint manager keeping only top-k checkpoints."""

    def __init__(self, save_dir: str, keep_top_k: int = 3, metric: str = "wer") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.checkpoints: List[Tuple[float, Path, int]] = []

    def save(self, model: nn.Module, epoch: int, metrics: Dict[str, float]) -> Path:
        """Save a checkpoint and prune by metric.

        Args:
            model: Model instance.
            epoch: Epoch number.
            metrics: Metric dictionary.

        Returns:
            Path of saved checkpoint.
        """
        metric_value = metrics[self.metric]
        path = self.save_dir / f"epoch_{epoch:03d}_{self.metric}_{metric_value:.4f}.pth"
        torch.save(model.state_dict(), path)

        self.checkpoints.append((metric_value, path, epoch))
        self.checkpoints.sort(key=lambda x: x[0])

        if len(self.checkpoints) > self.keep_top_k:
            _, worst_path, _ = self.checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()

        return path

    def get_best_path(self) -> Optional[Path]:
        """Get best checkpoint path.

        Returns:
            Best checkpoint path or None.
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]


def collapse_ctc(token_ids: Sequence[int]) -> List[int]:
    """Remove blank (0) and consecutive duplicates.

    Args:
        token_ids: Token sequence.

    Returns:
        Collapsed sequence.
    """
    result: List[int] = []
    prev: Optional[int] = None
    for idx in token_ids:
        v = int(idx)
        if v != 0 and v != prev:
            result.append(v)
        prev = v
    return result


def greedy_decode_with_prior(
    log_probs_tbc: torch.Tensor,
    beta: float = 0.0,
    log_prior: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Greedy CTC decoding with optional prior scaling.

    Args:
        log_probs_tbc: Log probabilities with shape (T, B, C).
        beta: Prior scaling factor.
        log_prior: Log prior over classes.

    Returns:
        List of decoded sequences.
    """
    t, b, _ = log_probs_tbc.shape
    results: List[List[int]] = []
    for bi in range(b):
        lp = log_probs_tbc[:, bi, :].clone()
        if beta > 0.0 and log_prior is not None:
            lp = lp - beta * log_prior.unsqueeze(0)
        best_ids = lp.argmax(dim=-1).tolist()
        results.append(collapse_ctc(best_ids))
    return results


def compute_wer(refs: List[List[int]], hyps: List[List[int]]) -> Tuple[float, Dict[str, int]]:
    """Compute word error rate.

    Args:
        refs: Reference sequences.
        hyps: Hypothesis sequences.

    Returns:
        Tuple with WER and details.
    """
    total_s, total_d, total_i, total_n = 0, 0, 0, 0
    for ref, hyp in zip(refs, hyps):
        r, h = len(ref), len(hyp)
        total_n += r
        if r == 0:
            total_i += h
            continue

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
                    sub = dp[i - 1][j - 1] + 1
                    dlt = dp[i - 1][j] + 1
                    ins = dp[i][j - 1] + 1
                    dp[i][j] = min(sub, dlt, ins)

        i, j = r, h
        s, d, ins = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                s += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                d += 1
                i -= 1
            else:
                ins += 1
                j -= 1

        total_s += s
        total_d += d
        total_i += ins

    n = max(total_n, 1)
    wer = (total_s + total_d + total_i) / n
    return wer, {"S": total_s, "D": total_d, "I": total_i, "N": n}


def compute_word_recognition_metrics(
    all_refs: List[List[int]],
    all_hyps: List[List[int]],
    i2g: Dict[int, str],
    min_support: int = 5,
    include_unk: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:
    """Compute per-class precision/recall/F1.

    Args:
        all_refs: Reference sequences.
        all_hyps: Hypothesis sequences.
        i2g: Mapping from id to gloss.
        min_support: Minimum support to report issues.
        include_unk: Include <unk> in metrics.

    Returns:
        Summary, per-class metrics, never predicted list, low recall list.
    """
    ref_count: Dict[int, int] = defaultdict(int)
    hyp_count: Dict[int, int] = defaultdict(int)
    tp_count: Dict[int, int] = defaultdict(int)

    for ref, hyp in zip(all_refs, all_hyps):
        ref_bag: Dict[int, int] = defaultdict(int)
        hyp_bag: Dict[int, int] = defaultdict(int)
        for r in ref:
            ref_bag[r] += 1
        for h in hyp:
            hyp_bag[h] += 1

        all_keys = set(ref_bag) | set(hyp_bag)
        for k in all_keys:
            ref_count[k] += ref_bag[k]
            hyp_count[k] += hyp_bag[k]
            tp_count[k] += min(ref_bag[k], hyp_bag[k])

    min_id = 1 if include_unk else 2
    gloss_ids = [k for k in ref_count if k >= min_id]

    per_class: List[Dict[str, float]] = []
    for k in gloss_ids:
        support = ref_count[k]
        tp = tp_count[k]
        precision = tp / hyp_count[k] if hyp_count[k] > 0 else 0.0
        recall = tp / ref_count[k] if ref_count[k] > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append(
            {
                "id": k,
                "gloss": i2g.get(k, f"[{k}]"),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "tp": tp,
                "predicted": hyp_count[k],
            }
        )

    per_class.sort(key=lambda x: -x["support"])

    total_support = sum(d["support"] for d in per_class)
    if total_support > 0:
        macro_f1 = sum(d["f1"] * d["support"] for d in per_class) / total_support
        macro_precision = sum(d["precision"] * d["support"] for d in per_class) / total_support
        macro_recall = sum(d["recall"] * d["support"] for d in per_class) / total_support
    else:
        macro_f1 = macro_precision = macro_recall = 0.0

    never_predicted = [
        d for d in per_class if d["predicted"] == 0 and d["support"] >= min_support
    ]
    low_recall = [
        d for d in per_class if 0 < d["recall"] < 0.20 and d["support"] >= min_support
    ]

    summary = {
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "n_classes": len(per_class),
        "n_never_pred": len(never_predicted),
        "n_low_recall": len(low_recall),
        "include_unk": include_unk,
    }
    return summary, per_class, never_predicted, low_recall


def compute_epoch_metrics(
    all_refs: List[List[int]],
    all_hyps: List[List[int]],
    all_pred_counter: Counter,
    all_ref_counter: Counter,
    i2g: Dict[int, str],
    top_k: int = 8,
) -> Dict[str, object]:
    """Compute aggregated epoch metrics.

    Args:
        all_refs: Reference sequences.
        all_hyps: Hypothesis sequences.
        all_pred_counter: Counter of predictions.
        all_ref_counter: Counter of references.
        i2g: Mapping from id to gloss.
        top_k: Top-k classes to report.

    Returns:
        Dictionary of metrics.
    """
    total_ref_tokens = sum(all_ref_counter.values())
    total_pred_tokens = sum(all_pred_counter.values())

    ref_dist = {
        i2g.get(k, k): round(v / max(total_ref_tokens, 1) * 100, 2)
        for k, v in all_ref_counter.most_common(top_k)
    }
    pred_dist = {
        i2g.get(k, k): round(v / max(total_pred_tokens, 1) * 100, 2)
        for k, v in all_pred_counter.most_common(top_k)
    }

    ref_set = set(all_ref_counter.keys())
    pred_set = set(all_pred_counter.keys())
    correct = len(ref_set & pred_set)

    return {
        "correct_classes": correct,
        "total_classes": len(ref_set),
        "class_accuracy": correct / max(len(ref_set), 1) * 100,
        "ref_dist": ref_dist,
        "pred_dist": pred_dist,
        "total_pred": total_pred_tokens,
    }


def ids_to_gloss_text(ids: Sequence[int], i2g: Dict[int, str]) -> str:
    """Convert ids to gloss text.

    Args:
        ids: Sequence of ids.
        i2g: Mapping from id to gloss.

    Returns:
        Gloss string.
    """
    tokens = [i2g.get(int(i), f"[{i}]") for i in ids if int(i) != 0]
    return " ".join(tokens) if tokens else "<blank>"


def format_counter(counter: Counter, i2g: Dict[int, str], top_k: int = 8) -> str:
    """Format a counter as a compact string.

    Args:
        counter: Counter instance.
        i2g: Mapping from id to gloss.
        top_k: Top-k entries.

    Returns:
        Formatted string.
    """
    if not counter:
        return "<empty>"
    return ", ".join(
        f"{i2g.get(idx, f'<unk:{idx}>')}:{count}"
        for idx, count in counter.most_common(top_k)
    )


def print_word_metrics(
    summary: Dict[str, float],
    per_class: List[Dict[str, float]],
    never_predicted: List[Dict[str, float]],
    low_recall: List[Dict[str, float]],
    summary_unk: Optional[Dict[str, float]] = None,
    per_class_unk: Optional[List[Dict[str, float]]] = None,
    never_predicted_unk: Optional[List[Dict[str, float]]] = None,
    low_recall_unk: Optional[List[Dict[str, float]]] = None,
    top_k: int = 15,
    epoch: Optional[int] = None,
    split: str = "VAL",
) -> None:
    """Log a word metrics report.

    Args:
        summary: Summary metrics.
        per_class: Per-class metrics.
        never_predicted: Never predicted list.
        low_recall: Low recall list.
        summary_unk: Summary with <unk> included.
        per_class_unk: Per-class metrics with <unk>.
        never_predicted_unk: Never predicted list with <unk>.
        low_recall_unk: Low recall list with <unk>.
        top_k: Number of classes to show.
        epoch: Epoch number.
        split: Split name.
    """
    ep_str = f" ep {epoch}" if epoch is not None else ""
    LOGGER.info("%s%s word metrics (exclude <unk>)", split, ep_str)
    LOGGER.info(
        "Macro-F1 %.2f | Macro-Precision %.2f | Macro-Recall %.2f | classes %d",
        summary["macro_f1"] * 100,
        summary["macro_precision"] * 100,
        summary["macro_recall"] * 100,
        summary["n_classes"],
    )

    for row in per_class[:top_k]:
        LOGGER.info(
            "Class %s support=%d prec=%.2f rec=%.2f f1=%.2f",
            row["gloss"],
            row["support"],
            row["precision"] * 100,
            row["recall"] * 100,
            row["f1"] * 100,
        )

    if never_predicted:
        LOGGER.warning("Never predicted (top 10)")
        for row in sorted(never_predicted, key=lambda x: -x["support"])[:10]:
            LOGGER.warning("%s support=%d", row["gloss"], row["support"])

    if summary_unk is None:
        return

    LOGGER.info("%s%s word metrics (include <unk>)", split, ep_str)
    LOGGER.info(
        "Macro-F1 %.2f | Macro-Precision %.2f | Macro-Recall %.2f | classes %d",
        summary_unk["macro_f1"] * 100,
        summary_unk["macro_precision"] * 100,
        summary_unk["macro_recall"] * 100,
        summary_unk["n_classes"],
    )

    if per_class_unk:
        unk_rows = [d for d in per_class_unk if d["gloss"] == "<unk>"]
        if unk_rows:
            row = unk_rows[0]
            LOGGER.info(
                "<unk> support=%d prec=%.2f rec=%.2f f1=%.2f",
                row["support"],
                row["precision"] * 100,
                row["recall"] * 100,
                row["f1"] * 100,
            )

    if never_predicted_unk:
        LOGGER.warning("Never predicted incl <unk> (top 10)")
        for row in sorted(never_predicted_unk, key=lambda x: -x["support"])[:10]:
            LOGGER.warning("%s support=%d", row["gloss"], row["support"])


def summarize_label_distribution(
    labels: Iterable[Iterable[int]],
    i2g: Dict[int, str],
    top_k: int = 20,
) -> Counter:
    """Summarize label distribution.

    Args:
        labels: Iterable of label sequences.
        i2g: Mapping from id to gloss.
        top_k: Top-k classes to log.

    Returns:
        Counter of labels.
    """
    counter: Counter = Counter()
    for seq in labels:
        counter.update(seq)

    LOGGER.info("Observed classes: %d", len(counter))
    for idx, count in counter.most_common(top_k):
        LOGGER.info("%s %d", i2g.get(idx, str(idx)), count)

    return counter


def preview_one_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    split_name: str,
    device: torch.device,
    config: object,
    log_prior: torch.Tensor,
    i2g: Dict[int, str],
) -> None:
    """Preview a single batch with decoded sequences.

    Args:
        model: Model instance.
        dataloader: DataLoader instance.
        split_name: Split name.
        device: Torch device.
        config: Config object with debug options.
        log_prior: Log prior for decoding.
        i2g: Mapping from id to gloss.
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (tssies, targets, input_lengths, target_lengths) in enumerate(dataloader):
            tssies = tssies.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=getattr(config, "use_amp", True) and device.type == "cuda",
            ):
                log_probs = model(tssies)
                lp_float = log_probs.permute(1, 0, 2).float()

            if split_name == "train":
                decoded = greedy_decode_with_prior(lp_float.cpu(), beta=0.0)
            else:
                decoded = greedy_decode_with_prior(
                    lp_float.cpu(), beta=getattr(config, "prior_beta", 0.0), log_prior=log_prior.cpu()
                )

            refs: List[List[int]] = []
            offset = 0
            for tlen in target_lengths.tolist():
                refs.append(targets[offset : offset + tlen].tolist())
                offset += tlen

            frame_argmax = lp_float.argmax(dim=-1)
            frame_counter = Counter(frame_argmax.reshape(-1).tolist())
            pred_counter: Counter = Counter()
            ref_counter: Counter = Counter()
            for seq in decoded:
                pred_counter.update(seq)
            for seq in refs:
                ref_counter.update(seq)

            LOGGER.info("Preview %s batch %d", split_name, batch_idx + 1)
            LOGGER.info("Frame argmax: %s", format_counter(frame_counter, i2g, 10))
            LOGGER.info("Ref top: %s", format_counter(ref_counter, i2g, 10))
            LOGGER.info("Pred top: %s", format_counter(pred_counter, i2g, 10))

            preview_n = min(getattr(config, "debug_preview_samples", 3), len(refs))
            for sample_idx in range(preview_n):
                LOGGER.info("sample %02d", sample_idx + 1)
                LOGGER.info("REF: %s", ids_to_gloss_text(refs[sample_idx], i2g))
                LOGGER.info("HYP: %s", ids_to_gloss_text(decoded[sample_idx], i2g))
            break


def save_json(path: str, payload: Dict[str, object]) -> None:
    """Save a JSON file with UTF-8 encoding.

    Args:
        path: Output path.
        payload: Dictionary to serialize.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def plot_training_history(
    history: Dict[str, List[float]],
    results_dir: str,
    phase1_epochs: int,
    best_wer: float,
    ensemble_test_wer: Optional[float] = None,
) -> None:
    """Plot loss and WER curves.

    Args:
        history: Training history dictionary.
        results_dir: Output directory.
        phase1_epochs: Number of phase1 epochs.
        best_wer: Best dev WER.
        ensemble_test_wer: Ensemble test WER if available.
    """
    import matplotlib.pyplot as plt

    epochs_all = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].axvspan(1, phase1_epochs, alpha=0.08, color="blue", label="Phase 1")
    axes[0].axvspan(phase1_epochs, len(history["train_loss"]), alpha=0.08, color="green", label="Phase 2")
    axes[0].plot(epochs_all, history["train_loss"], label="Train Loss", alpha=0.8)
    axes[0].plot(epochs_all, history["dev_loss"], label="Dev Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvspan(1, phase1_epochs, alpha=0.08, color="blue")
    axes[1].axvspan(phase1_epochs, len(history["dev_wer"]), alpha=0.08, color="green")
    axes[1].plot(epochs_all, [w * 100 for w in history["dev_wer"]], label="Dev WER (%)", color="red", alpha=0.8)
    axes[1].axhline(best_wer * 100, color="green", linestyle="--", label=f"Best dev {best_wer*100:.1f}%")
    if ensemble_test_wer is not None:
        axes[1].axhline(ensemble_test_wer * 100, color="purple", linestyle=":", label=f"Ensemble {ensemble_test_wer*100:.1f}%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("WER (%)")
    axes[1].set_title("Word Error Rate (Validation)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(results_dir) / "training_history.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_bigram_alpha(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    log_prior: torch.Tensor,
    alpha_candidates: Sequence[float],
    greedy_fn,
    compute_wer_fn,
) -> Tuple[float, float]:
    """Evaluate a set of alpha candidates for bigram rescoring.

    Args:
        model: Model instance.
        dataloader: DataLoader for dev split.
        device: Torch device.
        log_prior: Log prior for decoding.
        alpha_candidates: Candidate alpha values.
        greedy_fn: Greedy decoding function.
        compute_wer_fn: Function to compute WER.

    Returns:
        Tuple (best_alpha, best_wer).
    """
    best_alpha, best_wer = 0.0, float("inf")
    model.eval()
    for alpha in alpha_candidates:
        all_refs: List[List[int]] = []
        all_hyps: List[List[int]] = []
        with torch.no_grad():
            for tssies, targets, _, target_lengths in dataloader:
                tssies = tssies.to(device)
                lp = model(tssies).permute(1, 0, 2).float().cpu()
                decoded = greedy_fn(lp, beta=0.0, log_prior=log_prior.cpu(), alpha=alpha)
                all_hyps.extend(decoded)
                offset = 0
                for tlen in target_lengths.tolist():
                    all_refs.append(targets[offset : offset + tlen].tolist())
                    offset += tlen
        wer, _ = compute_wer_fn(all_refs, all_hyps)
        LOGGER.info("alpha=%.2f -> dev WER %.2f", alpha, wer * 100)
        if wer < best_wer:
            best_wer = wer
            best_alpha = alpha

    return best_alpha, best_wer
