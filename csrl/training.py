"""
training.py — Loop di training, ottimizzatori, scheduler e checkpoint per CSLR.

Funzioni:
    freeze_backbone / unfreeze_backbone — two-phase finetuning (Deep Sign §6.2)
    make_optimizer_phase1 / phase2      — AdamW con LR differenziale
    make_scheduler                      — warmup lineare + cosine annealing
    CheckpointManager                   — salva solo top-K checkpoint
    run_epoch                           — training/validation epoch completa
    print_epoch_summary                 — stampa riassunto epoch
    preview_one_batch                   — debug pre-training
"""

import os
import gc
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

from config import CONFIG
from utils import ids_to_gloss_text, format_counter
from decoding import greedy_decode_with_prior
from metrics import compute_wer, compute_epoch_metrics, compute_word_recognition_metrics, print_word_metrics


# ============================================================
# TWO-PHASE FINETUNING
# ============================================================

def freeze_backbone(model: nn.Module):
    """Fase 1: congela TCN e TemporalAttention, lascia libera la testa."""
    for module in [model.tcn_first, model.tcn_second, model.temporal_attn]:
        for p in module.parameters():
            p.requires_grad = False
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone congelato — parametri trainable: {n:,}")


def unfreeze_backbone(model: nn.Module):
    """Fase 2: sblocca tutti i parametri."""
    for p in model.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone sbloccato — parametri trainable: {n:,}")


# ============================================================
# OTTIMIZZATORI
# ============================================================

def make_optimizer_phase1(model: nn.Module, lr: float, weight_decay: float):
    """Fase 1: solo parametri con requires_grad=True (BiLSTM + testa)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_optimizer_phase2(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
):
    """
    Fase 2: LR differenziale — backbone lento, testa veloce.
    Compatibile con PoseNetworkCTC (tcn_first/second, temporal_attn, bilstm, fc, aux_proj).
    """
    return torch.optim.AdamW([
        {"params": model.tcn_first.parameters(),     "lr": lr_backbone},
        {"params": model.tcn_second.parameters(),    "lr": lr_backbone},
        {"params": model.temporal_attn.parameters(), "lr": lr_backbone},
        {"params": model.bilstm.parameters(),        "lr": lr_head},
        {"params": model.norm.parameters(),          "lr": lr_head},
        {"params": model.fc.parameters(),            "lr": lr_head},
        {"params": model.aux_proj.parameters(),      "lr": lr_head},
    ], weight_decay=weight_decay)


# ============================================================
# SCHEDULER
# ============================================================

def make_scheduler(optimizer, num_epochs: int, steps_per_epoch: int):
    """Warmup lineare (5 epoch) + Cosine Annealing."""
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_steps = 5 * steps_per_epoch
    total_steps  = num_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """Mantiene solo i top-K checkpoint per metrica, cancella i peggiori."""

    def __init__(self, save_dir: str, keep_top_k: int = 3, metric: str = "wer"):
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k  = keep_top_k
        self.metric      = metric
        self.checkpoints = []   # list of (metric_value, path, epoch)

    def save(self, model: nn.Module, epoch: int, metrics: dict) -> Path:
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

    def get_best_path(self) -> Path:
        return self.checkpoints[0][1] if self.checkpoints else None


# ============================================================
# TRAINING EPOCH
# ============================================================

def run_epoch(
    model: nn.Module,
    dl,
    criterion: nn.Module,
    optimizer,
    scaler,
    scheduler,
    training: bool,
    device,
    config: dict,
    log_prior: torch.Tensor,
    i2g: dict,
):
    """
    Esegue una singola epoch di training o validazione.

    Gestisce:
    - SR-CTC (tuple output in training mode): loss = 0.7 * main + 0.3 * aux
    - Gradient accumulation corretta
    - Mixed precision (AMP)
    - Greedy decode con/senza prior scaling

    Returns:
        (avg_loss, avg_wer, metrics,
         word_summary, per_class, never_pred, low_rec,
         word_summary_unk, per_class_unk, never_pred_unk, low_rec_unk)
    """
    model.train() if training else model.eval()

    total_loss       = 0.0
    all_refs, all_hyps = [], []
    all_pred_counter = Counter()
    all_ref_counter  = Counter()

    accum_steps = config["gradient_accumulation_steps"]
    beta        = config["prior_beta"]
    grad_clip   = config.get("grad_clip", 5.0)
    use_amp     = config["use_amp"] and device.type == "cuda"

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (tssies, targets, input_lengths, target_lengths) in enumerate(
            tqdm(dl, leave=False, desc="Train" if training else "Val")):

        tssies = tssies.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,
                                 dtype=torch.float16, enabled=use_amp):

                output = model(tssies)
                if isinstance(output, tuple):
                    main_lp, aux_lp = output
                    has_aux = True
                else:
                    main_lp = output
                    has_aux = False

                lp_float = main_lp.permute(1, 0, 2).float()   # (T, B, C)
                T_out    = lp_float.shape[0]
                T_in     = tssies.shape[3]
                scale    = T_out / max(T_in, 1)

                ctc_input_lengths  = (input_lengths.float() * scale).long().clamp(1, T_out).cpu()
                ctc_target_lengths = target_lengths.to(dtype=torch.long, device="cpu")
                ctc_targets        = targets.to(dtype=torch.long, device="cpu")

                loss_main = criterion(lp_float, ctc_targets,
                                      ctc_input_lengths, ctc_target_lengths)

                if training and has_aux:
                    aux_float = aux_lp.permute(1, 0, 2).float()
                    loss_aux  = criterion(aux_float, ctc_targets,
                                          ctc_input_lengths, ctc_target_lengths)
                    loss = 0.7 * loss_main + 0.3 * loss_aux
                else:
                    loss = loss_main

            if training:
                scaler.scale(loss / accum_steps).backward()
                if (batch_idx + 1) % accum_steps == 0 or \
                   (batch_idx + 1) == len(dl):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        total_loss += loss.item()

        # Decode
        with torch.no_grad():
            lp_cpu = lp_float.detach().cpu()
            if training:
                decoded = greedy_decode_with_prior(lp_cpu, beta=0.0)
            else:
                decoded = greedy_decode_with_prior(lp_cpu, beta=beta,
                                                   log_prior=log_prior.cpu())

        # Accumula refs
        refs, offset = [], 0
        for tlen in ctc_target_lengths.tolist():
            refs.append(ctc_targets[offset:offset + tlen].tolist())
            offset += tlen

        all_refs.extend(refs)
        all_hyps.extend(decoded)
        for ref in refs:
            all_ref_counter.update(ref)
        for hyp in decoded:
            all_pred_counter.update(hyp)

        del tssies, lp_float, lp_cpu, loss

    # Metriche finali
    avg_loss             = total_loss / len(dl)
    avg_wer, wer_details = compute_wer(all_refs, all_hyps)

    metrics              = compute_epoch_metrics(all_refs, all_hyps,
                                                 all_pred_counter, all_ref_counter,
                                                 i2g, top_k=8)
    metrics["wer_details"] = wer_details

    word_summary, per_class, never_pred, low_rec = compute_word_recognition_metrics(
        all_refs, all_hyps, i2g, include_unk=False)
    word_summary_unk, per_class_unk, never_pred_unk, low_rec_unk = \
        compute_word_recognition_metrics(all_refs, all_hyps, i2g, include_unk=True)

    return (avg_loss, avg_wer, metrics,
            word_summary, per_class, never_pred, low_rec,
            word_summary_unk, per_class_unk, never_pred_unk, low_rec_unk)


# ============================================================
# STAMPA EPOCH
# ============================================================

def print_epoch_summary(epoch: int, metrics: dict, loss: float,
                        wer: float, training: bool = True):
    """Stampa un riassunto conciso dell'epoch."""
    split   = "TRAIN" if training else "VAL"
    details = metrics.get("wer_details", {"S": 0, "D": 0, "I": 0, "N": 1})
    N       = details["N"]
    s_pct   = details["S"] / N * 100
    d_pct   = details["D"] / N * 100
    i_pct   = details["I"] / N * 100
    print(
        f"[{split}] Ep {epoch:3d} | Loss {loss:.4f} | WER {wer*100:.2f}%\n"
        f"    ↳ Breakdown: Sost={s_pct:.1f}% | Canc={d_pct:.1f}% | Ins={i_pct:.1f}%\n"
        f"    ↳ Classi corrette: {metrics['correct_classes']}/{metrics['total_classes']} "
        f"({metrics['class_accuracy']:.1f}%) | Pred token: {metrics['total_pred']}"
    )


# ============================================================
# DEBUG PRE-TRAINING
# ============================================================

def preview_one_batch(model, dataloader, split_name: str,
                      device, config: dict, log_prior, i2g: dict):
    """Decodifica un singolo batch e stampa confronto REF/HYP."""
    model.eval()
    with torch.no_grad():
        for tssies, targets, input_lengths, target_lengths in dataloader:
            tssies = tssies.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=config["use_amp"] and device.type == "cuda"):
                log_probs = model(tssies)
                lp_float  = log_probs.permute(1, 0, 2).float()

            beta    = 0.0 if split_name == "train" else config["prior_beta"]
            decoded = greedy_decode_with_prior(lp_float.cpu(), beta=beta,
                                               log_prior=log_prior.cpu())

            refs, offset = [], 0
            for tlen in target_lengths.tolist():
                refs.append(targets[offset:offset + tlen].tolist())
                offset += tlen

            frame_counter = Counter(lp_float.argmax(dim=-1).reshape(-1).tolist())
            pred_counter  = Counter()
            ref_counter   = Counter()
            for seq in decoded:  pred_counter.update(seq)
            for seq in refs:     ref_counter.update(seq)

            print(f"\n=== Preview {split_name.upper()} batch ===")
            print(f"Logits shape: {tuple(lp_float.shape)} | batch size: {len(refs)}")
            print(f"Frame argmax top-10: {format_counter(frame_counter, i2g, 10)}")
            print(f"Ref top-10:          {format_counter(ref_counter,   i2g, 10)}")
            print(f"Pred top-10:         {format_counter(pred_counter,  i2g, 10)}")

            for sample_idx in range(min(config["debug_preview_samples"], len(refs))):
                print(f"  sample {sample_idx+1:02d}")
                print(f"    REF: {ids_to_gloss_text(refs[sample_idx], i2g)}")
                print(f"    HYP: {ids_to_gloss_text(decoded[sample_idx], i2g)}")
            break
