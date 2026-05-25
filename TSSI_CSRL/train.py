from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import TrainingConfig, load_config
from data_loader import (
    build_dataloaders,
    build_vocab_and_prior,
    cache_tssi_for_split,
    index_pose_files,
    load_split,
)
from model import build_model
from utils import (
    CTCLossWithEntropy,
    beam_search_ctc_optimized,
    compute_epoch_metrics,
    compute_wer,
    compute_word_recognition_metrics,
    evaluate_bigram_alpha,
    greedy_decode_with_prior,
    ids_to_gloss_text,
    plot_training_history,
    print_word_metrics,
    save_json,
    setup_logging,
    set_seed,
    summarize_label_distribution,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train PHOENIX-2014-T CSLR model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--cache-tssi", action="store_true", help="Precompute TSSI cache")
    parser.add_argument("--run-ensemble", action="store_true", help="Run ensemble evaluation")
    parser.add_argument("--run-bigram", action="store_true", help="Run bigram rescoring")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    return parser.parse_args()


def freeze_backbone(model: nn.Module) -> None:
    """Freeze TCN backbone (phase 1)."""
    for p in model.pose_network.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze full model (phase 2)."""
    for p in model.parameters():
        p.requires_grad = True


def make_optimizer_phase1(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create phase 1 optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_optimizer_phase2(
    model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Create phase 2 optimizer."""
    return torch.optim.AdamW(
        [
            {"params": model.pose_network.parameters(), "lr": lr_backbone},
            {"params": model.bilstm.parameters(), "lr": lr_head},
            {"params": model.norm.parameters(), "lr": lr_head},
            {"params": model.fc.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create warmup + cosine scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = 5 * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    training: bool,
    device: torch.device,
    config: TrainingConfig,
    log_prior: torch.Tensor,
    i2g: Dict[int, str],
) -> Tuple[float, float, Dict[str, object], Dict[str, float], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]], Dict[str, float], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:
    """Run one epoch for training or evaluation.

    Returns:
        Tuple with loss, wer, metrics, and word metrics.
    """
    model.train() if training else model.eval()
    total_loss = 0.0
    all_refs: List[List[int]] = []
    all_hyps: List[List[int]] = []
    all_pred_counter: Dict[int, int] = {}
    all_ref_counter: Dict[int, int] = {}

    accum_steps = config.gradient_accumulation_steps
    beta = config.prior_beta

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (tssies, targets, input_lengths, target_lengths) in enumerate(
        tqdm(dataloader, leave=False, desc="Train" if training else "Val")
    ):
        tssies = tssies.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.use_amp and device.type == "cuda",
            ):
                log_probs = model(tssies)
                lp_float = log_probs.permute(1, 0, 2).float()
                t_out = lp_float.shape[0]
                t_in = tssies.shape[3]
                scale = t_out / max(t_in, 1)

                ctc_input_lengths = (input_lengths.float() * scale).long().clamp(1, t_out).cpu()
                ctc_target_lengths = target_lengths.to(dtype=torch.long, device="cpu")
                ctc_targets = targets.to(dtype=torch.long, device="cpu")

                loss = criterion(lp_float, ctc_targets, ctc_input_lengths, ctc_target_lengths)

            if training and optimizer is not None:
                scaler.scale(loss / accum_steps).backward()
                if (batch_idx + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        with torch.no_grad():
            lp_cpu = lp_float.detach().cpu()
            if training:
                decoded = greedy_decode_with_prior(lp_cpu, beta=0.0)
            else:
                decoded = greedy_decode_with_prior(lp_cpu, beta=beta, log_prior=log_prior.cpu())

        refs: List[List[int]] = []
        offset = 0
        for tlen in ctc_target_lengths.tolist():
            refs.append(ctc_targets[offset : offset + tlen].tolist())
            offset += tlen

        all_refs.extend(refs)
        all_hyps.extend(decoded)

        for ref in refs:
            for token in ref:
                all_ref_counter[token] = all_ref_counter.get(token, 0) + 1
        for hyp in decoded:
            for token in hyp:
                all_pred_counter[token] = all_pred_counter.get(token, 0) + 1

        del tssies, log_probs, lp_float, lp_cpu, loss

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_wer, wer_details = compute_wer(all_refs, all_hyps)
    metrics = compute_epoch_metrics(
        all_refs,
        all_hyps,
        Counter(all_pred_counter),
        Counter(all_ref_counter),
        i2g,
        top_k=config.debug_topk_classes,
    )
    metrics["wer_details"] = wer_details

    word_summary, per_class, never_pred, low_rec = compute_word_recognition_metrics(
        all_refs, all_hyps, i2g, include_unk=False
    )
    word_summary_unk, per_class_unk, never_pred_unk, low_rec_unk = compute_word_recognition_metrics(
        all_refs, all_hyps, i2g, include_unk=True
    )

    return (
        avg_loss,
        avg_wer,
        metrics,
        word_summary,
        per_class,
        never_pred,
        low_rec,
        word_summary_unk,
        per_class_unk,
        never_pred_unk,
        low_rec_unk,
    )


def ensemble_decode(
    model_paths: List[str],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig,
    log_prior: torch.Tensor,
    num_classes: int,
) -> Tuple[float, Dict[str, int]]:
    """Decode with model ensemble.

    Args:
        model_paths: List of checkpoint paths.
        dataloader: DataLoader for evaluation.
        device: Torch device.
        config: TrainingConfig instance.
        log_prior: Log prior tensor.
        num_classes: Number of classes.

    Returns:
        Tuple of WER and details.
    """
    ensemble: List[nn.Module] = []
    for path in model_paths:
        m = build_model(
            num_classes=num_classes,
            num_joints=config.num_joints,
            hidden_dim=config.hidden_dim,
            tcn_blocks=config.tcn_blocks,
            lstm_layers=config.num_layers,
            dropout=0.0,
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        ensemble.append(m)

    all_refs: List[List[int]] = []
    all_hyps: List[List[int]] = []

    with torch.no_grad():
        for tssies, targets, _, target_lengths in tqdm(dataloader, desc="Ensemble"):
            tssies = tssies.to(device)
            avg_lp: Optional[torch.Tensor] = None
            for m in ensemble:
                lp = m(tssies).permute(1, 0, 2).float()
                avg_lp = lp if avg_lp is None else avg_lp + lp
            assert avg_lp is not None
            avg_lp = avg_lp / len(ensemble)

            decoded = beam_search_ctc_optimized(
                avg_lp.cpu(), beam_width=config.beam_width, beta=config.prior_beta, log_prior=log_prior.cpu()
            )

            offset = 0
            for tlen in target_lengths.tolist():
                all_refs.append(targets[offset : offset + tlen].tolist())
                offset += tlen
            all_hyps.extend(decoded)

    return compute_wer(all_refs, all_hyps)


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    config.ensure_dirs()
    set_seed(config.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    train_df = load_split(config.annotations_dir, "train")
    dev_df = load_split(config.annotations_dir, "dev")
    test_df = load_split(config.annotations_dir, "test")

    vocab_artifacts = build_vocab_and_prior(
        train_df,
        dev_df,
        test_df,
        use_gloss_merge=config.use_gloss_merge,
        merge_map_path=config.merge_map_path,
        min_gloss_freq=config.min_gloss_freq,
        debug_save_reports=config.debug_save_reports,
        results_dir=config.results_dir,
    )

    train_kp = index_pose_files(config.pose_dir, "train")
    dev_kp = index_pose_files(config.pose_dir, "dev")
    test_kp = index_pose_files(config.pose_dir, "test")

    if args.cache_tssi:
        cache_tssi_for_split("train", train_kp, config.tssi_output_dir, config.frame_h)
        cache_tssi_for_split("dev", dev_kp, config.tssi_output_dir, config.frame_h)
        cache_tssi_for_split("test", test_kp, config.tssi_output_dir, config.frame_h)

    train_dl, dev_dl, test_dl = build_dataloaders(
        train_df,
        dev_df,
        test_df,
        train_kp,
        dev_kp,
        test_kp,
        vocab_artifacts.g2i,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        frame_h=config.frame_h,
        tssi_output_dir=config.tssi_output_dir,
    )

    model = build_model(
        num_classes=len(vocab_artifacts.vocab),
        num_joints=config.num_joints,
        hidden_dim=config.hidden_dim,
        tcn_blocks=config.tcn_blocks,
        lstm_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = CTCLossWithEntropy(blank=0, entropy_weight=config.ctc_smoothing)
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")

    if config.debug_training:
        summarize_label_distribution((s["labels"] for s in train_dl.dataset.samples), vocab_artifacts.i2g, config.debug_topk_classes)

    best_wer = float("inf")
    patience = 0
    history = {"train_loss": [], "dev_loss": [], "dev_wer": [], "phase": []}
    checkpoint_paths: List[Tuple[str, float]] = []

    LOGGER.info("Phase 1: %d epochs", config.phase1_epochs)
    freeze_backbone(model)
    optimizer_p1 = make_optimizer_phase1(model, lr=config.phase1_lr, weight_decay=config.weight_decay)
    scheduler_p1 = make_scheduler(optimizer_p1, config.phase1_epochs, len(train_dl))

    global_epoch = 0
    for epoch in range(config.phase1_epochs):
        global_epoch += 1
        (train_loss, train_wer, train_metrics, train_wm, train_pc, train_never, train_low,
         train_wm_unk, train_pc_unk, train_never_unk, train_low_unk) = run_epoch(
            model,
            train_dl,
            criterion,
            optimizer_p1,
            scaler,
            scheduler_p1,
            training=True,
            device=device,
            config=config,
            log_prior=vocab_artifacts.log_prior,
            i2g=vocab_artifacts.i2g,
        )
        (dev_loss, dev_wer, dev_metrics, dev_wm, dev_pc, dev_never, dev_low,
         dev_wm_unk, dev_pc_unk, dev_never_unk, dev_low_unk) = run_epoch(
            model,
            dev_dl,
            criterion,
            optimizer_p1,
            scaler,
            scheduler_p1,
            training=False,
            device=device,
            config=config,
            log_prior=vocab_artifacts.log_prior,
            i2g=vocab_artifacts.i2g,
        )

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_wer"].append(dev_wer)
        history["phase"].append(1)

        LOGGER.info("Phase 1 ep %d/%d | loss %.4f/%.4f | WER %.2f/%.2f", epoch + 1, config.phase1_epochs, train_loss, dev_loss, train_wer * 100, dev_wer * 100)
        print_word_metrics(dev_wm, dev_pc, dev_never, dev_low, summary_unk=dev_wm_unk, per_class_unk=dev_pc_unk, never_predicted_unk=dev_never_unk, low_recall_unk=dev_low_unk, epoch=global_epoch, split="VAL")

        if dev_wer < best_wer:
            best_wer = dev_wer
            torch.save(model.state_dict(), Path(config.results_dir) / "best_model.pth")

    phase2_epochs = config.num_epochs - config.phase1_epochs
    LOGGER.info("Phase 2: %d epochs", phase2_epochs)
    unfreeze_backbone(model)
    optimizer_p2 = make_optimizer_phase2(model, config.phase2_lr_backbone, config.phase2_lr_head, config.weight_decay)
    scheduler_p2 = make_scheduler(optimizer_p2, phase2_epochs, len(train_dl))

    for epoch in range(phase2_epochs):
        global_epoch += 1
        (train_loss, train_wer, train_metrics, train_wm, train_pc, train_never, train_low,
         train_wm_unk, train_pc_unk, train_never_unk, train_low_unk) = run_epoch(
            model,
            train_dl,
            criterion,
            optimizer_p2,
            scaler,
            scheduler_p2,
            training=True,
            device=device,
            config=config,
            log_prior=vocab_artifacts.log_prior,
            i2g=vocab_artifacts.i2g,
        )
        (dev_loss, dev_wer, dev_metrics, dev_wm, dev_pc, dev_never, dev_low,
         dev_wm_unk, dev_pc_unk, dev_never_unk, dev_low_unk) = run_epoch(
            model,
            dev_dl,
            criterion,
            optimizer_p2,
            scaler,
            scheduler_p2,
            training=False,
            device=device,
            config=config,
            log_prior=vocab_artifacts.log_prior,
            i2g=vocab_artifacts.i2g,
        )

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_wer"].append(dev_wer)
        history["phase"].append(2)

        LOGGER.info("Phase 2 ep %d/%d | loss %.4f/%.4f | WER %.2f/%.2f", epoch + 1, phase2_epochs, train_loss, dev_loss, train_wer * 100, dev_wer * 100)
        print_word_metrics(dev_wm, dev_pc, dev_never, dev_low, summary_unk=dev_wm_unk, per_class_unk=dev_pc_unk, never_predicted_unk=dev_never_unk, low_recall_unk=dev_low_unk, epoch=global_epoch, split="VAL")

        ckpt_path = Path(config.results_dir) / f"checkpoint_ep{global_epoch:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        checkpoint_paths.append((str(ckpt_path), dev_wer))
        if len(checkpoint_paths) > config.keep_last_n_checkpoints:
            old_path, _ = checkpoint_paths.pop(0)
            if Path(old_path).exists():
                Path(old_path).unlink()

        if dev_wer < best_wer:
            best_wer = dev_wer
            patience = 0
            torch.save(model.state_dict(), Path(config.results_dir) / "best_model.pth")
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %d", global_epoch)
                break

    LOGGER.info("Training complete. Best dev WER %.2f", best_wer * 100)

    best_path = Path(config.results_dir) / "best_model.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    (test_loss, test_wer_single, test_metrics, test_wm, test_pc, test_never, test_low,
     test_wm_unk, test_pc_unk, test_never_unk, test_low_unk) = run_epoch(
        model,
        test_dl,
        criterion,
        None,
        scaler,
        None,
        training=False,
        device=device,
        config=config,
        log_prior=vocab_artifacts.log_prior,
        i2g=vocab_artifacts.i2g,
    )

    print_word_metrics(test_wm, test_pc, test_never, test_low, summary_unk=test_wm_unk, per_class_unk=test_pc_unk, never_predicted_unk=test_never_unk, low_recall_unk=test_low_unk, split="TEST")
    LOGGER.info("Test WER (single): %.2f", test_wer_single * 100)

    ensemble_test_wer: Optional[float] = None
    if args.run_ensemble and checkpoint_paths:
        n_ensemble = min(config.ensemble_n, len(checkpoint_paths))
        best_ckpts = sorted(checkpoint_paths, key=lambda x: x[1])[:n_ensemble]
        ensemble_paths = [p for p, _ in best_ckpts]
        ensemble_wer, _ = ensemble_decode(
            ensemble_paths, dev_dl, device, config, vocab_artifacts.log_prior, len(vocab_artifacts.vocab)
        )
        LOGGER.info("Ensemble WER (dev): %.2f", ensemble_wer * 100)
        ensemble_test_wer, _ = ensemble_decode(
            ensemble_paths, test_dl, device, config, vocab_artifacts.log_prior, len(vocab_artifacts.vocab)
        )
        LOGGER.info("Ensemble WER (test): %.2f", ensemble_test_wer * 100)

    if args.run_bigram:
        from collections import defaultdict

        bigram_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for seq in train_df["orth"].fillna(""):
            tokens = [vocab_artifacts.g2i.get(t) for t in str(seq).strip().upper().split()]
            ids = [t for t in tokens if t is not None]
            for a, b in zip(ids, ids[1:]):
                bigram_counts[a][b] += 1

        log_bigram: Dict[int, Dict[int, float]] = {}
        for prev, nexts in bigram_counts.items():
            total = sum(nexts.values())
            log_bigram[prev] = {nxt: float(torch.log(torch.tensor(cnt / total))) for nxt, cnt in nexts.items()}

        def greedy_decode_with_bigram(lp_tbc: torch.Tensor, beta: float, log_prior: torch.Tensor, alpha: float) -> List[List[int]]:
            t, b, c = lp_tbc.shape
            results: List[List[int]] = []
            for bi in range(b):
                lp = lp_tbc[:, bi, :].clone()
                if beta > 0.0:
                    lp = lp - beta * log_prior.unsqueeze(0)
                seq: List[int] = []
                prev: Optional[int] = None
                for ti in range(t):
                    if prev is not None and prev in log_bigram and alpha > 0:
                        bigram_scores = torch.tensor(
                            [log_bigram[prev].get(ci, -10.0) for ci in range(c)],
                            dtype=torch.float32,
                        )
                        combined = lp[ti] + alpha * bigram_scores
                        best = int(combined.argmax().item())
                    else:
                        best = int(lp[ti].argmax().item())
                    seq.append(best)
                    if best != 0:
                        prev = best
                results.append([t for t in seq if t != 0])
            return results

        best_alpha, best_alpha_wer = evaluate_bigram_alpha(
            model,
            dev_dl,
            device,
            vocab_artifacts.log_prior,
            config.bigram_alpha_candidates,
            greedy_decode_with_bigram,
            compute_wer,
        )
        LOGGER.info("Best bigram alpha %.2f with dev WER %.2f", best_alpha, best_alpha_wer * 100)

    results = {
        "config": config.to_dict(),
        "raw_gloss_count": vocab_artifacts.raw_gloss_count,
        "merged_gloss_count": vocab_artifacts.merged_gloss_count,
        "merged_reduction": vocab_artifacts.merged_gloss_reduction,
        "safe_merge_count": len(vocab_artifacts.safe_merge_map),
        "best_dev_wer": float(best_wer),
        "test_wer_single": float(test_wer_single),
        "test_wer_ensemble": float(ensemble_test_wer) if ensemble_test_wer is not None else None,
        "vocab_size": len(vocab_artifacts.vocab),
        "history": history,
    }
    save_json(str(Path(config.results_dir) / "training_results.json"), results)

    if args.plot:
        plot_training_history(history, config.results_dir, config.phase1_epochs, best_wer, ensemble_test_wer)


if __name__ == "__main__":
    main()
