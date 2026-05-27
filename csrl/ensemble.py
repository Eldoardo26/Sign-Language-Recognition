"""
ensemble.py — Model ensemble con media log-probabilità (Deep Sign §6.5).

La media delle log-prob equivale al log-linear combination del paper;
con N modelli uniformi δ = 1/N. Per ottimizzare i δ si può fare una
ricerca su griglia sul dev set (non implementata qui).
"""

import gc
import os

import torch
from tqdm import tqdm

from model import PoseNetworkCTC
from decoding import beam_search_ctc_optimized
from metrics import compute_wer


def ensemble_decode(
    model_paths: list,
    dl,
    device,
    config: dict,
    log_prior: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Carica N checkpoint, media le log-prob e decodifica con beam search.

    Args:
        model_paths : lista di percorsi .pth
        dl          : DataLoader di valutazione
        device      : torch.device
        config      : CONFIG dict
        log_prior   : Tensor (num_classes,)
        num_classes : numero di classi del vocabolario

    Returns:
        wer : float — Word Error Rate dell'ensemble
    """
    ensemble = []
    for path in model_paths:
        m = PoseNetworkCTC(
            num_classes=num_classes,
            num_joints=config["num_joints"],
            hidden_dim=config["hidden_dim"],
            tcn_blocks=config["tcn_blocks"],
            lstm_layers=config["num_layers"],
            dropout=0.0,
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        ensemble.append(m)
    print(f"Ensemble: {len(ensemble)} modelli caricati")

    beta        = config["prior_beta"]
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for tssies, targets, input_lengths, target_lengths in tqdm(dl, desc="Ensemble"):
            tssies = tssies.to(device)

            # Media log-prob
            avg_lp = None
            for m in ensemble:
                lp     = m(tssies).permute(1, 0, 2).float()   # (T, B, C)
                avg_lp = lp if avg_lp is None else avg_lp + lp
            avg_lp = avg_lp / len(ensemble)

            decoded = beam_search_ctc_optimized(
                avg_lp.cpu(),
                beam_width=config["beam_width"],
                beta=beta,
                log_prior=log_prior.cpu(),
            )

            refs, offset = [], 0
            for tlen in target_lengths.tolist():
                refs.append(targets[offset:offset + tlen].tolist())
                offset += tlen
            all_refs.extend(refs)
            all_hyps.extend(decoded)

            del tssies, avg_lp
            gc.collect()

    wer, _ = compute_wer(all_refs, all_hyps)
    return wer
