"""Headless teacher training on the RTMW 133-kp data with the spatial-GCN model.

Runs the same two-phase schedule as runner.ipynb but as a script, so it can be
launched in the background after prepare_phoenix_133.py. Reports best dev WER and
a greedy + beam test-set decoding at the end.

Usage:  python train_teacher.py
"""

import os
import torch

from config import DEVICE, AMP, DATA_DIR, REPO_ROOT, CFG
from vocab import build_vocab_from_raw
from dataset import load_pkl, make_dataloaders
from model import PoseNetworkCTC
from training import run_training, evaluate
import skeleton as s


def main():
    print(f"device {DEVICE} | amp {AMP} | layout {s.LAYOUT} | joints {s.NUM_JOINTS}")
    print(f"DATA_DIR {DATA_DIR}")

    train_raw = load_pkl(os.path.join(DATA_DIR, "Phoenix-2014T.train"))
    dev_raw = load_pkl(os.path.join(DATA_DIR, "Phoenix-2014T.dev"))
    test_raw = load_pkl(os.path.join(DATA_DIR, "Phoenix-2014T.test"))
    print(f"raw: train {len(train_raw)} | dev {len(dev_raw)} | test {len(test_raw)}")

    V = build_vocab_from_raw(train_raw, dev_raw, test_raw)
    NUM_CLASSES = V["num_classes"]
    LOG_PRIOR = V["log_prior"].to(DEVICE)

    train_loader, dev_loader, test_loader, *_ = make_dataloaders(
        DATA_DIR, V["gloss_to_ids"], CFG
    )

    model = PoseNetworkCTC(
        num_classes=NUM_CLASSES,
        in_channels=CFG["num_joints"] * 3,
        hidden_dim=CFG["hidden_dim"],
        tcn_blocks=CFG["tcn_blocks"],
        lstm_layers=CFG["num_layers"],
        dropout=CFG["dropout"],
        drop_path_rate=CFG["drop_path_rate"],
        attn_heads=CFG["attn_heads"],
        use_gcn=CFG["use_gcn"],
        gcn_channels=CFG["gcn_channels"],
    ).to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f} M | "
          f"use_gcn {CFG['use_gcn']} | classes {NUM_CLASSES}")

    ckpt_dir = os.path.join(REPO_ROOT, "dataset", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "tssi133_gcn_best.pt")

    hist, best = run_training(
        model, train_loader, dev_loader, CFG, DEVICE, AMP, LOG_PRIOR,
        ckpt_path=ckpt, fresh_start=True,
    )

    # final test-set evaluation at the best-on-dev checkpoint
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    wer_g, dg, _ = evaluate(model, test_loader, CFG, DEVICE, LOG_PRIOR, use_beam=False)
    wer_b, db, _ = evaluate(model, test_loader, CFG, DEVICE, LOG_PRIOR, use_beam=True)

    print("=" * 60)
    print(f"BEST DEV WER : {best*100:.2f}%")
    print(f"TEST greedy  : {wer_g*100:.2f}%  (S{dg['S']} D{dg['D']} I{dg['I']} N{dg['N']})")
    print(f"TEST beam    : {wer_b*100:.2f}%  (S{db['S']} D{db['D']} I{db['I']} N{db['N']})")
    print(f"ckpt: {ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
