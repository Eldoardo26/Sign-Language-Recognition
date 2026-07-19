# coding: utf-8
"""
extract_transformer_feats.py — export the encoder features of the Signformer
(the appearance / I3D student of the forward direction) so that it can act as the
TEACHER in the reversed distillation, where the skeleton model is the student.

For every video the frozen Signformer encoder produces a (T, hidden) sequence;
hidden is 256 for the baseline. The sequences are stored as float16, one pickle
per split, keyed by video name --- the same format extract_skeleton_feats.py uses,
so the reverse trainer can reuse the existing distillation code unchanged.

Usage (run from code/distillation/):
    python extract_transformer_feats.py

Environment overrides:
    SIGN_CFG        : Signformer config (default ../signformer/configs/sign.yaml)
    STUDENT_CKPT    : baseline checkpoint (default dataset/checkpoints/sign_sample/best.ckpt)
    TRANSFORMER_FEATS_DIR : output dir (default dataset/features/transformer_feats)
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.enabled = False   # env cuDNN issue (Conv1d/LSTM NOT_INITIALIZED)

# The Signformer framework lives under code/signformer/.
_HERE = Path(__file__).resolve().parent
_SIGNFORMER = _HERE.parent / "signformer"
for _p in (str(_SIGNFORMER), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main.training as mt
from main.batch import Batch
from main.data import load_data, make_data_iter
from main.model import build_model
from main.helpers import load_checkpoint

REPO_ROOT = _HERE.parent.parent
CFG_FILE = os.environ.get("SIGN_CFG", str(_SIGNFORMER / "configs" / "sign.yaml"))
CKPT = os.environ.get(
    "STUDENT_CKPT",
    str(REPO_ROOT / "dataset" / "checkpoints" / "sign_sample" / "best.ckpt"))
OUT_DIR = os.environ.get(
    "TRANSFORMER_FEATS_DIR",
    str(REPO_ROOT / "dataset" / "features" / "transformer_feats"))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = mt.load_config(CFG_FILE)
    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()

    # sign.yaml's data_path is relative to code/signformer/; resolve it to an
    # absolute path so this script works regardless of the current directory.
    dp = cfg["data"]["data_path"]
    if not os.path.isabs(dp):
        cfg["data"]["data_path"] = os.path.normpath(
            os.path.join(str(_SIGNFORMER), dp)) + os.sep

    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"])

    sgn_dim = (sum(cfg["data"]["feature_size"])
               if isinstance(cfg["data"]["feature_size"], list)
               else cfg["data"]["feature_size"])
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0

    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab, txt_vocab=txt_vocab, sgn_dim=sgn_dim,
        do_recognition=do_recognition, do_translation=do_translation)
    ck = load_checkpoint(CKPT, use_cuda=use_cuda)
    model.load_state_dict(ck["model_state"])
    if use_cuda:
        model.cuda()
    model.eval()

    txt_pad_index = txt_vocab.stoi["<pad>"]
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")

    for split, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        feats = {}
        data_iter = make_data_iter(
            dataset=data, batch_size=batch_size, batch_type=batch_type,
            shuffle=False, train=False)
        with torch.no_grad():
            for torch_batch in iter(data_iter):
                batch = Batch(
                    is_train=False, torch_batch=torch_batch,
                    txt_pad_index=txt_pad_index, sgn_dim=sgn_dim,
                    use_cuda=use_cuda,
                    frame_subsampling_ratio=frame_subsampling_ratio)
                # sort so encode sees decreasing lengths; sequence is sorted too,
                # so names and features stay aligned without un-sorting.
                batch.sort_by_sgn_lengths()
                enc_out, _ = model.encode(
                    sgn=batch.sgn, sgn_mask=batch.sgn_mask,
                    sgn_length=batch.sgn_lengths)
                enc_out = enc_out.cpu().float()
                lengths = batch.sgn_lengths.cpu().tolist()
                for name, seq, L in zip(batch.sequence, enc_out, lengths):
                    feats[name] = seq[:int(L)].numpy().astype(np.float16)
        out = os.path.join(OUT_DIR, f"{split}.pkl")
        with open(out, "wb") as f:
            pickle.dump(feats, f)
        dim = next(iter(feats.values())).shape[1] if feats else 0
        print(f"[{split}] wrote {len(feats)} feature sequences (dim {dim}) -> {out}")

    print("Done. Transformer teacher features written to", OUT_DIR)


if __name__ == "__main__":
    main()
