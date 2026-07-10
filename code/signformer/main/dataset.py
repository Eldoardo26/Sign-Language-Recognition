# coding: utf-8
"""
Data module — rewritten to remove torchtext legacy API (Field/Example/Dataset).
Uses plain torch.utils.data.Dataset instead.
"""
import gzip
import pickle
from types import SimpleNamespace
from typing import List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


class SignTranslationDataset(Dataset):
    """Dataset for sign-language translation (I3D features → gloss/text)."""

    def __init__(
        self,
        path,
        fields: Tuple,          # kept for API compatibility, unused
        filter_pred: Optional[Callable] = None,
        **kwargs,
    ):
        if not isinstance(path, list):
            path = [path]

        # ── load and deduplicate samples ─────────────────────────────────────
        raw: dict = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in raw:
                    assert raw[seq_id]["signer"] == s["signer"]
                    assert raw[seq_id]["gloss"]  == s["gloss"]
                    assert raw[seq_id]["text"]   == s["text"]
                    raw[seq_id]["sign"] = torch.cat(
                        [raw[seq_id]["sign"], s["sign"]], dim=1
                    )
                else:
                    raw[seq_id] = {
                        "name":   s["name"],
                        "signer": s["signer"],
                        "gloss":  s["gloss"],
                        "text":   s["text"],
                        "sign":   s["sign"],
                    }

        # ── build Example objects (SimpleNamespace) ───────────────────────────
        # sgn  = list of 1-D tensors (one per frame), matching original tokenize_features
        # gls  = list of gloss tokens
        # txt  = list of text tokens  (lowercased later in load_data)
        examples = []
        for s in raw.values():
            sign: torch.Tensor = s["sign"].float() + 1e-8   # (T, feat_dim)
            sgn_frames = [sign[i] for i in range(sign.shape[0])]
            gls_tokens = s["gloss"].strip().split()
            txt_tokens = s["text"].strip().split()

            ex = SimpleNamespace(
                sequence=s["name"],
                signer=s["signer"],
                sgn=sgn_frames,          # list[Tensor(feat_dim)]
                gls=gls_tokens,          # list[str]
                txt=txt_tokens,          # list[str]
            )
            examples.append(ex)

        # ── optional filter (e.g. max sequence length) ───────────────────────
        if filter_pred is not None:
            examples = [e for e in examples if filter_pred(e)]

        self.examples: List[SimpleNamespace] = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    # ── aggregate views over all examples (compat con validate_on_data) ──────
    # prediction.py accede a data.gls / data.txt / data.sequence come liste
    # allineate su tutti gli esempi del dataset.
    @property
    def gls(self):
        return [e.gls for e in self.examples]

    @property
    def txt(self):
        return [e.txt for e in self.examples]

    @property
    def sequence(self):
        return [e.sequence for e in self.examples]

    @property
    def signer(self):
        return [e.signer for e in self.examples]

    @property
    def sgn(self):
        return [e.sgn for e in self.examples]
