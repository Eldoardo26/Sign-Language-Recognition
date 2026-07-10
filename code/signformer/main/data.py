# coding: utf-8
"""
Data module — rewritten to remove torchtext legacy API (Field/BucketIterator).
Uses torch.utils.data.DataLoader with a custom collate_fn instead.
"""
import os
import sys
import random
from types import SimpleNamespace
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main.dataset import SignTranslationDataset
from main.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict):
    """Load train/dev/test datasets and build vocabularies."""

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths   = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths  = [os.path.join(data_path, x) for x in data_cfg["test"]]
        pad_feature_size = sum(data_cfg["feature_size"])
    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths   = os.path.join(data_path, data_cfg["dev"])
        test_paths  = os.path.join(data_path, data_cfg["test"])
        pad_feature_size = data_cfg["feature_size"]

    level         = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def _filter(ex):
        return (
            len(ex.sgn) <= max_sent_length
            and len(ex.txt) <= max_sent_length
        )

    train_data = SignTranslationDataset(
        path=train_paths, fields=None, filter_pred=_filter
    )
    dev_data  = SignTranslationDataset(path=dev_paths,  fields=None)
    test_data = SignTranslationDataset(path=test_paths, fields=None)

    # ── lowercase text tokens if requested ───────────────────────────────────
    if txt_lowercase:
        for ds in (train_data, dev_data, test_data):
            for ex in ds.examples:
                ex.txt = [t.lower() for t in ex.txt]

    # ── word/char tokenization ────────────────────────────────────────────────
    if level == "char":
        for ds in (train_data, dev_data, test_data):
            for ex in ds.examples:
                ex.gls = list(" ".join(ex.gls))
                ex.txt = list(" ".join(ex.txt))

    # ── vocabularies ─────────────────────────────────────────────────────────
    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab = build_vocab(
        field="gls", min_freq=gls_min_freq, max_size=gls_max_size,
        dataset=train_data, vocab_file=data_cfg.get("gls_vocab"),
    )
    txt_vocab = build_vocab(
        field="txt", min_freq=txt_min_freq, max_size=txt_max_size,
        dataset=train_data, vocab_file=data_cfg.get("txt_vocab"),
    )

    # ── store vocab on datasets so make_data_iter can access them ────────────
    for ds in (train_data, dev_data, test_data):
        ds.gls_vocab       = gls_vocab
        ds.txt_vocab       = txt_vocab
        ds.pad_feature_size = pad_feature_size

    # ── optional random subsets ───────────────────────────────────────────────
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        k = min(random_train_subset, len(train_data))
        train_data.examples = random.sample(train_data.examples, k)

    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        k = min(random_dev_subset, len(dev_data))
        dev_data.examples = random.sample(dev_data.examples, k)

    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# ── collation helpers ──────────────────────────────────────────────────────────

def _numericalize(token_lists: List[List[str]], vocab: Vocabulary,
                  bos: bool = False, eos: bool = False,
                  pad_id: int = None) -> tuple:
    """Convert list-of-token-lists → padded LongTensor + lengths."""
    if pad_id is None:
        pad_id = vocab.stoi[PAD_TOKEN]

    seqs = []
    for tokens in token_lists:
        ids = []
        if bos:
            ids.append(vocab.stoi[BOS_TOKEN])
        ids.extend(vocab.stoi[t] for t in tokens)
        if eos:
            ids.append(vocab.stoi[EOS_TOKEN])
        seqs.append(ids)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()
    padded  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths


def _pad_sgn(frame_lists: List[List[torch.Tensor]], feat_dim: int) -> tuple:
    """Pad variable-length sign feature sequences."""
    lengths = torch.tensor([len(f) for f in frame_lists], dtype=torch.long)
    max_len = lengths.max().item()
    padded  = torch.zeros(len(frame_lists), max_len, feat_dim)
    for i, frames in enumerate(frame_lists):
        t = torch.stack(frames, dim=0)          # (T, feat_dim)
        padded[i, : t.shape[0]] = t
    return padded, lengths


def _make_collate(gls_vocab: Vocabulary, txt_vocab: Vocabulary, pad_feature_size: int,
                  sort_by_sgn: bool = False):
    gls_pad = gls_vocab.stoi[PAD_TOKEN]
    txt_pad = txt_vocab.stoi[PAD_TOKEN]

    def collate_fn(batch):
        # batch is a list of SimpleNamespace examples
        if sort_by_sgn:
            batch = sorted(batch, key=lambda ex: len(ex.sgn), reverse=True)

        sgn_padded, sgn_lengths = _pad_sgn(
            [ex.sgn for ex in batch], pad_feature_size
        )

        gls_padded, gls_lengths = _numericalize(
            [ex.gls for ex in batch], gls_vocab, pad_id=gls_pad
        )

        # txt: add BOS at front, EOS at end (teacher forcing)
        txt_with_bos_eos, txt_lengths = _numericalize(
            [ex.txt for ex in batch], txt_vocab,
            bos=True, eos=True, pad_id=txt_pad
        )

        return SimpleNamespace(
            sequence=[ex.sequence for ex in batch],
            signer  =[ex.signer   for ex in batch],
            sgn     =(sgn_padded,  sgn_lengths),
            gls     =(gls_padded,  gls_lengths),
            txt     =(txt_with_bos_eos, txt_lengths),
        )

    return collate_fn


# ── public API ────────────────────────────────────────────────────────────────

def make_data_iter(
    dataset: SignTranslationDataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader that produces batches compatible with Batch in batch.py."""

    collate_fn = _make_collate(
        gls_vocab       = dataset.gls_vocab,
        txt_vocab       = dataset.txt_vocab,
        pad_feature_size= dataset.pad_feature_size,
        # NON pre-ordinare: Batch.sort_by_sgn_lengths() ordina internamente e
        # restituisce il reverse-index. Pre-ordinare qui romperebbe l'allineamento
        # hyp/ref in validate_on_data (data.gls è in ordine dataset) → WER errato.
        sort_by_sgn     = False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and train),
        collate_fn=collate_fn,
        num_workers=0,        # 0 = main process, safe on Windows
        drop_last=False,
    )
