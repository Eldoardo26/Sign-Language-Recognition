from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import get_dfs_order_phoenix_correct

LOGGER = logging.getLogger(__name__)


@dataclass
class VocabArtifacts:
    """Artifacts from vocabulary construction."""

    vocab: List[str]
    g2i: Dict[str, int]
    i2g: Dict[int, str]
    log_prior: torch.Tensor
    raw_gloss_count: int
    merged_gloss_count: int
    merged_gloss_reduction: int
    safe_merge_map: Dict[str, str]


def load_split(annotations_dir: str, split: str) -> pd.DataFrame:
    """Load a split CSV file.

    Args:
        annotations_dir: Directory containing annotation CSV files.
        split: Split name (train/dev/test).

    Returns:
        Loaded DataFrame.
    """
    path = Path(annotations_dir) / f"PHOENIX-2014-T.{split}.corpus.csv"
    return pd.read_csv(path, sep="|")


def normalize_gloss_token(token: str) -> str:
    """Normalize a gloss token.

    Args:
        token: Token string.

    Returns:
        Normalized token.
    """
    token = str(token).strip().upper()
    token = token.replace("_", "-")
    token = "".join(ch for ch in token if ch.isalnum() or ch == "-")
    token = token.replace(" ", "")
    return token


def build_safe_merge_map(glosses: Iterable[str]) -> Dict[str, str]:
    """Build a safe merge map by collapsing hyphen variants.

    Args:
        glosses: Iterable of gloss tokens.

    Returns:
        Safe merge map.
    """
    gloss_set = set(glosses)
    safe_merge_map: Dict[str, str] = {}
    for token in sorted(gloss_set):
        compact = token.replace("-", "")
        if "-" in token and compact in gloss_set and compact != token:
            safe_merge_map[token] = compact
    return safe_merge_map


def apply_merge(gloss_seq_str: str, merge_map: Dict[str, str]) -> List[str]:
    """Apply merge mapping to a gloss sequence string.

    Args:
        gloss_seq_str: Gloss sequence as a string.
        merge_map: Mapping from source to target gloss.

    Returns:
        List of merged tokens.
    """
    merged_tokens: List[str] = []
    for token in str(gloss_seq_str).strip().split():
        normalized = normalize_gloss_token(token)
        if not normalized:
            continue
        merged_tokens.append(merge_map.get(normalized, normalized))
    return merged_tokens


def build_vocab_and_prior(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_gloss_merge: bool,
    merge_map_path: str,
    min_gloss_freq: int,
    debug_save_reports: bool,
    results_dir: str,
) -> VocabArtifacts:
    """Build vocabulary, prior and merge artifacts.

    Args:
        train_df: Train DataFrame.
        dev_df: Dev DataFrame.
        test_df: Test DataFrame.
        use_gloss_merge: Whether to use gloss merge.
        merge_map_path: Path to merge map JSON.
        min_gloss_freq: Minimum frequency for gloss to keep.
        debug_save_reports: Save diagnostics.
        results_dir: Results directory.

    Returns:
        VocabArtifacts instance.
    """
    raw_glosses: List[str] = []
    raw_gloss_counter: Dict[str, int] = {}
    for df in [train_df, dev_df, test_df]:
        for gloss_seq in df["orth"].fillna(""):
            tokens = [normalize_gloss_token(tok) for tok in str(gloss_seq).split()]
            tokens = [tok for tok in tokens if tok]
            for tok in tokens:
                raw_gloss_counter[tok] = raw_gloss_counter.get(tok, 0) + 1
            raw_glosses.extend(tokens)

    raw_gloss_set = set(raw_glosses)
    merge_map: Dict[str, str] = {}
    if use_gloss_merge and Path(merge_map_path).exists():
        with Path(merge_map_path).open("r", encoding="utf-8") as handle:
            merge_map = json.load(handle)
        LOGGER.info("Loaded merge map entries: %d", len(merge_map))

    safe_merge_map = build_safe_merge_map(raw_gloss_set)
    if use_gloss_merge:
        if merge_map:
            for k, v in safe_merge_map.items():
                merge_map.setdefault(k, v)
        else:
            merge_map = dict(safe_merge_map)

    merged_gloss_counter: Dict[str, int] = {}
    merged_glosses: List[str] = []
    for df in [train_df, dev_df, test_df]:
        for gloss_seq in df["orth"].fillna(""):
            tokens = apply_merge(gloss_seq, merge_map)
            for tok in tokens:
                merged_gloss_counter[tok] = merged_gloss_counter.get(tok, 0) + 1
            merged_glosses.extend(tokens)

    raw_gloss_count = len(raw_gloss_set)
    merged_gloss_set = set(merged_glosses)
    merged_gloss_count = len(merged_gloss_set)
    merged_gloss_reduction = raw_gloss_count - merged_gloss_count

    rare_glosses = {g for g, cnt in merged_gloss_counter.items() if cnt < min_gloss_freq and g != "<blank>"}
    vocab = ["<blank>", "<unk>"] + sorted(
        g for g in merged_gloss_set if g not in rare_glosses and g != "<blank>"
    )
    g2i = {g: i for i, g in enumerate(vocab)}
    i2g = {i: g for g, i in g2i.items()}

    token_counts: Dict[int, int] = {}
    for seq in train_df["orth"].fillna(""):
        for g in apply_merge(seq, merge_map):
            if g in g2i:
                token_counts[g2i[g]] = token_counts.get(g2i[g], 0) + 1

    prior = torch.zeros(len(vocab))
    for idx, cnt in token_counts.items():
        prior[idx] = cnt
    prior[0] = prior[1:].sum() * 0.01
    prior = prior / prior.sum()
    log_prior = torch.log(prior.clamp(min=1e-8))

    if debug_save_reports and use_gloss_merge and merge_map and not Path(merge_map_path).exists():
        with Path(merge_map_path).open("w", encoding="utf-8") as handle:
            json.dump(merge_map, handle, indent=2, ensure_ascii=False)

    return VocabArtifacts(
        vocab=vocab,
        g2i=g2i,
        i2g=i2g,
        log_prior=log_prior,
        raw_gloss_count=raw_gloss_count,
        merged_gloss_count=merged_gloss_count,
        merged_gloss_reduction=merged_gloss_reduction,
        safe_merge_map=safe_merge_map,
    )


def index_pose_files(pose_dir: str, split: str) -> Dict[str, str]:
    """Index pose files for a split.

    Args:
        pose_dir: Pose directory.
        split: Split name.

    Returns:
        Mapping from video id to file path.
    """
    split_dir = Path(pose_dir) / split
    kp_dict: Dict[str, str] = {}
    if not split_dir.exists():
        LOGGER.warning("Pose split missing: %s", split_dir)
        return kp_dict
    for fname in split_dir.iterdir():
        if fname.suffix == ".npy":
            kp_dict[fname.stem] = str(fname)
    return kp_dict


def generate_tssi_optimized(keypoints: np.ndarray, frame_h: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Generate TSSI tensor from keypoints.

    Args:
        keypoints: Keypoint array (T, J, C).
        frame_h: Optional target temporal height.

    Returns:
        Tuple (tssi, seq_len).
    """
    try:
        kp = keypoints.astype(np.float32)
        t_raw, j, c = kp.shape

        if frame_h is None:
            frame_h = max(120, t_raw)

        if c == 2:
            x = kp[:, :, 0].copy()
            y = kp[:, :, 1].copy()
            conf = np.ones((t_raw, j), dtype=np.float32)
        else:
            x = kp[:, :, 0].copy()
            y = kp[:, :, 1].copy()
            conf = kp[:, :, 2].copy()

        for jj in range(j):
            valid_idx = np.where(conf[:, jj] > 0.3)[0]
            if len(valid_idx) >= 2:
                x[:, jj] = np.interp(np.arange(t_raw), valid_idx, x[valid_idx, jj])
                y[:, jj] = np.interp(np.arange(t_raw), valid_idx, y[valid_idx, jj])
                conf[:, jj] = np.interp(np.arange(t_raw), valid_idx, conf[valid_idx, jj])
            elif len(valid_idx) == 0:
                conf[:, jj] = 0.0

        x = gaussian_filter1d(x, sigma=1.5, axis=0)
        y = gaussian_filter1d(y, sigma=1.5, axis=0)
        conf = gaussian_filter1d(conf, sigma=1.5, axis=0)

        for start, end in [(0, 6), (6, 27), (27, 48)]:
            end = min(end, j)
            if end <= start:
                continue
            for arr in [x, y]:
                a_min = arr[:, start:end].min()
                a_max = arr[:, start:end].max()
                rng = max(a_max - a_min, 1e-6)
                arr[:, start:end] = (arr[:, start:end] - a_min) / rng

        def _resize_temporal(arr: np.ndarray, target_h: int) -> np.ndarray:
            img = arr.astype(np.float32)
            if img.shape[0] == 1:
                img = np.repeat(img, target_h, axis=0)
                return img[:target_h]
            return cv2.resize(img, (j, target_h), interpolation=cv2.INTER_LINEAR)

        x_r = _resize_temporal(x, frame_h)
        y_r = _resize_temporal(y, frame_h)
        conf_r = _resize_temporal(conf, frame_h)

        dfs_order = list(dict.fromkeys(get_dfs_order_phoenix_correct()))
        n_unique = len(dfs_order)

        tssi = np.zeros((3, n_unique, frame_h), dtype=np.float32)
        for col_idx, joint_id in enumerate(dfs_order):
            if joint_id >= j:
                continue
            tssi[0, col_idx, :] = np.clip(x_r[:, joint_id], 0, 1)
            tssi[1, col_idx, :] = np.clip(y_r[:, joint_id], 0, 1)
            tssi[2, col_idx, :] = np.clip(conf_r[:, joint_id], 0, 1)

        return tssi, min(t_raw, frame_h)
    except Exception as exc:
        LOGGER.exception("Error in generate_tssi_optimized: %s", exc)
        fallback_h = frame_h if frame_h is not None else 120
        expected_j = len(list(dict.fromkeys(get_dfs_order_phoenix_correct())))
        return np.zeros((3, expected_j, fallback_h), dtype=np.float32), 1


def augment_tssi_fixed(tssi: np.ndarray, augment: bool = True) -> np.ndarray:
    """Augment TSSI in a memory-efficient way.

    Args:
        tssi: TSSI array (C, J, T).
        augment: Whether to apply augmentation.

    Returns:
        Augmented TSSI.
    """
    if not augment:
        return tssi

    c, j, t = tssi.shape
    if np.random.rand() < 0.8:
        noise = gaussian_filter1d(np.random.randn(2, j, t).astype(np.float32), sigma=3, axis=2)
        tssi[:2] = np.clip(tssi[:2] + noise * 0.008, 0, 1)

    if np.random.rand() < 0.7:
        speeds = np.random.uniform(0.7, 1.3, t).astype(np.float32)
        indices = np.cumsum(speeds)
        indices = (indices / indices[-1] * (t - 1)).astype(np.float32)
        src = np.arange(t, dtype=np.float32)
        warped = np.stack(
            [np.clip(np.array([np.interp(indices, src, tssi[ch, jj]) for jj in range(j)]), 0, 1) for ch in range(c)]
        )
        tssi = warped

    if np.random.rand() < 0.5:
        spatial_noise = np.random.randn(2, j, t).astype(np.float32) * 0.005
        tssi[:2] = np.clip(tssi[:2] + spatial_noise, 0, 1)

    if np.random.rand() < 0.7:
        for _ in range(2):
            t_len = np.random.randint(1, max(2, int(t * 0.15)))
            t_start = np.random.randint(0, max(1, t - t_len))
            tssi[:, :, t_start : t_start + t_len] = 0.0

    if np.random.rand() < 0.5:
        mask = np.random.rand(j) < 0.15
        tssi[:, mask, :] = 0.0

    return tssi


class PhoenixDatasetContinuous(Dataset):
    """Dataset for TSSI pose sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        kp_dict: Dict[str, str],
        g2i: Dict[str, int],
        augment: bool = False,
        frame_h: int = 240,
        tssi_dir: Optional[str] = None,
    ) -> None:
        self.kp_dict = kp_dict
        self.g2i = g2i
        self.augment = augment
        self.frame_h = frame_h
        self.tssi_dir = tssi_dir
        self.samples: List[Dict[str, object]] = []

        for _, row in df.iterrows():
            vid = str(row["name"])
            if vid not in kp_dict:
                continue
            gloss_str = str(row["orth"]).strip().upper()
            labels = [g2i[g] for g in gloss_str.split() if g in g2i]
            if labels:
                self.samples.append({"vid": vid, "labels": labels})

        LOGGER.info("Dataset samples: %d | augment=%s", len(self.samples), augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        vid = sample["vid"]

        if self.tssi_dir is not None:
            npz_path = Path(self.tssi_dir) / f"{vid}.npz"
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=False)
                tssi = data["tssi"].astype(np.float32)
                seq_len = int(data["seq_len"])
            else:
                kp = np.load(self.kp_dict[vid]).astype(np.float32)
                tssi, seq_len = generate_tssi_optimized(kp, frame_h=self.frame_h)
        else:
            kp = np.load(self.kp_dict[vid]).astype(np.float32)
            tssi, seq_len = generate_tssi_optimized(kp, frame_h=self.frame_h)

        if self.augment:
            tssi = augment_tssi_fixed(tssi, augment=True)

        return {
            "tssi": torch.from_numpy(tssi).float(),
            "labels": sample["labels"],
            "seq_len": seq_len,
        }


def collate_fn_ctc(batch: List[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for CTC training.

    Args:
        batch: Batch of samples.

    Returns:
        Tuple of tensors (tssies, targets, input_lengths, target_lengths).
    """
    max_h = max(item["tssi"].shape[2] for item in batch)
    tssies: List[torch.Tensor] = []
    input_lengths: List[int] = []
    for item in batch:
        t = item["tssi"]
        pad = max_h - t.shape[2]
        tssies.append(F.pad(t, (0, pad)))
        input_lengths.append(t.shape[2])

    tssies_tensor = torch.stack(tssies)
    targets = torch.cat([torch.LongTensor(item["labels"]) for item in batch])
    input_lengths_tensor = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor([len(item["labels"]) for item in batch])
    return tssies_tensor, targets, input_lengths_tensor, target_lengths


def cache_tssi_for_split(
    split: str,
    kp_dict: Dict[str, str],
    output_dir: str,
    frame_h: int,
) -> None:
    """Cache TSSI tensors for a split.

    Args:
        split: Split name.
        kp_dict: Mapping from video id to keypoint file path.
        output_dir: Output directory.
        frame_h: TSSI frame height.
    """
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Caching TSSI for split %s", split)
    for vid_name, kp_path in tqdm(kp_dict.items(), desc=split):
        save_path = split_dir / f"{vid_name}.npz"
        if save_path.exists():
            continue
        kp_array = np.load(kp_path)
        tssi_tensor, seq_len = generate_tssi_optimized(kp_array, frame_h=frame_h)
        np.savez_compressed(save_path, tssi=tssi_tensor, seq_len=seq_len)


def build_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_kp: Dict[str, str],
    dev_kp: Dict[str, str],
    test_kp: Dict[str, str],
    g2i: Dict[str, int],
    batch_size: int,
    num_workers: int,
    frame_h: int,
    tssi_output_dir: Optional[str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/dev/test dataloaders.

    Args:
        train_df: Train DataFrame.
        dev_df: Dev DataFrame.
        test_df: Test DataFrame.
        train_kp: Train keypoint index.
        dev_kp: Dev keypoint index.
        test_kp: Test keypoint index.
        g2i: Gloss-to-id mapping.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        frame_h: TSSI frame height.
        tssi_output_dir: Base TSSI cache directory.

    Returns:
        Tuple of train/dev/test DataLoader.
    """
    train_dir = str(Path(tssi_output_dir) / "train") if tssi_output_dir else None
    dev_dir = str(Path(tssi_output_dir) / "dev") if tssi_output_dir else None
    test_dir = str(Path(tssi_output_dir) / "test") if tssi_output_dir else None

    train_ds = PhoenixDatasetContinuous(
        train_df, train_kp, g2i, augment=True, frame_h=frame_h, tssi_dir=train_dir
    )
    dev_ds = PhoenixDatasetContinuous(
        dev_df, dev_kp, g2i, augment=False, frame_h=frame_h, tssi_dir=dev_dir
    )
    test_ds = PhoenixDatasetContinuous(
        test_df, test_kp, g2i, augment=False, frame_h=frame_h, tssi_dir=test_dir
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_ctc,
        drop_last=True,
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_ctc,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_ctc,
    )

    return train_dl, dev_dl, test_dl
