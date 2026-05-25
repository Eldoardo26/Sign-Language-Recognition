from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Configuration container for training and data paths.

    Attributes:
        base_path: Root folder for the dataset.
        annotations_dir: Folder containing annotation CSV files.
        pose_dir: Folder containing pose files by split.
        results_dir: Output directory for results.
        tssi_output_dir: Output directory for cached TSSI files.
        merge_map_path: Optional path to a gloss merge map JSON file.
        seed: Random seed for reproducibility.
        num_epochs: Total number of training epochs.
        early_stopping_patience: Early stopping patience on validation WER.
        checkpoint_every: Save checkpoint every N epochs.
        keep_last_n_checkpoints: Number of recent checkpoints to keep.
        batch_size: Batch size for data loaders.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_amp: Enable mixed precision.
        grad_clip: Gradient clipping value.
        weight_decay: Weight decay for optimizers.
        hidden_dim: Hidden dimension for model.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        num_joints: Number of joints in pose.
        tcn_blocks: Number of TCN blocks.
        phase1_epochs: Number of frozen-backbone epochs.
        phase1_lr: Learning rate for phase 1.
        phase2_lr_backbone: Learning rate for backbone in phase 2.
        phase2_lr_head: Learning rate for head in phase 2.
        prior_beta: Prior scaling beta for decoding.
        ensemble_n: Number of checkpoints for ensemble.
        ctc_smoothing: Entropy regularization weight.
        beam_width: Beam width for decoding.
        frame_h: TSSI temporal height.
        use_gloss_merge: Enable gloss merging.
        min_gloss_freq: Minimum gloss frequency (rare -> <unk>).
        num_workers: DataLoader workers.
        debug_training: Enable debug previews.
        debug_preview_batches: Number of preview batches.
        debug_preview_samples: Number of preview samples per batch.
        debug_topk_classes: Top-k classes for debug summaries.
        debug_save_reports: Save diagnostic reports.
        bigram_alpha_candidates: Candidate alphas for bigram rescoring.
    """

    base_path: str = "C:/data/phoenix"
    annotations_dir: str = ""
    pose_dir: str = ""
    results_dir: str = "results"
    tssi_output_dir: str = "tssi_cache"
    merge_map_path: str = "gloss_merge_map.json"

    seed: int = 42

    num_epochs: int = 150
    early_stopping_patience: int = 20
    checkpoint_every: int = 1
    keep_last_n_checkpoints: int = 10

    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    grad_clip: float = 5.0
    weight_decay: float = 5e-4

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_joints: int = 48
    tcn_blocks: int = 4

    phase1_epochs: int = 15
    phase1_lr: float = 3e-4
    phase2_lr_backbone: float = 1e-4
    phase2_lr_head: float = 1e-4

    prior_beta: float = 0.3
    ensemble_n: int = 3
    ctc_smoothing: float = 0.05
    beam_width: int = 10

    frame_h: int = 240
    use_gloss_merge: bool = True
    min_gloss_freq: int = 5

    num_workers: int = 4

    debug_training: bool = False
    debug_preview_batches: int = 2
    debug_preview_samples: int = 3
    debug_topk_classes: int = 8
    debug_save_reports: bool = True

    bigram_alpha_candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_paths(self) -> "TrainingConfig":
        """Resolve derived paths and normalize to POSIX-like strings.

        Returns:
            Self for chaining.
        """
        base = Path(self.base_path)
        self.annotations_dir = str(base / "annotations" / "annotations" / "manual")
        self.pose_dir = str(base / "pose" / "pose")
        self.results_dir = str(Path(self.results_dir))
        self.tssi_output_dir = str(Path(self.tssi_output_dir))
        self.merge_map_path = str(Path(self.merge_map_path))
        return self

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tssi_output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary with configuration values.
        """
        data = self.__dict__.copy()
        return data


def load_config(path: Optional[str] = None) -> TrainingConfig:
    """Load configuration from YAML or return defaults.

    Args:
        path: Optional path to a YAML configuration file.

    Returns:
        TrainingConfig instance.
    """
    cfg = TrainingConfig().resolve_paths()

    if not path:
        return cfg

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load a YAML config.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value

    return cfg.resolve_paths()
