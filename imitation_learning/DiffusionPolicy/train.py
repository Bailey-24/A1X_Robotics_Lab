#!/usr/bin/env python3
"""
Diffusion Policy Training for A1X Robot
========================================

Standalone, modifiable training script using LeRobot's DiffusionPolicy
with a local LeRobot-format dataset. Uses DDIM noise scheduler for faster
inference.

Usage:
    # Train with default config
    python imitation_learning/DiffusionPolicy/train.py

    # Override specific parameters via CLI
    python imitation_learning/DiffusionPolicy/train.py --batch_size 32 --lr 5e-5

    # Use a custom config file
    python imitation_learning/DiffusionPolicy/train.py --config path/to/config.yaml

    # Resume from checkpoint
    python imitation_learning/DiffusionPolicy/train.py --resume outputs/train/.../checkpoints/last

Prerequisites:
    1. Conda environment: conda activate a1x_ros
    2. LeRobot installed: pip install -e refence_code/lerobot
    3. Dataset converted to LeRobot format
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_SCRIPT_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dp_train")


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetCfg:
    repo_id: str = "a1x/yoloe_grasp_white_object"
    root: str = "~/.cache/huggingface/lerobot/a1x/yoloe_grasp_white_object"
    use_imagenet_stats: bool = True
    video_backend: str = "pyav"


@dataclass
class PolicyCfg:
    # Temporal
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None  # None when use_group_norm=True
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Image preprocessing
    resize_shape: tuple[int, int] | None = (240, 320)
    crop_ratio: float = 0.95
    crop_is_random: bool = True

    # UNet
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    num_inference_steps: int | None = 10
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Normalization
    normalization: dict[str, str] = field(default_factory=lambda: {
        "visual": "MEAN_STD",
        "state": "MIN_MAX",
        "action": "MIN_MAX",
    })

    # Loss
    do_mask_loss_for_padding: bool = False


@dataclass
class TrainingCfg:
    epochs: int = 300
    steps: int | None = None
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42

    # Optimizer
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    grad_clip_norm: float = 10.0

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 500

    # AMP
    use_amp: bool = False


@dataclass
class LoggingCfg:
    log_freq: int = 100
    save_freq: int = 5000
    save_last: bool = True
    wandb_enable: bool = False
    wandb_project: str = "a1x_dp_training"
    wandb_entity: str | None = None
    wandb_name: str | None = None


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    output_dir: str | None = None
    device: str = "cuda"
    gpu_id: int = 0


def load_config(config_path: str | Path, cli_overrides: dict | None = None) -> TrainConfig:
    """Load config from YAML, then apply CLI overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file %s not found, using defaults", config_path)
        return TrainConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = TrainConfig()

    # ── Dataset section ──
    ds = raw.get("dataset", {})
    cfg.dataset.repo_id = ds.get("repo_id", cfg.dataset.repo_id)
    cfg.dataset.root = ds.get("root", cfg.dataset.root)
    cfg.dataset.use_imagenet_stats = ds.get("use_imagenet_stats", cfg.dataset.use_imagenet_stats)
    cfg.dataset.video_backend = ds.get("video_backend", cfg.dataset.video_backend)

    # ── Policy section ──
    pol = raw.get("policy", {})
    for key in [
        "n_obs_steps", "horizon", "n_action_steps",
        "vision_backbone", "pretrained_backbone_weights",
        "use_group_norm", "spatial_softmax_num_keypoints",
        "use_separate_rgb_encoder_per_camera",
        "crop_ratio", "crop_is_random",
        "kernel_size", "n_groups", "diffusion_step_embed_dim",
        "use_film_scale_modulation",
        "noise_scheduler_type", "num_train_timesteps", "num_inference_steps",
        "beta_schedule", "beta_start", "beta_end",
        "prediction_type", "clip_sample", "clip_sample_range",
        "do_mask_loss_for_padding",
    ]:
        if key in pol:
            setattr(cfg.policy, key, pol[key])

    if "resize_shape" in pol:
        v = pol["resize_shape"]
        cfg.policy.resize_shape = tuple(v) if v is not None else None
    if "down_dims" in pol:
        cfg.policy.down_dims = tuple(pol["down_dims"])
    if "normalization" in pol:
        cfg.policy.normalization = pol["normalization"]

    # ── Training section ──
    tr = raw.get("training", {})
    for key in [
        "epochs", "steps", "batch_size", "num_workers", "seed",
        "lr", "eps", "weight_decay", "grad_clip_norm",
        "scheduler", "warmup_steps", "use_amp",
    ]:
        if key in tr:
            setattr(cfg.training, key, tr[key])
    if "betas" in tr:
        cfg.training.betas = tuple(tr["betas"])

    # ── Logging section ──
    log_raw = raw.get("logging", {})
    for key in ["log_freq", "save_freq", "save_last"]:
        if key in log_raw:
            setattr(cfg.logging, key, log_raw[key])
    wb = log_raw.get("wandb", {})
    if wb:
        cfg.logging.wandb_enable = wb.get("enable", cfg.logging.wandb_enable)
        cfg.logging.wandb_project = wb.get("project", cfg.logging.wandb_project)
        cfg.logging.wandb_entity = wb.get("entity", cfg.logging.wandb_entity)
        cfg.logging.wandb_name = wb.get("name", cfg.logging.wandb_name)

    # ── Output / hardware ──
    out = raw.get("output", {})
    cfg.output_dir = out.get("dir", cfg.output_dir)

    hw = raw.get("hardware", {})
    cfg.device = hw.get("device", cfg.device)
    cfg.gpu_id = hw.get("gpu_id", cfg.gpu_id)

    # ── CLI overrides ──
    if cli_overrides:
        _apply_cli_overrides(cfg, cli_overrides)

    return cfg


def _apply_cli_overrides(cfg: TrainConfig, overrides: dict) -> None:
    """Apply flat CLI overrides like --lr, --batch_size, --horizon, etc."""
    mapping = {
        # Training
        "lr": ("training", "lr"),
        "batch_size": ("training", "batch_size"),
        "epochs": ("training", "epochs"),
        "steps": ("training", "steps"),
        "num_workers": ("training", "num_workers"),
        "seed": ("training", "seed"),
        "grad_clip_norm": ("training", "grad_clip_norm"),
        "warmup_steps": ("training", "warmup_steps"),
        "use_amp": ("training", "use_amp"),
        # Policy
        "horizon": ("policy", "horizon"),
        "n_obs_steps": ("policy", "n_obs_steps"),
        "n_action_steps": ("policy", "n_action_steps"),
        "vision_backbone": ("policy", "vision_backbone"),
        "noise_scheduler_type": ("policy", "noise_scheduler_type"),
        "num_train_timesteps": ("policy", "num_train_timesteps"),
        "num_inference_steps": ("policy", "num_inference_steps"),
        # Dataset
        "dataset": ("dataset", "repo_id"),
        "dataset_root": ("dataset", "root"),
        # Logging
        "log_freq": ("logging", "log_freq"),
        "save_freq": ("logging", "save_freq"),
        "wandb": ("logging", "wandb_enable"),
        "wandb_project": ("logging", "wandb_project"),
        # Output / Hardware
        "output_dir": (None, "output_dir"),
        "device": (None, "device"),
        "gpu_id": (None, "gpu_id"),
    }
    for cli_key, value in overrides.items():
        if value is None:
            continue
        if cli_key in mapping:
            section, attr = mapping[cli_key]
            if section is None:
                setattr(cfg, attr, value)
            else:
                setattr(getattr(cfg, section), attr, value)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(cfg: TrainConfig):
    """Build a LeRobotDataset with proper delta_timestamps for diffusion."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    root = str(Path(cfg.dataset.root).expanduser())

    # We need a DiffusionConfig to compute delta_timestamps from its properties
    policy_cfg = _build_diffusion_config(cfg)

    # Load dataset metadata first
    ds_meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset.repo_id,
        root=root,
    )

    # Compute delta_timestamps from observation/action delta indices
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    logger.info("Delta timestamps: %s", {
        k: f"[{v[0]:.3f} ... {v[-1]:.3f}] ({len(v)} steps)"
        for k, v in (delta_timestamps or {}).items()
    })

    # Build dataset
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )

    # Inject ImageNet stats for camera features (standard practice for pretrained backbones)
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats[key]["mean"] = torch.tensor(
                [[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32,
            )
            dataset.meta.stats[key]["std"] = torch.tensor(
                [[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32,
            )
        logger.info("Injected ImageNet stats for cameras: %s", dataset.meta.camera_keys)

    logger.info(
        "Dataset loaded: %d episodes, %d frames, %d fps",
        dataset.meta.total_episodes, dataset.meta.total_frames, dataset.meta.fps,
    )

    return dataset, ds_meta


def build_dataloader(dataset, cfg: TrainConfig):
    """Build DataLoader with EpisodeAwareSampler (drops last N frames per episode)."""
    from lerobot.datasets.sampler import EpisodeAwareSampler

    drop_n = cfg.policy.horizon - cfg.policy.n_action_steps - cfg.policy.n_obs_steps + 1
    drop_n = max(0, drop_n)
    logger.info("EpisodeAwareSampler: drop_n_last_frames=%d", drop_n)

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=drop_n,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.device == "cuda",
        drop_last=True,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )

    return dataloader


# ═══════════════════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════════════════

def _build_diffusion_config(cfg: TrainConfig):
    """Create a DiffusionConfig from our YAML-based config."""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.configs.types import NormalizationMode

    norm_map = {}
    norm_str_to_mode = {
        "MEAN_STD": NormalizationMode.MEAN_STD,
        "MIN_MAX": NormalizationMode.MIN_MAX,
        "IDENTITY": NormalizationMode.IDENTITY,
    }
    for key, val in cfg.policy.normalization.items():
        norm_map[key.upper()] = norm_str_to_mode[val.upper()]

    # Validate: pretrained weights + group norm are incompatible
    if cfg.policy.use_group_norm and cfg.policy.pretrained_backbone_weights is not None:
        raise ValueError(
            "use_group_norm=True and pretrained_backbone_weights are mutually exclusive. "
            "GroupNorm replaces BatchNorm, ruining pretrained weights. "
            "Set pretrained_backbone_weights to null, or set use_group_norm to false."
        )

    return DiffusionConfig(
        n_obs_steps=cfg.policy.n_obs_steps,
        horizon=cfg.policy.horizon,
        n_action_steps=cfg.policy.n_action_steps,
        normalization_mapping=norm_map,
        vision_backbone=cfg.policy.vision_backbone,
        resize_shape=cfg.policy.resize_shape,
        crop_ratio=cfg.policy.crop_ratio,
        crop_is_random=cfg.policy.crop_is_random,
        pretrained_backbone_weights=cfg.policy.pretrained_backbone_weights,
        use_group_norm=cfg.policy.use_group_norm,
        spatial_softmax_num_keypoints=cfg.policy.spatial_softmax_num_keypoints,
        use_separate_rgb_encoder_per_camera=cfg.policy.use_separate_rgb_encoder_per_camera,
        down_dims=cfg.policy.down_dims,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        use_film_scale_modulation=cfg.policy.use_film_scale_modulation,
        noise_scheduler_type=cfg.policy.noise_scheduler_type,
        num_train_timesteps=cfg.policy.num_train_timesteps,
        beta_schedule=cfg.policy.beta_schedule,
        beta_start=cfg.policy.beta_start,
        beta_end=cfg.policy.beta_end,
        prediction_type=cfg.policy.prediction_type,
        clip_sample=cfg.policy.clip_sample,
        clip_sample_range=cfg.policy.clip_sample_range,
        num_inference_steps=cfg.policy.num_inference_steps,
        do_mask_loss_for_padding=cfg.policy.do_mask_loss_for_padding,
        optimizer_lr=cfg.training.lr,
        optimizer_betas=cfg.training.betas,
        optimizer_eps=cfg.training.eps,
        optimizer_weight_decay=cfg.training.weight_decay,
        scheduler_name=cfg.training.scheduler,
        scheduler_warmup_steps=cfg.training.warmup_steps,
        device=cfg.device,
    )


def build_policy(cfg: TrainConfig, ds_meta):
    """Build DiffusionPolicy with features inferred from the dataset."""
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import FeatureType

    policy_cfg = _build_diffusion_config(cfg)

    # Map dataset features → PolicyFeature objects
    features = dataset_to_policy_features(ds_meta.features)
    policy_cfg.output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    policy_cfg.input_features = {
        k: v for k, v in features.items() if k not in policy_cfg.output_features
    }

    logger.info("Input features:  %s", {
        k: (v.type.name, v.shape) for k, v in policy_cfg.input_features.items()
    })
    logger.info("Output features: %s", {
        k: (v.type.name, v.shape) for k, v in policy_cfg.output_features.items()
    })

    # Build policy
    policy = DiffusionPolicy(config=policy_cfg)
    policy.to(cfg.device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("Policy built: %.2fM params (%.2fM trainable)", n_params / 1e6, n_train / 1e6)

    return policy


def build_preprocessor(policy, dataset):
    """Build pre/post-processor pipelines for normalization."""
    from lerobot.policies.diffusion.processor_diffusion import (
        make_diffusion_pre_post_processors,
    )

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset.meta.stats,
    )
    logger.info("Pre/post-processors built")
    return preprocessor, postprocessor


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer & Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(policy, cfg: TrainConfig):
    """Build AdamW optimizer."""
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg.training.lr,
        betas=cfg.training.betas,
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )
    logger.info(
        "Optimizer: AdamW (lr=%.1e, betas=%s, wd=%.1e)",
        cfg.training.lr, cfg.training.betas, cfg.training.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, total_steps: int, cfg: TrainConfig):
    """Build LR scheduler (cosine with warmup)."""
    from diffusers.optimization import get_scheduler

    scheduler = get_scheduler(
        name=cfg.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Scheduler: %s (warmup=%d, total=%d)",
        cfg.training.scheduler, cfg.training.warmup_steps, total_steps,
    )
    return scheduler


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    output_dir: Path,
    step: int,
    total_steps: int,
    policy,
    optimizer,
    scheduler,
    preprocessor,
    postprocessor,
    cfg: TrainConfig,
    is_last: bool = False,
) -> Path:
    """Save a checkpoint compatible with DiffusionPolicy.from_pretrained().

    Directory layout:
        checkpoints/<step>/
        ├── pretrained_model/
        │   ├── config.json          # DiffusionConfig
        │   ├── model.safetensors    # weights
        │   ├── policy_preprocessor.json
        │   └── policy_postprocessor.json
        └── training_state/
            ├── optimizer.pt
            ├── scheduler.pt
            └── step.json
    """
    step_str = f"{step:0{len(str(total_steps))}d}"
    ckpt_dir = output_dir / "checkpoints" / step_str
    pretrained_dir = ckpt_dir / "pretrained_model"
    training_dir = ckpt_dir / "training_state"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Save policy (config.json + model.safetensors)
    policy.save_pretrained(pretrained_dir)

    # Save pre/post-processors
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    # Save training state
    torch.save(optimizer.state_dict(), training_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), training_dir / "scheduler.pt")
    with open(training_dir / "step.json", "w") as f:
        json.dump({"step": step, "total_steps": total_steps}, f)

    # Save our training config (for reference/reproducibility)
    with open(pretrained_dir / "train_config.yaml", "w") as f:
        yaml.dump(_config_to_dict(cfg), f, default_flow_style=False, sort_keys=False)

    # Update "last" symlink
    if is_last or True:  # always update last
        last_link = output_dir / "checkpoints" / "last"
        if last_link.is_symlink():
            last_link.unlink()
        last_link.symlink_to(step_str)

    logger.info("Checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def load_checkpoint_for_resume(
    ckpt_dir: Path,
    policy,
    optimizer,
    scheduler,
    device: str,
) -> int:
    """Load training state from a checkpoint. Returns the step to resume from."""
    training_dir = ckpt_dir / "training_state"

    # Load optimizer
    opt_path = training_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
        logger.info("Optimizer state loaded from %s", opt_path)

    # Load scheduler
    sched_path = training_dir / "scheduler.pt"
    if sched_path.exists() and scheduler is not None:
        scheduler.load_state_dict(torch.load(sched_path, map_location=device, weights_only=True))
        logger.info("Scheduler state loaded from %s", sched_path)

    # Load step
    step_path = training_dir / "step.json"
    if step_path.exists():
        with open(step_path) as f:
            step_info = json.load(f)
        start_step = step_info["step"]
        logger.info("Resuming from step %d", start_step)
        return start_step

    return 0


def _config_to_dict(cfg: TrainConfig) -> dict:
    """Serialize TrainConfig to a plain dict for YAML."""
    import dataclasses
    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            result = {}
            for fld in dataclasses.fields(obj):
                val = getattr(obj, fld.name)
                result[fld.name] = _to_dict(val)
            return result
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj
    return _to_dict(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Seeding
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig) -> None:
    """Main training function."""

    # ── Setup ────────────────────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    set_seed(cfg.training.seed)

    # Output directory
    if cfg.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_name = cfg.dataset.repo_id.replace("/", "_")
        cfg.output_dir = str(
            Path(_PROJECT_ROOT) / "outputs" / "train" / f"{timestamp}_dp_{task_name}"
        )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(_config_to_dict(cfg), f, default_flow_style=False, sort_keys=False)
    logger.info("Output directory: %s", output_dir)

    # WandB
    if cfg.logging.wandb_enable:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_name or output_dir.name,
            config=_config_to_dict(cfg),
            dir=str(output_dir),
        )

    # ── Build components ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building dataset...")
    dataset, ds_meta = build_dataset(cfg)

    logger.info("Building policy...")
    policy = build_policy(cfg, ds_meta)

    logger.info("Building preprocessor...")
    preprocessor, postprocessor = build_preprocessor(policy, dataset)

    # ── Compute total training steps ─────────────────────────────────────────
    dataloader = build_dataloader(dataset, cfg)
    steps_per_epoch = len(dataloader)

    if cfg.training.steps is not None:
        total_steps = cfg.training.steps
        total_epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        total_epochs = cfg.training.epochs
        total_steps = total_epochs * steps_per_epoch

    logger.info(
        "Training plan: %d steps (%d epochs × %d steps/epoch), batch_size=%d",
        total_steps, total_epochs, steps_per_epoch, cfg.training.batch_size,
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = build_optimizer(policy, cfg)
    scheduler = build_scheduler(optimizer, total_steps, cfg)

    # ── Resume (optional) ────────────────────────────────────────────────────
    start_step = 0
    resume_dir = getattr(cfg, "_resume_dir", None)
    if resume_dir is not None:
        resume_path = Path(resume_dir)
        # Load policy weights from pretrained_model/
        pretrained_path = resume_path / "pretrained_model"
        if pretrained_path.exists():
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from safetensors.torch import load_model as load_model_safetensors
            load_model_safetensors(policy, str(pretrained_path / "model.safetensors"))
            logger.info("Policy weights loaded from %s", pretrained_path)

        start_step = load_checkpoint_for_resume(
            resume_path, policy, optimizer, scheduler, cfg.device,
        )

    # ── AMP scaler ───────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_amp)

    # ── Infinite dataloader ──────────────────────────────────────────────────
    from lerobot.datasets.utils import cycle
    dl_iter = cycle(dataloader)

    # ── Print summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Diffusion Policy Training for A1X")
    logger.info("=" * 60)
    logger.info("  Noise scheduler : %s", cfg.policy.noise_scheduler_type)
    logger.info("  Train timesteps : %d", cfg.policy.num_train_timesteps)
    logger.info("  Infer timesteps : %s", cfg.policy.num_inference_steps)
    logger.info("  Horizon         : %d", cfg.policy.horizon)
    logger.info("  Obs steps       : %d", cfg.policy.n_obs_steps)
    logger.info("  Action steps    : %d", cfg.policy.n_action_steps)
    logger.info("  Down dims       : %s", cfg.policy.down_dims)
    logger.info("  Resize shape    : %s", cfg.policy.resize_shape)
    logger.info("  Batch size      : %d", cfg.training.batch_size)
    logger.info("  LR              : %.1e", cfg.training.lr)
    logger.info("  Total steps     : %d", total_steps)
    logger.info("  Start step      : %d", start_step)
    logger.info("  Device          : %s (GPU %d)", cfg.device, cfg.gpu_id)
    logger.info("=" * 60)

    # ── Training loop ────────────────────────────────────────────────────────
    policy.train()
    running_loss = 0.0
    log_count = 0
    t0 = time.time()

    pbar = tqdm(
        range(start_step + 1, total_steps + 1),
        initial=start_step,
        total=total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    for step in pbar:
        # 1. Get batch
        batch = next(dl_iter)

        # 2. Preprocess (normalize + move to device)
        batch = preprocessor(batch)

        # 3. Forward pass
        with torch.amp.autocast("cuda", enabled=cfg.training.use_amp):
            loss, _ = policy.forward(batch)

        # 4. Backward pass
        scaler.scale(loss).backward()

        # 5. Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=cfg.training.grad_clip_norm,
        )

        # 6. Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 7. LR scheduler step (per iteration, not per epoch)
        scheduler.step()

        # ── Progress bar update ─────────────────────────────────────────
        loss_val = loss.item()
        running_loss += loss_val
        log_count += 1
        current_lr = scheduler.get_last_lr()[0]
        epoch = step / steps_per_epoch
        grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            epoch=f"{epoch:.1f}",
            lr=f"{current_lr:.1e}",
            grad=f"{grad_norm_val:.1f}",
        )

        # ── WandB logging (every step) ──────────────────────────────────
        if cfg.logging.wandb_enable:
            import wandb
            wandb.log({
                "train/loss": loss_val,
                "train/lr": current_lr,
                "train/grad_norm": grad_norm_val,
                "train/epoch": epoch,
            }, step=step)

        # ── Console logging (periodic) ──────────────────────────────────
        if step % cfg.logging.log_freq == 0:
            avg_loss = running_loss / log_count
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed if elapsed > 0 else 0

            log_msg = (
                f"step {step:>{len(str(total_steps))}d}/{total_steps} "
                f"| epoch {epoch:.1f} "
                f"| loss {avg_loss:.4f} "
                f"| lr {current_lr:.2e} "
                f"| grad_norm {grad_norm_val:.2f} "
                f"| {steps_per_sec:.1f} steps/s"
            )
            # Write to log file without disturbing tqdm bar
            logger.info(log_msg)

            running_loss = 0.0
            log_count = 0

        # ── Checkpointing ───────────────────────────────────────────────
        if step % cfg.logging.save_freq == 0 or step == total_steps:
            save_checkpoint(
                output_dir, step, total_steps,
                policy, optimizer, scheduler,
                preprocessor, postprocessor, cfg,
            )

    # ── Done ─────────────────────────────────────────────────────────────────
    pbar.close()
    total_time = time.time() - t0
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Total time     : %.1f min", total_time / 60)
    logger.info("  Final loss     : %.4f", loss_val)
    logger.info("  Checkpoints at : %s", output_dir / "checkpoints")
    logger.info("")
    logger.info("To deploy this model, use:")
    logger.info(
        "  python imitation_learning/DiffusionPolicy/deploy.py "
        "--checkpoint %s",
        output_dir / "checkpoints" / "last" / "pretrained_model",
    )
    logger.info("=" * 60)

    if cfg.logging.wandb_enable:
        import wandb
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy for A1X Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Train with default config
    python imitation_learning/DiffusionPolicy/train.py

    # Train with custom learning rate and batch size
    python imitation_learning/DiffusionPolicy/train.py --lr 5e-5 --batch_size 32

    # Train for a specific number of steps
    python imitation_learning/DiffusionPolicy/train.py --steps 50000

    # Train with different dataset
    python imitation_learning/DiffusionPolicy/train.py \\
        --dataset a1x/my_other_task \\
        --dataset_root ~/.cache/huggingface/lerobot/a1x/my_other_task

    # Resume from checkpoint
    python imitation_learning/DiffusionPolicy/train.py \\
        --resume outputs/train/.../checkpoints/last

    # Enable WandB logging
    python imitation_learning/DiffusionPolicy/train.py --wandb
""",
    )

    # Config file
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config.yaml"),
        help="Path to YAML config file (default: config.yaml in script dir)",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume training from",
    )

    # Quick overrides (most commonly tuned knobs)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--steps", type=int, default=None, help="Total steps (overrides epochs)")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Policy overrides
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon")
    parser.add_argument("--n_obs_steps", type=int, default=None, help="Observation steps")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Action steps")
    parser.add_argument("--vision_backbone", type=str, default=None, help="Vision backbone")
    parser.add_argument("--noise_scheduler_type", type=str, default=None, help="DDPM or DDIM")
    parser.add_argument("--num_train_timesteps", type=int, default=None, help="Training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Inference timesteps")

    # Dataset overrides
    parser.add_argument("--dataset", type=str, default=None, help="Dataset repo_id")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root path")

    # Logging overrides
    parser.add_argument("--log_freq", type=int, default=None, help="Log frequency")
    parser.add_argument("--save_freq", type=int, default=None, help="Save frequency")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable WandB")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")

    # Output/hardware
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU device ID")
    parser.add_argument("--grad_clip_norm", type=float, default=None, help="Gradient clip norm")
    parser.add_argument("--warmup_steps", type=int, default=None, help="LR warmup steps")
    parser.add_argument("--use_amp", action="store_true", default=None, help="Use mixed precision")

    return parser.parse_args()


def main():
    args = parse_args()

    # Collect CLI overrides (non-None values)
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ("config", "resume")
    }

    # Load config
    cfg = load_config(args.config, cli_overrides)

    # Handle resume
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            logger.error("Resume path not found: %s", resume_path)
            sys.exit(1)
        cfg._resume_dir = str(resume_path)
        logger.info("Will resume from: %s", resume_path)

    # Run training
    train(cfg)


if __name__ == "__main__":
    main()
