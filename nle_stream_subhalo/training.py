"""Shared helpers for the NLE training script (setup, callbacks, fit loop).

Adapted from the jgnn package to keep the wandb/checkpoint/trainer
boilerplate identical to train_npe.
"""

import os
import socket
from pathlib import Path
from typing import Optional

import torch
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


def _wandb_server_reachable(timeout: float = 3.0) -> bool:
    """Check whether the WandB API is reachable, with a short timeout.

    Lets us pick 'online' vs 'offline' up front, so training doesn't stall
    behind wandb's own connection retries on machines without internet
    (e.g. HPC compute nodes).
    """
    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def create_wandb_logger(config: ml_collections.ConfigDict, tag: str):
    """Create a WandbLogger for the run, tagged with `tag`.

    Returns the logger and the project directory (where checkpoints and the
    config snapshot are saved). If `config.wandb_mode` is unset, the mode is
    'online' when the WandB API is reachable, else 'offline'.
    """
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if config.get('debug', False):
        wandb_mode = 'disabled'
    else:
        wandb_mode = config.get('wandb_mode', None)
        if wandb_mode is None:
            wandb_mode = 'online' if _wandb_server_reachable() else 'offline'
    print(f"[WandB] Mode: {wandb_mode}")

    tags = set(config.get('tags', [])) | {tag}
    logger = WandbLogger(
        project=config.get("wandb_project", "nle_stream"),
        name=config.get("name"),
        entity=config.get("entity", None),
        id=config.get("id", None),
        save_dir=str(workdir),
        log_model="all",
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
        tags=list(tags),
    )
    project_dir = workdir / logger.experiment.project / logger.experiment.id
    project_dir.mkdir(parents=True, exist_ok=True)

    print(logger, project_dir)
    return logger, project_dir


def get_checkpoint_path(
    config: ml_collections.ConfigDict, project_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the checkpoint path to resume from, if any."""
    if config.get('checkpoint') is None:
        return None

    ckpt = config.checkpoint
    if os.path.isabs(ckpt):
        return ckpt

    if project_dir is None:
        raise ValueError(
            "If `config.checkpoint` is a relative path, `project_dir` must be"
            " provided to resolve the full path.")
    return str(project_dir / 'checkpoints' / ckpt)


def create_base_callbacks(config: ml_collections.ConfigDict) -> list:
    """Standard early-stopping / checkpointing / LR-monitor callbacks."""
    return [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True,
            check_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-loss={val/loss:.4f}",
            monitor='val/loss',
            mode='min',
            save_top_k=config.get('save_top_k', 3),
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="last",
            save_weights_only=False,
            save_last=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def report_param_counts(model) -> None:
    """Print total, trainable, and frozen parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total:,}")
    print(f"[Model] Trainable parameters: {trainable:,}")
    print(f"[Model] Frozen parameters: {total - trainable:,}")


def build_trainer(
    config: ml_collections.ConfigDict,
    project_dir: Path,
    callbacks: list,
    wandb_logger: WandbLogger,
    **trainer_kwargs,
) -> pl.Trainer:
    """Construct the PyTorch Lightning Trainer."""
    print(f"[Trainer] Max epochs: {config.num_epochs}, Max steps: {config.num_steps}")
    print(f"[Trainer] Accelerator: {config.accelerator}")

    return pl.Trainer(
        default_root_dir=str(project_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
        **trainer_kwargs,
    )


def fit(
    trainer: pl.Trainer,
    model,
    train_loader,
    val_loader,
    checkpoint_path: Optional[str],
    reset_optimizer: bool,
) -> None:
    """Run `trainer.fit`, handling checkpoint resume and weights-only reset."""
    if checkpoint_path and reset_optimizer:
        print("[Training] Loading model weights with fresh optimizer state")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    elif checkpoint_path:
        print("[Training] Resuming training from full checkpoint")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        print("[Training] Starting fresh training")
        trainer.fit(model, train_loader, val_loader)

    print("[Training] Training complete!")
