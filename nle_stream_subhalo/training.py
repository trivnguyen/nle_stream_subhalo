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
    (e.g. HPC compute nodes). The timeout bounds the TCP connect, not the
    DNS lookup ahead of it -- `resolve_wandb_mode` avoids calling this at
    all under SLURM for that reason.
    """
    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def resolve_wandb_mode(config: ml_collections.ConfigDict) -> str:
    """Pick the wandb mode for this run.

    In precedence order: `config.debug` disables logging entirely; an
    explicit `config.wandb_mode` wins next; then a `WANDB_MODE` set in the
    environment; then SLURM, where compute nodes have no outbound network,
    so 'offline' is assumed without probing; otherwise the API is probed
    and the run goes 'online' only if it answers.

    Offline runs are complete on disk and are uploaded afterwards from a
    login node with `wandb sync` (see `report_offline_sync`).
    """
    if config.get('debug', False):
        return 'disabled'

    mode = config.get('wandb_mode', None)
    if mode is not None:
        return mode

    mode = os.environ.get('WANDB_MODE')
    if mode:
        print(f"[WandB] Using WANDB_MODE={mode} from the environment")
        return mode

    if os.environ.get('SLURM_JOB_ID'):
        print("[WandB] SLURM job detected; assuming no outbound network")
        return 'offline'

    return 'online' if _wandb_server_reachable() else 'offline'


def create_wandb_logger(config: ml_collections.ConfigDict, tag: str):
    """Create a WandbLogger for the run, tagged with `tag`.

    Returns the logger and the project directory (where checkpoints and the
    config snapshot are saved). The mode comes from `resolve_wandb_mode`.
    `config.log_model` controls checkpoint artifacts ('all' by default);
    offline they are staged under $WANDB_DATA_DIR and uploaded by the later
    `wandb sync`, which costs one extra on-disk copy per checkpoint.
    """
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    wandb_mode = resolve_wandb_mode(config)
    print(f"[WandB] Mode: {wandb_mode}")

    tags = set(config.get('tags', [])) | {tag}
    logger = WandbLogger(
        project=config.get("wandb_project", "nle_stream"),
        name=config.get("name"),
        entity=config.get("entity", None),
        id=config.get("id", None),
        save_dir=str(workdir),
        log_model=config.get("log_model", "all"),
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
        tags=list(tags),
    )
    project_dir = workdir / logger.experiment.project / logger.experiment.id
    project_dir.mkdir(parents=True, exist_ok=True)

    print(logger, project_dir)
    return logger, project_dir


def report_offline_sync(logger: WandbLogger) -> None:
    """Print the `wandb sync` command for an offline run, if it is one.

    Call before `wandb.finish()` while the run directory is still known.
    Nothing is printed for online/disabled runs.
    """
    run = getattr(logger, 'experiment', None)
    run_dir = getattr(run, 'dir', None)
    # `debug` runs get a NoopRun, which reports offline=False.
    if run_dir is None or not getattr(run, 'offline', False):
        return

    # run.dir is the run's `files/` subdir; `wandb sync` wants its parent.
    sync_dir = Path(run_dir).parent
    print(f"[WandB] Offline run saved to: {sync_dir}")
    print(f"[WandB] Upload it from a login node with: wandb sync {sync_dir}")


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
