"""Training script for Neural Likelihood Estimation (NLE)."""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('WANDB_DATA_DIR', '/scratch/tvnguyen/wandb_data')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
import torch
from absl import flags
from ml_collections import config_flags

from nle_stream_subhalo import datasets, training
from nle_stream_subhalo.nle import NLE
from nle_stream_subhalo.transforms import build_transformation


def save_config_snapshot(
    config: ml_collections.ConfigDict, snapshot_dir: Path, config_path: str = None,
) -> None:
    """Write a JSON (and .py) config snapshot next to the checkpoints.

    Lets downstream consumers reconstruct the model / pre_transforms from
    the checkpoint directory alone, without wandb.
    """
    snapshot_path = snapshot_dir / 'config_snapshot.json'
    with open(snapshot_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"[Setup] Wrote config snapshot -> {snapshot_path}")

    if config_path and os.path.exists(config_path):
        shutil.copy2(config_path, snapshot_dir / 'config_snapshot.py')


def prepare_data(config: ml_collections.ConfigDict, norm_dict=None):
    """Load data and build per-star train/val dataloaders.

    Each star is one row of `config.x_labels` (stream-frame observables);
    the per-stream `config.labels` are broadcast to its member stars.

    Returns (train_loader, val_loader, norm_dict). When `norm_dict` is given
    (e.g. reused from a resumed checkpoint) it is passed through unchanged
    instead of recomputing field stats from this call's training data.
    """
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        is_directory=True,
        concat=True,
    )

    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.x_labels,
        config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict,
        pre_transform_kwargs=config.pre_transforms.to_dict(),
    )

    return train_loader, val_loader, norm_dict


def create_model(config: ml_collections.ConfigDict, pre_transforms, norm_dict) -> NLE:
    """Create the NLE model from the config."""
    print("[Model] Creating NLE model...")
    context_embedding = config.model.get('context_embedding', None)
    if context_embedding is not None:
        context_embedding = context_embedding.to_dict()

    return NLE(
        norm_dict=norm_dict,
        flows_args=config.model.flows.to_dict(),
        context_embedding_args=context_embedding,
        optimizer_args=config.optimizer.to_dict(),
        scheduler_args=config.scheduler.to_dict(),
        pre_transforms=pre_transforms,
    )


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create the training callbacks."""
    return training.create_base_callbacks(config)


def main(config: ml_collections.ConfigDict, config_path: str = None):
    """Train the NLE model with wandb logging."""
    resume_training = config.get('checkpoint') is not None
    wandb_logger, project_dir = training.create_wandb_logger(config, tag='nle')

    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Project directory: {project_dir}")

    print(f"[Setup] Saving config snapshot to: {project_dir}")
    save_config_snapshot(config, project_dir, config_path)

    checkpoint_path = None
    norm_dict = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, project_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

        # Reuse the norm_dict from the resumed checkpoint's hyper_parameters.
        resume_checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        norm_dict = resume_checkpoint['hyper_parameters']['norm_dict']
        print("[Checkpoint] Reusing norm_dict from resumed checkpoint")

    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(config, norm_dict=norm_dict)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(**config.pre_transforms.to_dict())

    print("[Model] Creating NLE model...")
    model = create_model(config, pre_transforms, norm_dict)
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    wandb_logger.watch(model, log="all", log_freq=1000, log_graph=False)

    callbacks = create_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    trainer = training.build_trainer(
        config, project_dir, callbacks, wandb_logger, num_sanity_val_steps=0)

    pl.seed_everything(config.seed_training, workers=True)
    print(f"[Seed] Training seed set to: {config.seed_training}")

    training.fit(
        trainer, model, train_loader, val_loader,
        checkpoint_path=checkpoint_path,
        reset_optimizer=config.get('reset_optimizer', False),
    )

    wandb.finish()
    print("[WandB] Finished")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(
        config=FLAGS.config,
        config_path=config_flags.get_config_filename(FLAGS['config']),
    )
