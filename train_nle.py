"""Training script for Neural Likelihood Estimation (NLE)."""

import getpass
import json
import os
import shutil
import sys
from pathlib import Path

# $HOME is read-only on this cluster's compute nodes, so every wandb
# directory that otherwise defaults under it (artifact staging, artifact
# cache, downloaded artifacts) is redirected to $SCRATCH before wandb is
# imported. Set these in the environment to override.
_SCRATCH = os.environ.get('SCRATCH', f'/scratch/{getpass.getuser()}')
os.environ.setdefault('WANDB_DATA_DIR', f'{_SCRATCH}/wandb_data')
os.environ.setdefault('WANDB_CACHE_DIR', f'{_SCRATCH}/cache/wandb')
os.environ.setdefault('WANDB_ARTIFACT_DIR', f'{_SCRATCH}/wandb_artifacts')
os.environ.setdefault('MPLCONFIGDIR', f'{_SCRATCH}/cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', f'{_SCRATCH}/cache')

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
from nle_stream_subhalo.transforms import (
    StreamTrack, build_transformation, load_track_dict)


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

    # A parameterized config arrives as `path.py:variant`; only the path
    # part is a file. The variant is already captured in the JSON above.
    if config_path:
        config_path = config_path.split(':')[0]
    if config_path and os.path.exists(config_path):
        shutil.copy2(config_path, snapshot_dir / 'config_snapshot.py')


def load_track(config: ml_collections.ConfigDict, snapshot_dir: Path = None):
    """Load the unperturbed stream track, if this run uses one.

    Fitted separately by `stream_sims.track` in the sbi_stream_pipeline
    repo -- it needs the simulator, and it depends on nothing that varies
    between training examples. A copy is dropped next to the checkpoints
    so a run stays reproducible from its own directory.

    Returns:
        (track_dict, track) or (None, None) when `config.track_path` is
        unset. `track_dict` goes to the model; `track` is the built
        module, needed to measure the normalization on projected stars.
    """
    path = config.get('track_path', None)
    if path is None:
        print('[Track] No track_path: training on raw coordinates')
        return None, None

    track_dict = load_track_dict(path)
    track = StreamTrack(track_dict, config.x_labels)
    print(f'[Track] Loaded {path}')
    print(f'[Track] {len(track.coords)} coordinates '
          f"({', '.join(track.coords)}), {track.knots.numel()} knots, "
          f'phi1 in [{track.knots[0]:.1f}, {track.knots[-1]:.1f}]')
    print(f"[Track] source stream sha256: "
          f"{track.meta.get('stream_sha256', 'unknown')}")

    if snapshot_dir is not None:
        shutil.copy2(path, snapshot_dir / 'track_snapshot.npz')

    return track_dict, track


def prepare_data(
    config: ml_collections.ConfigDict, norm_dict=None, track=None,
    track_dict=None):
    """Load data and build per-star train/val dataloaders.

    Each star is one row of `config.x_labels` (stream-frame observables);
    the per-stream `config.labels` are broadcast to its member stars.

    Returns (train_loader, val_loader, norm_dict). When `norm_dict` is given
    (e.g. reused from a resumed checkpoint) it is passed through unchanged
    instead of recomputing field stats from this call's training data.
    `track` is only used when the stats are recomputed, so that they
    describe the offsets the model actually normalizes. With
    `config.model.x_stages` set, the whole `x` transform is fitted as a
    stage chain instead and `track_dict` becomes the track stage's state.
    `config.norm_max_stars` caps how many training stars that measurement
    reads; unset means all of them, which is only affordable on the
    smaller subsets.
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
        track=track,
        norm_kwargs=_norm_kwargs(config, track_dict),
        norm_max_stars=config.get('norm_max_stars', None),
    )

    return train_loader, val_loader, norm_dict


def _norm_kwargs(config: ml_collections.ConfigDict, track_dict=None) -> dict:
    """Stage-chain arguments for `compute_field_norm`, if this run uses one.

    Empty dict means the legacy single affine `x` scaling.
    """
    stages = config.model.get('x_stages', None)
    if stages is None:
        return {}
    stages = list(stages)
    print(f"[Stages] Fitting x chain: {' -> '.join(stages)}")
    return dict(x_stages=stages, track_dict=track_dict,
                x_labels=list(config.x_labels),
                n_knots=config.model.get('n_knots', 256))


def create_model(
    config: ml_collections.ConfigDict, pre_transforms, norm_dict,
    track_dict=None,
) -> NLE:
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
        val_nll_batches=config.model.get('val_nll_batches', 8),
        track_dict=track_dict,
        x_labels=config.x_labels,
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
    resume_track_dict = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, project_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

        # Reuse the norm_dict from the resumed checkpoint's hyper_parameters.
        resume_checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        norm_dict = resume_checkpoint['hyper_parameters']['norm_dict']
        print("[Checkpoint] Reusing norm_dict from resumed checkpoint")

        # And its track: the weights were fitted against that one, so a
        # config now pointing elsewhere would silently change what every
        # input means.
        resume_track_dict = resume_checkpoint['hyper_parameters'].get(
            'track_dict')

    print("[Track] Loading stream track...")
    track_dict, track = load_track(config, project_dir)
    if resume_training and resume_track_dict != track_dict:
        sha = lambda d: (d or {}).get('meta', {}).get('stream_sha256')
        raise ValueError(
            'the resumed checkpoint was trained with a different track '
            f'(stream_sha256 {sha(resume_track_dict)} vs '
            f'{sha(track_dict)}). Point config.track_path at the one it '
            'was trained with, or start a fresh run.')

    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(
        config, norm_dict=norm_dict, track=track, track_dict=track_dict)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(**config.pre_transforms.to_dict())

    print("[Model] Creating NLE model...")
    model = create_model(config, pre_transforms, norm_dict, track_dict)
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

    training.report_offline_sync(wandb_logger)
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
