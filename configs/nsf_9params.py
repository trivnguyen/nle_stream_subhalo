"""Per-star NLE config for the 9-param AAU perturber dataset.

Trains p(star observables | perturber theta, xerr) on the same HDF5
datasets as the NPE pipeline's `configs/chebconv_9params.py`, one training
example per stream particle instead of one per stream.
"""

import numpy as np
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 142
    config.seed_training = np.random.randint(0, 1_000_000)

    ### DATA CONFIGURATION ###
    config.data_root = '/scratch/tvnguyen/stream_datasets'
    config.data_name = '9p_AAU'
    config.num_datasets = 75
    config.init = 0
    config.x_labels = ('phi1', 'phi2', 'vr', 'pm1', 'pm2', 'dist')
    config.labels = (
        'log_mass', 'log_radius', 'v_rel_perp', 'v_rel_para',
        'angle_pos_impact', 'angle_vel_delta', 'impact_param', 'time_impact',
        'phi1_impact_today',
    )
    # Measured conditioning variables (cond), fixed or MC-sampled at
    # inference. None: theta and xerr are the only context.
    config.cond_labels = None
    config.train_frac = 0.9
    config.num_workers = 0

    ### LOGGING AND WANDB CONFIGURATION ###
    # Shared root across runs; WandbLogger nests <wandb_project>/<run_id>.
    config.workdir = '/scratch/tvnguyen/trained_models/nle'
    config.wandb_project = '9p_AAU_nle'
    config.entity = 'desc_sbi_stream'
    config.name = None
    config.id = None
    config.tags = ['nle', 'nsf']
    config.debug = False
    config.checkpoint = None  # Path to an NLE checkpoint to resume from
    config.reset_optimizer = False
    config.enable_progress_bar = True

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()

    # Conditional normalizing flow p(x | theta, cond, xerr).
    model.flows = ConfigDict()
    model.flows.flow_type = 'spline'
    model.flows.num_transforms = 10
    model.flows.hidden_features = [64, 64]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'

    # Optional single MLP over the combined context before the flow.
    # model.context_embedding = ConfigDict()
    # model.context_embedding.hidden_sizes = [128, 128]
    # model.context_embedding.output_size = 64
    # model.context_embedding.act_name = 'relu'
    # model.context_embedding.batch_norm = False
    # model.context_embedding.dropout = 0.0

    ### OBSERVATION PIPELINE (pre_transforms) ###
    # One entry per feature with a measurement uncertainty; each adds a
    # column to `xerr` (in this order), which the flow conditions on.
    # Matches the NPE config's uncertainty model.
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_uncertainty = True
    pre_transforms.uncertainty_args = [
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.01),
             width_range=(5.0, 20.0), feature_idx=2),   # vr, km/s
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.01),
             width_range=(0.2, 0.5), feature_idx=3),    # pm1, mas/yr
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.01),
             width_range=(0.2, 0.5), feature_idx=4),    # pm2, mas/yr
        dict(distribution_type='jeffreys_varied', low_range=(0.1, 0.1),
             width_range=(2.0, 5.0), feature_idx=5),    # dist, kpc
    ]

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 5e-4
    optimizer.weight_decay = 2e-3

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 40
    scheduler.warmup_steps = 2
    scheduler.eta_min = 1e-6
    scheduler.interval = 'epoch'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 256
    config.eval_batch_size = 256
    config.num_epochs = 40
    config.num_steps = -1
    config.patience = 20
    config.gradient_clip_val = 1.0
    config.save_top_k = 5

    return config
