"""Per-star NLE config for the 9-param AAU dataset — full 75-file run.

Successor to `nsf_9params.py`, which trained on 20 files with a 144k-param
flow. Inference on that model (see `nle_inference.ipynb`) gave posteriors
that were tight but biased by several sigma, and emcee/pocomc disagreed at
the same level -- the signature of a likelihood surface that is smooth
enough to sample but not accurate enough to sum over 200 stars. This config
attacks that with 3.75x the data and ~10x the flow capacity.

Changes vs `nsf_9params.py`:
  * num_datasets 20 -> 75 (data.0..74; data.75..77 stay held out)
  * hidden_features [64, 64] -> [256, 256], transforms 10 -> 12, bins 8 -> 12
  * train_batch_size 1024 -> 8192, lr held at 5e-4
  * train_frac 0.9 -> 0.95, num_epochs 40 -> 80

The batch-size jump is the cheap one: at 1024 the step is dominated by
kernel-launch overhead (34k stars/s), and 8192 costs the same per step
(~250k stars/s), so the bigger flow is close to free.

lr stays at the 5e-4 that trained the 20-file run: batch-size scaling rules
would allow more (4e-3 linear, 1.4e-3 sqrt), but spline coupling parameters
-- bin widths, heights, derivatives -- are the brittle part of an NSF and
scaling rules say nothing about them. The 8x larger batch therefore buys
stability rather than speed, and num_epochs is raised to 80 to make up the
optimizer steps it costs.
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
    # 75 of the 78 files: 750k streams, 150M stars (~35 GB resident).
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
    # 5% of 750k streams is still 37.5k streams of validation.
    config.train_frac = 0.95
    config.num_workers = 0

    ### LOGGING AND WANDB CONFIGURATION ###
    # Shared root across runs; WandbLogger nests <wandb_project>/<run_id>.
    config.workdir = '/scratch/tvnguyen/trained_models/nle'
    config.wandb_project = '9p_AAU_nle'
    config.entity = 'desc_sbi_stream'
    config.name = None
    config.id = 'fux1gqy9'
    config.tags = ['nle', 'nsf', '75ds']
    config.debug = False
    config.checkpoint = 'last.ckpt'  # Path to an NLE checkpoint to resume from
    config.reset_optimizer = False
    config.enable_progress_bar = True

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()

    # Conditional normalizing flow p(x | theta, cond, xerr).
    model.flows = ConfigDict()
    model.flows.flow_type = 'spline'
    model.flows.num_transforms = 12
    model.flows.hidden_features = [256, 256]
    model.flows.num_bins = 12
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
    # Unchanged from nsf_9params.py: the inference notebooks and the NPE
    # configs assume this exact uncertainty model.
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
    # Unchanged from the 20-file run; see the module docstring for why the
    # larger batch does not come with a larger lr.
    optimizer.lr = 5e-4
    optimizer.weight_decay = 1e-4

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    # Stepped per epoch, so these are epochs; decay_steps == num_epochs so
    # the cosine finishes exactly at the end of training. eta_min is a
    # multiplier on the base lr, not an absolute floor (see utils.py):
    # 1e-3 -> a final lr of 1e-6.
    scheduler.decay_steps = 80
    scheduler.warmup_steps = 3
    scheduler.eta_min = 1e-3
    scheduler.interval = 'epoch'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    # 17.4k steps/epoch, ~10 min/epoch on one H100 -> ~15 h for 80 epochs,
    # inside the compute partition's 24 h limit with margin.
    config.train_batch_size = 8192
    config.eval_batch_size = 16384
    config.num_epochs = 80
    config.num_steps = -1
    config.patience = 20
    config.gradient_clip_val = 1.0
    config.save_top_k = 5

    return config
