"""Per-star NLE config for the 9-param AAU dataset — flow matching.

Same data, observation pipeline and step budget as `nsf_9params_75ds.py`;
the flow is the only thing that changes. Instead of 12 stacked spline
transforms trained by maximum likelihood, one MLP velocity field is
trained by conditional flow matching, and the density comes from a
fixed-step ODE solve (`flow_matching.FlowMatching`).

What that costs and buys, measured on this dataset at 6 target dimensions:

  * `log_prob` is ~3x an NSF evaluation -- 32 velocity evaluations plus an
    exact 6-wide Jacobian trace, against the NSF's 12 conditioner passes.
    A vectorized pocomc call (4096 particles x 200 stars) goes from ~1.4 s
    to ~4.5 s. Everything downstream is likelihood-bound, so budget ~3x
    for a posterior run.
  * Sampling is ~10x *faster*: an ODE solve without the trace, against the
    NSF's one autoregressive pass per target dimension. The predictive
    checks in the inference notebooks get cheaper.
  * Training costs about the same per step as the NSF: the loss is a
    regression on the velocity field and never solves an ODE.

Capacity is the part that does not transfer. A spline transform spends
its parameters on bin edges, so an NSF gets a lot of expressiveness per
parameter; the velocity field has to represent the same density with a
plain MLP and needs to be wide to compete -- see the module-level note in
`flow_matching.py` and the sweep behind `hidden_features` below.

Two knobs have no analogue in the discrete configs:

  * `steps` -- midpoint steps per density solve. 16 matched zuko's own
    adaptive solver to ~2e-4 nats per star on this problem, i.e. ~0.04
    nats on a 200-star stream. Re-check it on the trained model before
    trusting a posterior: `flow(context).log_prob(target)` is the
    adaptive reference.
  * `val_nll_batches` -- `val/loss` is the velocity MSE here, not a
    likelihood, so it is comparable across flow-matching runs but not
    against the NSF/MAF runs. `val/nll` is the real thing, measured by an
    ODE solve on this many validation batches per epoch.

Early stopping and checkpointing still key off `val/loss`, which is the
right objective to select on within a run.
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
    config.id = None
    config.tags = ['nle', 'flow_matching', '75ds']
    config.debug = False
    config.checkpoint = None  # Path to an NLE checkpoint to resume from
    config.reset_optimizer = False
    config.enable_progress_bar = True

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()

    # Continuous flow p(x | theta, cond, xerr), trained by flow matching.
    model.flows = ConfigDict()
    model.flows.flow_type = 'flow_matching'
    # One MLP for the whole velocity field, so this is the model's entire
    # capacity -- unlike the NSF, where it is the size of each of 12
    # conditioners. Depth matters less than width here.
    model.flows.hidden_features = [512, 512, 512]
    model.flows.activation = 'elu'
    # Fourier features on t. The default (3) is enough for a velocity
    # field this smooth.
    model.flows.freqs = 3
    # Midpoint steps per density solve, at training-time validation and at
    # inference alike.
    model.flows.steps = 16
    # Rows per solve. The exact trace holds `features` copies of the
    # activations, so an unchunked inference batch (~800k rows) peaks near
    # 18 GiB; this bounds it without costing time.
    model.flows.chunk = 262_144

    # Validation batches on which the true NLL is also measured (one ODE
    # solve each) and logged as `val/nll`.
    model.val_nll_batches = 8

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
    # Unchanged from the NSF configs: the inference notebooks and the NPE
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
    # The velocity field is a plain MLP with an L2 target, none of the
    # spline brittleness that kept the NSF at 5e-4 on an 8192 batch, so
    # this takes the sqrt-scaling lr the batch size allows.
    optimizer.lr = 1.5e-3
    optimizer.weight_decay = 1e-4

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    # Stepped per epoch; decay_steps == num_epochs so the cosine finishes
    # exactly at the end of training. eta_min multiplies the base lr.
    scheduler.decay_steps = 80
    scheduler.warmup_steps = 3
    scheduler.eta_min = 1e-3
    scheduler.interval = 'epoch'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 8192
    config.eval_batch_size = 16384
    config.num_epochs = 80
    config.num_steps = -1
    config.patience = 20
    config.gradient_clip_val = 1.0
    config.save_top_k = 5

    return config
