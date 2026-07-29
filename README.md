# nle_stream_subhalo
NLE for stellar stream-subhalo impact

## Choosing a flow (`model.flows.flow_type`)

`'spline'` (NSF) and `'maf'` are discrete flows trained by maximum
likelihood: `val/loss` *is* the negative log-likelihood, and everything
works the way it always has.

`'flow_matching'` is a continuous flow — one MLP velocity field, trained
by conditional flow matching, with the density recovered from a
fixed-step ODE solve (`nle_stream_subhalo/flow_matching.py`, example
config `configs/fm_9params_75ds.py`). Three things change:

* **`val/loss` is no longer a likelihood.** It is the velocity-regression
  loss, so it is comparable between flow-matching runs — and it is still
  the right thing to early-stop and checkpoint on — but not against an
  NSF/MAF run. `val/nll` is logged alongside it for that, measured by an
  ODE solve on `model.val_nll_batches` validation batches per epoch.
* **`log_prob` costs ~3x an NSF evaluation**, `sample` about 10x less.
  Posterior runs are likelihood-bound, so budget ~3x; the predictive
  checks get cheaper.
* **`model.flows.steps`** sets the density solver. Validate it on the
  trained model before trusting a posterior — `flow(context).log_prob(
  target)` is zuko's adaptive solver and is the reference. Per-star error
  is multiplied by the number of stars in the stream.

Capacity does not transfer between the two families: an NSF spends its
parameters on spline bins and is very expressive per parameter, while the
velocity field is a plain MLP and has to be much wider to match. See the
docstring of `configs/fm_9params_75ds.py`.

## Weights & Biases logging (offline on compute)

Compute nodes have no outbound network, so `train_nle.py` never tries to
reach wandb from one. The mode is picked by
`training.resolve_wandb_mode`, in precedence order:

1. `config.debug = True` → `disabled` (no run at all)
2. `config.wandb_mode` → used as given
3. `WANDB_MODE` in the environment
4. any SLURM job (`SLURM_JOB_ID` set) → `offline`, without probing
5. otherwise → probe the wandb API, `online` only if it answers

An offline run is complete on disk: metrics, config, and checkpoint
artifacts all land under `config.workdir`. Upload it afterwards **from a
login node**, using the command the job prints at the end:

```bash
wandb sync /scratch/$USER/trained_models/nle/wandb/offline-run-<timestamp>-<id>
```

Checkpoints are always written to
`<workdir>/<wandb_project>/<run_id>/checkpoints/` regardless of mode, so
nothing depends on the sync having happened. `config.log_model`
(default `'all'`) additionally logs them as artifacts; offline those are
staged under `$WANDB_DATA_DIR`, which costs one extra on-disk copy per
checkpoint until you sync. Set `config.log_model = False` to skip it.

Because `$HOME` is read-only on compute nodes, `train_nle.py` redirects
`WANDB_DATA_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, and
`MPLCONFIGDIR` under `$SCRATCH` (default `/scratch/$USER`) before
importing wandb. All four honour a value already set in the environment,
which is how `slurm/train_nle.sbatch` overrides them.
