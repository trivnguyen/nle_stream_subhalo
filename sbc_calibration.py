"""Simulation-based calibration for the per-star NLE posterior.

Answers the question a single corner plot cannot: over many (theta, x)
pairs drawn from the prior and the simulator, is the posterior calibrated?
If the inference is correct, the rank of each true parameter within its own
posterior samples is Uniform over the sample size -- so the rank histogram
is flat. A U-shape means the posterior is too narrow (overconfident), a
central peak means too wide, and a slope means biased.

Run in two phases:

    # one SLURM array task, datasets [start, start + n_datasets)
    python sbc_calibration.py run --start 0 --n-datasets 8 \
        --output-dir /scratch/$USER/nle_sbc/<tag>

    # once every task has finished
    python sbc_calibration.py combine \
        --output-dir /scratch/$USER/nle_sbc/<tag>

`slurm/sbc_calibration.sbatch` is the array wrapper.

Efficiency
----------
The cost is dominated by flow evaluations: one MCMC step costs
(n_datasets_in_flight x n_walkers x n_stars) rows through the flow. The
`lockstep` sampler advances every dataset in the task's batch on the same
step, so those rows go through as one large call instead of one small call
per dataset -- the same total work, but at a batch size the GPU is actually
busy at. Measured on this project's checkpoint, a serial per-dataset run
sustains ~2.7e5 rows/s and the lockstep run ~2x that.

Budget, per dataset, at the defaults (64 walkers, 6000 steps):

    n_stars    rows / dataset    serial      lockstep
        250          9.6e7        ~6 min       ~3 min
       1000          3.8e8       ~24 min      ~12 min

200 datasets at 250 stars is therefore ~10 GPU-hours lockstep, which splits
cleanly over a 25-task array. Prefer 250 stars: the training streams have
~233 stars each, so that is also the star count the likelihood's i.i.d.
assumption was least wrong at.

Notes
-----
`--delta-vmin` defaults to 0.03, the cut the `9p_AAU_deltaVmin0p03` dataset
was generated with -- NOT `stream_sims.prior.Prior`'s 3.0 default. SBC is
only meaningful when the prior the thetas are drawn from is the prior the
sampler runs under and the prior the flow was trained under. Those three
must be the same number, and the script uses one value for all three.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent / 'sbi_stream_pipeline'))
sys.path.insert(0, os.path.expanduser('~/my_modules'))

from stream_sims import prior, sims                      # noqa: E402
from nle_stream_subhalo.nle import NLE                   # noqa: E402
from nle_stream_subhalo.transforms import (              # noqa: E402
    StarBatch, build_transformation)

# Renamed 2026-08-10 from nsf_9params_200ds_track_v1_fixedprior.py when
# the configs were consolidated. Verified byte-identical to the
# production run xntr3svo's config_snapshot.py, so `find_latest_checkpoint`
# still resolves to the same checkpoint every earlier SBC run used.
DEFAULT_CONFIG = PROJECT_DIR / 'configs' / 'nsf_9params_200ds_track.py'


# ---------------------------------------------------------------------------
# Model and simulator
# ---------------------------------------------------------------------------

def load_config(path):
    """Import a `get_config()` config module from a file path."""
    spec = importlib.util.spec_from_file_location('train_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def find_latest_checkpoint(config_path):
    """Find the newest checkpoint among runs of `config_path`.

    Args:
        config_path: Path to the training config (`configs/*.py`).

    Returns:
        (run_dir, checkpoint_path). `last.ckpt` wins when present.

    Raises:
        FileNotFoundError: If no run with a matching config snapshot has
            any checkpoint yet.
    """
    config = load_config(config_path)
    model_root = Path(config.workdir) / config.wandb_project
    config_text = Path(config_path).read_text()

    candidates = []
    for run_dir in model_root.iterdir():
        snapshot = run_dir / 'config_snapshot.py'
        ckpt_dir = run_dir / 'checkpoints'
        if not snapshot.is_file() or not ckpt_dir.is_dir():
            continue
        if snapshot.read_text() != config_text:
            continue
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if not ckpts:
            continue
        last = ckpt_dir / 'last.ckpt'
        ckpt = last if last.is_file() else max(
            ckpts, key=lambda p: p.stat().st_mtime)
        candidates.append((ckpt.stat().st_mtime, run_dir, ckpt))

    if not candidates:
        raise FileNotFoundError(
            f'No checkpointed run of {config_path} under {model_root}')
    _, run_dir, ckpt = max(candidates, key=lambda c: c[0])
    return run_dir, ckpt


class Setup:
    """Everything a task needs: model, prior, simulator column maps.

    Attributes:
        model: The loaded NLE, in eval mode on `device`.
        labels: theta column order.
        x_labels: x column order.
        prior_min, prior_max: The box.
        delta_vmin: The detectability cut used everywhere in this run.
    """

    def __init__(self, config_path, delta_vmin, phi1_range, device,
                 run_dir=None):
        self.device = device
        if run_dir is None:
            self.run_dir, self.ckpt_path = find_latest_checkpoint(config_path)
        else:
            # Sweeps share one config file across arms, so a snapshot
            # comparison cannot tell them apart -- name the run directly.
            self.run_dir = Path(run_dir)
            ckpt_dir = self.run_dir / 'checkpoints'
            last = ckpt_dir / 'last.ckpt'
            self.ckpt_path = last if last.is_file() else max(
                ckpt_dir.glob('*.ckpt'), key=lambda p: p.stat().st_mtime)
        snapshot = json.loads(
            (self.run_dir / 'config_snapshot.json').read_text())
        self.snapshot = snapshot
        self.x_labels = tuple(snapshot['x_labels'])
        self.labels = tuple(snapshot['labels'])
        self.ndim = len(self.labels)

        self.pre_transforms = build_transformation(
            **snapshot['pre_transforms'])
        self.model = NLE.load_from_checkpoint(
            self.ckpt_path, map_location=device).eval().to(device)

        self.sim_cols = [sims.NODE_FEATURE_NAMES.index(k)
                         for k in self.x_labels]
        self.err_cols = [a['feature_idx']
                         for a in snapshot['pre_transforms']
                         ['uncertainty_args']]
        self.n_err = len(self.err_cols)
        self.phi1_range = phi1_range

        self.delta_vmin = delta_vmin
        self.prior = prior.Prior(delta_vmin=delta_vmin)
        self.prior_min = self.prior.prior_min
        self.prior_max = self.prior.prior_max
        self.param_idx = {k: i for i, k in enumerate(self.labels)}
        if tuple(self.labels) != tuple(prior.Prior.label_ordering):
            raise ValueError(
                f'checkpoint labels {self.labels} != prior ordering '
                f'{prior.Prior.label_ordering}')

    # -- prior ------------------------------------------------------------

    def delta_v(self, theta):
        """2 G M / (b v_rel), the detectability statistic, on (n, D) rows."""
        theta = np.atleast_2d(theta)
        mass = 10 ** theta[:, self.param_idx['log_mass']] * 1e7
        v = np.hypot(theta[:, self.param_idx['v_rel_perp']],
                     theta[:, self.param_idx['v_rel_para']])
        return 2 * 4.302e-6 * mass / (
            theta[:, self.param_idx['impact_param']] * v)

    def log_prior(self, theta):
        """Uniform over the accepted region, -inf outside."""
        theta = np.atleast_2d(theta)
        inside = np.all(
            (theta >= self.prior_min) & (theta <= self.prior_max), axis=-1)
        ok = inside & (self.delta_v(theta) > self.delta_vmin)
        return np.where(ok, 0.0, -np.inf)

    def sample_prior(self, n, rng):
        """Rejection-draw `n` rows from the accepted prior region."""
        out, got = [], 0
        while got < n:
            cand = rng.uniform(
                self.prior_min, self.prior_max, size=(4 * n + 32, self.ndim))
            keep = cand[np.isfinite(self.log_prior(cand))]
            out.append(keep)
            got += len(keep)
        return np.concatenate(out)[:n]

    # -- simulator --------------------------------------------------------

    def simulate_stream(self, theta, n_particles, seed):
        """One seeded, phi1-cut simulator stream in `x_labels` order.

        Raises:
            RuntimeError: If the simulator rejected these parameters or
                the phi1 cut left nothing.
        """
        _, feats = sims.simulate_one(
            np.asarray(theta, dtype=np.float64),
            num_particles=n_particles, seed=int(seed))
        if feats is None:
            raise RuntimeError('simulator returned nothing')
        phi1 = feats[:, sims.NODE_FEATURE_NAMES.index('phi1')]
        feats = feats[(phi1 >= self.phi1_range[0])
                      & (phi1 <= self.phi1_range[1])]
        if len(feats) == 0:
            raise RuntimeError('phi1 cut removed every star')
        return feats[:, self.sim_cols]

    def make_observation(self, theta, n_stars, n_particles, seed):
        """Simulate at `theta` and observe once.

        Returns:
            (x, xerr) numpy arrays of shape (n_stars, n_features) and
            (n_stars, n_err).

        Raises:
            RuntimeError: If the simulation failed.
        """
        rng = np.random.default_rng(seed)
        stream = self.simulate_stream(
            theta, n_particles, rng.integers(2 ** 32))
        idx = rng.choice(len(stream), size=n_stars,
                         replace=len(stream) < n_stars)
        torch.manual_seed(int(rng.integers(2 ** 31)))
        obs = self.pre_transforms(
            StarBatch(x=torch.tensor(stream[idx], dtype=torch.float32)))
        if obs.xerr is None or obs.xerr.shape[1] != self.n_err:
            raise RuntimeError('the measurement model did not run once')
        return obs.x.numpy(), obs.xerr.numpy()


# ---------------------------------------------------------------------------
# Likelihood, batched over datasets and walkers at once
# ---------------------------------------------------------------------------

class BatchedLikelihood:
    """Summed per-star log p(x_d | theta) for several datasets at once.

    All datasets in a task share one call: the star blocks are
    concatenated, so the flow sees a single (sum_d n_theta_d * n_stars, .)
    batch rather than one small batch per dataset. `chunk` bounds the rows
    per forward pass, which is what actually caps memory -- the flow's
    activations, not the inputs, are the large thing.

    Args:
        setup: A `Setup`.
        obs_x: (D, n_stars, n_features) observed stars, one row per dataset.
        obs_err: (D, n_stars, n_err) their sigmas.
        chunk: Maximum rows per forward pass.
    """

    def __init__(self, setup, obs_x, obs_err, chunk=200_000):
        self.setup = setup
        self.device = setup.device
        self.x = torch.tensor(np.asarray(obs_x), dtype=torch.float32,
                              device=self.device)
        self.err = torch.tensor(np.asarray(obs_err), dtype=torch.float32,
                                device=self.device)
        self.n_datasets, self.n_stars = self.x.shape[0], self.x.shape[1]
        self.chunk = chunk
        self.rows = 0

    @torch.no_grad()
    def __call__(self, thetas):
        """Log-likelihood of each theta against its own dataset.

        Args:
            thetas: (D, W, ndim) -- W candidate thetas for each of the D
                datasets, in the same dataset order as `obs_x`.

        Returns:
            (D, W) log-likelihoods.
        """
        thetas = np.asarray(thetas)
        n_d, n_w = thetas.shape[0], thetas.shape[1]
        if n_d != self.n_datasets:
            raise ValueError(
                f'{n_d} theta blocks for {self.n_datasets} datasets')

        t = torch.tensor(thetas.reshape(-1, thetas.shape[-1]),
                         dtype=torch.float32, device=self.device)
        # Row (d * W + w) * n_stars + s is (dataset d, walker w, star s).
        star_x = self.x.repeat_interleave(n_w, dim=0).reshape(
            -1, self.x.shape[-1])
        star_e = self.err.repeat_interleave(n_w, dim=0).reshape(
            -1, self.err.shape[-1])
        theta_rows = t.repeat_interleave(self.n_stars, dim=0)

        total_rows = theta_rows.shape[0]
        self.rows += total_rows
        out = torch.empty(total_rows, device=self.device)
        for lo in range(0, total_rows, self.chunk):
            hi = min(lo + self.chunk, total_rows)
            out[lo:hi] = self.setup.model.log_prob(StarBatch(
                x=star_x[lo:hi], xerr=star_e[lo:hi],
                theta=theta_rows[lo:hi]))
        return out.view(n_d * n_w, self.n_stars).sum(1).view(
            n_d, n_w).cpu().numpy()


# ---------------------------------------------------------------------------
# pocomc, one dataset at a time
# ---------------------------------------------------------------------------

class _PocoPrior:
    """`Setup`'s prior in the interface pocomc expects."""

    def __init__(self, setup, seed):
        self.setup = setup
        self._rng = np.random.default_rng(seed)

    def logpdf(self, theta):
        return self.setup.log_prior(theta)

    def rvs(self, size=1):
        return self.setup.sample_prior(size, self._rng)

    @property
    def bounds(self):
        return np.stack([self.setup.prior_min, self.setup.prior_max], axis=1)

    @property
    def dim(self):
        return self.setup.ndim


def run_pocomc(setup, obs_x, obs_err, n_total, n_draws, chunk, seed,
               temperature=1.0):
    """Sample each dataset's posterior with pocomc, serially.

    Preconditioned SMC rather than the random-walk Metropolis below,
    because on this posterior the random walk does not mix: measured
    2026-08-06 over 160 datasets at 6000 steps, the median split R-hat was
    5.8 and the pooled chains still spanned 24-86% of the prior box. The
    posterior is ~1% of the box in `time_impact` and strongly correlated
    through `angle_pos_impact + angle_vel_delta`, which a diagonal
    proposal tuned to 0.234 acceptance cannot traverse. Calibrating a
    sampler that has not converged measures the sampler, and in that
    instance would have read as *under*-confident -- the exact opposite of
    the truth.

    It is also what the inference notebooks actually run, which is what
    SBC should be calibrating, and it is ~4x cheaper per dataset.

    Args:
        setup: A `Setup`.
        obs_x: (D, n_stars, n_features) observed stars.
        obs_err: (D, n_stars, n_err) their sigmas.
        n_total: pocomc particles per dataset.
        n_draws: Equal-weight draws kept per dataset. Fixed across
            datasets, since an SBC rank is only uniform over a fixed
            sample size.
        chunk: Rows per flow forward pass.
        seed: Base seed; dataset d uses `seed + d`.
        temperature: Divides the log-likelihood, so the sampler targets
            `prior * L^(1/T)`. T > 1 widens the posterior by ~sqrt(T) if
            the errors are Gaussian; SBC is how you find out whether they
            are. T = 1 is the untempered run.

    Returns:
        (chains, ess, rows) -- chains is (D, n_draws, ndim), ess is (D,)
        the importance-weight effective sample size, rows the flow rows
        evaluated.
    """
    import pocomc

    chains, ess, rows = [], [], 0
    for d in range(len(obs_x)):
        like = BatchedLikelihood(
            setup, obs_x[d:d + 1], obs_err[d:d + 1], chunk=chunk)

        def loglike(theta, _like=like, _t=temperature):
            """(n, ndim) physical theta -> (n,) tempered log-likelihood."""
            return _like(np.atleast_2d(theta)[None, ...])[0] / _t

        sampler = pocomc.Sampler(
            prior=_PocoPrior(setup, seed + d), likelihood=loglike,
            vectorize=True, random_state=seed + d)
        sampler.run(n_total=n_total, n_evidence=0, progress=False)

        chain, _, _ = sampler.posterior(resample=True)
        _, weights, _, _ = sampler.posterior()
        weights = weights / weights.sum()
        ess.append(1.0 / np.sum(weights ** 2))

        # Fixed draw count, so every dataset's rank is on the same scale.
        pick = np.random.default_rng(seed + d).choice(
            len(chain), size=n_draws, replace=len(chain) < n_draws)
        chains.append(chain[pick])
        rows += like.rows
        print(f'  dataset {d}: ESS {ess[-1]:.0f} / {len(chain)}', flush=True)

    return np.stack(chains), np.array(ess), rows


# ---------------------------------------------------------------------------
# Lockstep adaptive random-walk Metropolis
# ---------------------------------------------------------------------------

def lockstep_rwmh(setup, loglike, n_walkers, n_steps, n_burn, thin, rng):
    """Advance one MH ensemble per dataset, all on the same step.

    A plain random-walk Metropolis with a diagonal Gaussian proposal whose
    scale is adapted per dataset during burn-in toward a 0.234 acceptance
    rate. Deliberately vanilla: the point of a calibration run is that the
    sampler is simple enough to be above suspicion, and every dataset gets
    identical treatment.

    Args:
        setup: A `Setup`, for the prior.
        loglike: A `BatchedLikelihood`.
        n_walkers: Chains per dataset.
        n_steps: Steps per chain, including burn-in.
        n_burn: Steps discarded; also the adaptation window.
        thin: Keep every `thin`-th post-burn-in step.
        rng: numpy Generator.

    Returns:
        (chains, accept, scale) -- chains is
        (D, n_walkers, n_kept, ndim), accept is (D,), scale is
        (D, ndim), the adapted proposal.
    """
    n_d, ndim = loglike.n_datasets, setup.ndim
    span = setup.prior_max - setup.prior_min

    pos = np.stack([setup.sample_prior(n_walkers, rng) for _ in range(n_d)])
    logp = setup.log_prior(pos.reshape(-1, ndim)).reshape(n_d, n_walkers)
    logp = logp + loglike(pos)

    scale = np.tile(0.02 * span, (n_d, 1))
    kept = list(range(n_burn, n_steps, thin))
    chains = np.empty((n_d, n_walkers, len(kept), ndim), dtype=np.float32)
    accepted = np.zeros((n_d, n_walkers))
    window = np.zeros((n_d, n_walkers))
    keep_at = {step: k for k, step in enumerate(kept)}

    for step in range(n_steps):
        prop = pos + scale[:, None, :] * rng.normal(
            size=(n_d, n_walkers, ndim))
        lp = setup.log_prior(prop.reshape(-1, ndim)).reshape(n_d, n_walkers)
        live = np.isfinite(lp)
        if live.any():
            # The batch has to stay rectangular -- one likelihood row per
            # (dataset, walker) -- so out-of-prior proposals are replaced
            # by the walker's current position and their result thrown
            # away. Rejecting them costs a wasted row, not a wrong answer.
            filled = np.where(live[..., None], prop, pos)
            lp = np.where(live, lp + loglike(filled), -np.inf)

        take = np.log(rng.uniform(size=(n_d, n_walkers))) < (lp - logp)
        pos = np.where(take[..., None], prop, pos)
        logp = np.where(take, lp, logp)
        accepted += take
        window += take

        if step in keep_at:
            chains[:, :, keep_at[step]] = pos
        # Adaptation stops at burn-in: a proposal that keeps changing is
        # not a Markov chain with the right stationary distribution.
        if step < n_burn and (step + 1) % 100 == 0:
            rate = window.mean(axis=1) / 100
            scale *= np.exp(0.5 * (rate - 0.234))[:, None]
            window[:] = 0

    return chains, accepted.mean(axis=1) / n_steps, scale


def split_rhat(chains):
    """Split Gelman-Rubin R-hat per parameter.

    Args:
        chains: (n_chain, n_step, ndim).

    Returns:
        (ndim,) R-hat; < 1.01 is the usual pass.
    """
    n_step = chains.shape[1]
    half = n_step // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    n = split.shape[1]
    chain_means = split.mean(axis=1)
    w = split.var(axis=1, ddof=1).mean(axis=0)
    b = n * chain_means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * w + b / n
    return np.sqrt(np.where(w > 0, var_hat / w, np.nan))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(args):
    """Sample the posterior for one array task's slice of datasets."""
    device = ('cuda' if torch.cuda.is_available() and not args.cpu
              else 'cpu')
    torch.set_num_threads(args.threads)
    setup = Setup(args.config, args.delta_vmin,
                  (args.phi1_min, args.phi1_max), device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'device       : {device}')
    print(f'run          : {setup.run_dir.name} / {setup.ckpt_path.name}')
    print(f'delta_vmin   : {setup.delta_vmin}')
    print(f'temperature  : {args.temperature}')
    print(f'datasets     : [{args.start}, {args.start + args.n_datasets})')

    # Every dataset's truth and observation is a pure function of the
    # global seed and its own index, so tasks never have to coordinate and
    # a task can be re-run without changing what it is calibrating.
    obs_x, obs_err, thetas, indices = [], [], [], []
    for i in range(args.start, args.start + args.n_datasets):
        seq = np.random.SeedSequence([args.seed, i])
        rng = np.random.default_rng(seq)
        theta = setup.sample_prior(1, rng)[0]
        try:
            x, err = setup.make_observation(
                theta, args.n_stars, args.n_particles,
                int(rng.integers(2 ** 32)))
        except RuntimeError as exc:
            print(f'dataset {i}: skipped ({exc})')
            continue
        obs_x.append(x)
        obs_err.append(err)
        thetas.append(theta)
        indices.append(i)

    if not thetas:
        raise RuntimeError('every dataset in this task failed to simulate')
    thetas = np.stack(thetas)
    print(f'simulated    : {len(thetas)} datasets, '
          f'{args.n_stars} stars each')

    t0 = time.time()
    if args.sampler == 'pocomc':
        flat, ess, rows = run_pocomc(
            setup, np.stack(obs_x), np.stack(obs_err), args.n_total,
            args.sbc_draws, args.chunk, args.seed + 7000 + args.start,
            temperature=args.temperature)
        # pocomc returns independent equal-weight draws, so there is no
        # chain to compute R-hat on. ESS is the quality metric instead,
        # and `combine` filters on it.
        rhat = np.ones((len(thetas), setup.ndim))
        accept = np.full(len(thetas), np.nan)
    else:
        loglike = BatchedLikelihood(
            setup, np.stack(obs_x), np.stack(obs_err), chunk=args.chunk)
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, 999, args.start]))
        chains, accept, _ = lockstep_rwmh(
            setup, loglike, args.n_walkers, args.n_steps, args.n_burn,
            args.thin, rng)
        flat = chains.reshape(len(thetas), -1, setup.ndim)
        rhat = np.stack([split_rhat(chains[d]) for d in range(len(thetas))])
        ess = np.full(len(thetas), np.nan)
        rows = loglike.rows
    seconds = time.time() - t0
    print(f'sampled      : {seconds:.0f} s, {rows / seconds:,.0f} rows/s')

    # SBC rank: how many posterior draws fall below the truth. Uniform over
    # 0..n_draws if the posterior is calibrated.
    ranks = (flat < thetas[:, None, :]).sum(axis=1)

    path = out_dir / f'sbc_{args.start:05d}.npz'
    np.savez_compressed(
        path,
        index=np.array(indices), theta_true=thetas, ranks=ranks,
        n_draws=flat.shape[1], rhat=rhat, accept=accept, ess=ess,
        quantiles=np.percentile(flat, [2.5, 16, 50, 84, 97.5], axis=1),
        posterior=flat[:, ::max(1, flat.shape[1] // args.keep_draws)],
        labels=np.array(setup.labels), delta_vmin=setup.delta_vmin,
        n_stars=args.n_stars, sampler=args.sampler,
        temperature=args.temperature, seconds=seconds)
    print(f'wrote        : {path}')
    if args.sampler == 'pocomc':
        print(f'ESS          : median {np.median(ess):.0f}, '
              f'min {np.min(ess):.0f}')
    else:
        bad = int((np.nanmax(rhat, axis=1) > 1.01).sum())
        print(f'R-hat > 1.01 : {bad} / {len(thetas)} datasets')


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

def combine(args):
    """Aggregate every task's output into SBC and coverage diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import chisquare

    out_dir = Path(args.output_dir)
    files = sorted(out_dir.glob('sbc_*.npz'))
    if not files:
        raise FileNotFoundError(f'no sbc_*.npz under {out_dir}')

    parts = [np.load(f, allow_pickle=True) for f in files]
    labels = [str(s) for s in parts[0]['labels']]
    n_draws = int(parts[0]['n_draws'])
    ranks = np.concatenate([p['ranks'] for p in parts])
    theta_true = np.concatenate([p['theta_true'] for p in parts])
    rhat = np.concatenate([p['rhat'] for p in parts])
    posterior = np.concatenate([p['posterior'] for p in parts])
    quantiles = np.concatenate([p['quantiles'] for p in parts], axis=1)

    keep = np.nanmax(rhat, axis=1) <= args.rhat_max
    print(f'{len(files)} files, {len(ranks)} datasets, '
          f'{keep.sum()} with R-hat <= {args.rhat_max}')
    ranks, theta_true = ranks[keep], theta_true[keep]
    posterior, quantiles = posterior[keep], quantiles[:, keep]

    # -- SBC rank histograms ---------------------------------------------
    n_bins = args.bins
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), dpi=120)
    print(f'\n{"param":>18} {"chi2/dof":>9} {"p":>9} {"shape":>22}')
    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        counts, edges = np.histogram(
            ranks[:, i], bins=n_bins, range=(0, n_draws + 1))
        expected = len(ranks) / n_bins
        chi2 = chisquare(counts, [expected] * n_bins)
        # 99% band for a flat histogram, from the binomial on each bin.
        sigma = np.sqrt(expected * (1 - 1 / n_bins))
        ax.axhspan(expected - 3 * sigma, expected + 3 * sigma,
                   color='0.85', zorder=0)
        ax.axhline(expected, color='k', ls='--', lw=1)
        ax.stairs(counts, edges, color='C0', lw=1.5)
        ax.set_xlabel(f'{label} rank', fontsize=8)
        ax.tick_params(labelsize=6)

        first, last = counts[0] + counts[-1], counts.sum()
        middle = counts[n_bins // 3:2 * n_bins // 3].sum()
        if chi2.pvalue > 0.01:
            shape = 'flat (calibrated)'
        elif first / last > 2.5 / n_bins:
            shape = 'U -> OVERCONFIDENT'
        elif middle / last > 1.4 / 3:
            shape = 'peaked -> too wide'
        else:
            shape = 'sloped -> biased'
        ax.set_title(shape, fontsize=8)
        print(f'{label:>18} {chi2.statistic / (n_bins - 1):>9.2f} '
              f'{chi2.pvalue:>9.2g} {shape:>22}')
    fig.suptitle(
        f'SBC rank histograms, {len(ranks)} datasets '
        f'({n_draws} posterior draws each)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'sbc_ranks.png', bbox_inches='tight')
    print(f'\nwrote {out_dir / "sbc_ranks.png"}')

    # -- central-interval coverage ---------------------------------------
    # Model-free and easier to read than a rank histogram: what fraction of
    # truths fall inside the posterior's own 68% and 95% intervals.
    print(f'\n{"param":>18} {"68% cover":>10} {"95% cover":>10}')
    for i, label in enumerate(labels):
        c68 = np.mean((theta_true[:, i] >= quantiles[1, :, i])
                      & (theta_true[:, i] <= quantiles[3, :, i]))
        c95 = np.mean((theta_true[:, i] >= quantiles[0, :, i])
                      & (theta_true[:, i] <= quantiles[4, :, i]))
        print(f'{label:>18} {c68:>10.3f} {c95:>10.3f}')

    # -- expected-coverage curve, all parameters jointly ------------------
    # TARP: the fraction of posteriors whose credible region at level a
    # contains the truth, against a. The diagonal is calibration; below it
    # is overconfidence.
    try:
        from tarp import get_tarp_coverage
        # `samples` is (n_draws, n_sims, ndim) and the return is
        # (ecp, alpha) in that order -- both easy to get backwards.
        ecp, alpha = get_tarp_coverage(
            np.swapaxes(posterior, 0, 1), theta_true, norm=True,
            seed=0)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='calibrated')
        ax.plot(alpha, ecp, color='C0', lw=2)
        ax.set_xlabel('credibility level')
        ax.set_ylabel('expected coverage')
        ax.set_title('TARP coverage (joint, all 9 parameters)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / 'tarp_coverage.png', bbox_inches='tight')
        print(f'wrote {out_dir / "tarp_coverage.png"}')
    except ImportError:
        print('tarp not installed; skipping the joint coverage curve')


# ---------------------------------------------------------------------------

def main():
    """Parse arguments and dispatch to `run` or `combine`."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)

    r = sub.add_parser('run', help='sample one slice of datasets')
    r.add_argument('--config', default=str(DEFAULT_CONFIG))
    r.add_argument('--output-dir', required=True)
    r.add_argument('--start', type=int, default=0)
    r.add_argument('--n-datasets', type=int, default=8,
                   help='datasets in this task, all sampled in lockstep')
    r.add_argument('--n-stars', type=int, default=250,
                   help='observed stars per dataset; the training streams '
                        'have ~233, and cost is linear in this')
    r.add_argument('--n-particles', type=int, default=1000,
                   help='particles requested from the simulator, before '
                        'the phi1 cut')
    r.add_argument('--sampler', choices=('pocomc', 'mh'), default='pocomc',
                   help="'pocomc' is the default because the random walk "
                        'does not mix on this posterior -- see run_pocomc')
    r.add_argument('--n-total', type=int, default=2000,
                   help='pocomc particles per dataset')
    r.add_argument('--temperature', type=float, default=1.0,
                   help='divide the log-likelihood by this; T > 1 widens '
                        'the posterior. pocomc only')
    r.add_argument('--sbc-draws', type=int, default=999,
                   help='equal-weight draws per dataset; an SBC rank is '
                        'only uniform over a fixed sample size')
    r.add_argument('--n-walkers', type=int, default=64, help='mh only')
    r.add_argument('--n-steps', type=int, default=6000, help='mh only')
    r.add_argument('--n-burn', type=int, default=2000, help='mh only')
    r.add_argument('--thin', type=int, default=10, help='mh only')
    r.add_argument('--keep-draws', type=int, default=500,
                   help='posterior draws stored per dataset, for TARP')
    r.add_argument('--chunk', type=int, default=200_000,
                   help='max rows per flow forward pass; lower this '
                        'first if the job OOMs')
    r.add_argument('--delta-vmin', type=float, default=0.03,
                   help="the DATASET's cut, not prior.Prior's 3.0 default")
    r.add_argument('--phi1-min', type=float, default=-25.0)
    r.add_argument('--phi1-max', type=float, default=25.0)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--threads', type=int, default=8)
    r.add_argument('--cpu', action='store_true')
    r.set_defaults(func=run)

    c = sub.add_parser('combine', help='aggregate and plot')
    c.add_argument('--output-dir', required=True)
    c.add_argument('--bins', type=int, default=20)
    c.add_argument('--rhat-max', type=float, default=1.05)
    c.set_defaults(func=combine)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
