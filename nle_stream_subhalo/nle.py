"""Neural Likelihood Estimation (NLE) module for per-star stream data.

The model learns the single-star likelihood p(observables | theta, cond)
with a conditional normalizing flow, and can sample observables from it.
`theta` are the perturber parameters of the star's host stream, broadcast
to each of its member stars. It deliberately does ONLY what it is trained
on (per-star density + sampling) -- assumptions like i.i.d. stars, priors,
fixing/sampling `cond`, and MCMC live outside the model.

"""
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko

from .flow_matching import FlowMatching
from .transforms import StagePipeline, StarBatch, StreamTrack
from .utils import (MLP, configure_optimizers, get_activation,
                    resolve_schedule)


def build_flow(
    features: int,
    context: int,
    flow_type: str = 'spline',
    hidden_features=(64, 64),
    activation: str = None,
    **kwargs,
):
    """Build a conditional flow for p(target | context).

    `flow_type` is 'spline' or 'maf' (discrete, trained by maximum
    likelihood) or 'flow_matching' (continuous, trained on the velocity
    field -- see `flow_matching.FlowMatching`). `kwargs` are the remaining
    architecture arguments of the chosen type: `num_transforms` and, for
    splines, `num_bins`; `steps`, `chunk` and `freqs` for flow matching.
    An argument the type does not take raises, rather than silently doing
    nothing.
    """
    kwargs['hidden_features'] = hidden_features
    if activation is not None:
        kwargs['activation'] = get_activation(activation, return_instance=False)

    if flow_type == 'flow_matching':
        return FlowMatching(features, context, **kwargs)

    kwargs['transforms'] = kwargs.pop('num_transforms', 3)
    if flow_type == 'spline':
        return zuko.flows.NSF(
            features, context, bins=kwargs.pop('num_bins', 8), **kwargs)
    if flow_type == 'maf':
        return zuko.flows.MAF(features, context, **kwargs)
    raise ValueError(
        f"Unknown flow_type: {flow_type!r} "
        f"(use 'spline'/'maf'/'flow_matching')")


class NLE(pl.LightningModule):
    """Conditional-flow likelihood p(target | context) over single stars.

    Parameters
    ----------
    norm_dict : dict
        Per-field location/scale from `transforms.compute_field_norm`.
    flows_args : Mapping, optional
        Passed to `build_flow`.
    context_embedding_args : Mapping, optional
        If given, a single MLP embeds the full context before the flow.
        Keys: hidden_sizes, output_size, act_name, act_args, batch_norm,
        dropout.
    optimizer_args, scheduler_args : Mapping, optional
        Passed to `utils.configure_optimizers`.
    pre_transforms : callable, optional
        Observation pipeline (measurement uncertainty) applied to raw
        batches during training only.
    val_nll_batches : int
        Flow matching only: validation batches on which the true NLL is
        also measured. `val/loss` is then the velocity-regression loss,
        which is not comparable across architectures, so `val/nll` is
        logged alongside it for that. Each such batch costs an ODE solve.
    track_dict : dict, optional
        From `transforms.load_track_dict`. When given, `x` is
        re-expressed as offsets from the unperturbed stream track before
        being normalized, and `sample` maps its draws back. Lives on the
        model rather than in `pre_transforms` so training and inference
        cannot disagree about it, and so a checkpoint can never be paired
        with the wrong track.
    x_context_labels : sequence of str, optional
        Observable columns to condition on rather than model, so the
        flow learns p(x | x_context, theta, theta_context, xerr).
        Disjoint from `x_labels`: a column is modelled or conditioned
        on, never both.  `batch.x` holds them concatenated in that
        order, modelled first.  Columns
        shaped by the selection function belong here: a stream's
        footprint and membership cuts act almost entirely on phi1 and
        phi2, and conditioning on a coordinate removes its selection
        function from the likelihood without having to model it.  The
        cost is the marginal it discards -- conditioning on phi1 gives
        up the linear density along the stream.  Empty by default,
        which models every column.
    x_labels : sequence of str, optional
        Observable columns the flow models.  Disjoint from
        `x_context_labels`.
        Column order of `x`. Required with `track_dict`, which stores its
        coordinates by name.
    """

    def __init__(
        self,
        norm_dict: dict,
        flows_args: dict = None,
        context_embedding_args: dict = None,
        optimizer_args: dict = None,
        scheduler_args: dict = None,
        pre_transforms=None,
        val_nll_batches: int = 8,
        track_dict: dict = None,
        x_labels=None,
        x_context_labels=None,
    ):
        super().__init__()
        self.norm_dict = norm_dict
        self.flows_args = flows_args or {}
        self.context_embedding_args = context_embedding_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.val_nll_batches = val_nll_batches

        self.x_labels = list(x_labels) if x_labels is not None else None
        self.x_context_labels = (
            list(x_context_labels) if x_context_labels else [])
        # `batch.x` carries the modelled columns first, then the
        # conditioned ones, so the split is a prefix/suffix rather than
        # a lookup.  The track and the stage chain are defined over all
        # of them, which is what `x_all_labels` is for.
        self.x_all_labels = (
            (self.x_labels or []) + self.x_context_labels) or None
        if self.x_labels and set(self.x_labels) & set(self.x_context_labels):
            raise ValueError(
                'x_labels and x_context_labels must be disjoint; '
                f'both contain '
                f'{sorted(set(self.x_labels) & set(self.x_context_labels))}')

        self.save_hyperparameters(ignore=['pre_transforms'])

        # Order matters: the stage chain is built here, and the track may
        # be one of its stages.
        self._register_norm(norm_dict)
        self._register_track(track_dict, self.x_all_labels)
        self._setup_model()

    # -- setup --------------------------------------------------------------

    def _register_track(self, track_dict: dict, x_labels):
        """Attach the stream track, if this run uses one.

        Legacy path: runs whose `norm_dict` carries `x_stages` get the
        track as a stage in that chain instead, and `self.track` points at
        it so callers keep working either way.
        """
        if self.x_stages is not None:
            self.track = next(
                (stage for stage in self.x_stages.stages
                 if isinstance(stage, StreamTrack)), None)
            return
        if track_dict is None:
            self.track = None
            return
        if x_labels is None:
            raise ValueError(
                'track_dict needs x_labels: the track stores its '
                'coordinates by name, and the simulator\'s column order '
                'is not the one the configs use')

        self.track = StreamTrack(track_dict, x_labels)
        print(f"[NLE] Track over {len(self.track.coords)} coordinates "
              f"({', '.join(self.track.coords)}), "
              f"{self.track.knots.numel()} knots")

    def _register_norm(self, norm_dict: dict):
        """Register the context scalings, and the `x` stage chain.

        `x` goes through `transforms.stages`; `theta`, `xerr` and `cond`
        are always a plain affine scaling, since they are context rather
        than something the flow models.

        A `norm_dict` without `x_stages` is from before stages existed and
        keeps the old single affine `x` scaling, so old checkpoints load
        unchanged.
        """
        self.has_cond = 'cond_loc' in norm_dict
        self.x_stages = None

        fields = ['xerr', 'theta'] + (['cond'] if self.has_cond else [])
        if 'x_stages' not in norm_dict:
            fields.append('x')
        for field in fields:
            self.register_buffer(
                f'{field}_loc',
                torch.tensor(norm_dict[f'{field}_loc'], dtype=torch.float32))
            self.register_buffer(
                f'{field}_scale',
                torch.tensor(norm_dict[f'{field}_scale'], dtype=torch.float32))

        if 'x_stages' in norm_dict:
            self.x_stages = StagePipeline(
                norm_dict['x_stages'], self.x_all_labels)
            print(f'[NLE] x stages: {self.x_stages.describe()}')

    def _setup_model(self):
        """Set up the field layout, optional context MLP, and flow."""
        dims = {f: len(self.norm_dict[f'{f}_loc'])
                for f in ('xerr', 'theta')}
        # With stages there is no `x_loc` to read the width off; the chain
        # preserves the column count, so `x_labels` gives it.
        dims['x'] = (len(self.x_all_labels) if self.x_stages is not None
                     else len(self.norm_dict['x_loc']))
        if self.has_cond:
            dims['cond'] = len(self.norm_dict['cond_loc'])

        # Fixed order: theta, theta_context, x_context, xerr.  Anything
        # assembling a context by hand -- `inference.build_context` for
        # observed data above all -- has to lay it out the same way, so
        # it is written once here and read from `context_order`.
        self.target_fields = ['x', ]
        self.context_fields = (
            ['theta'] + (['cond'] if self.has_cond else []) + ['xerr'])
        self.context_order = (
            ['theta'] + (['cond'] if self.has_cond else [])
            + (['x_context'] if self.x_context_labels else []) + ['xerr'])

        if self.x_context_labels and not self.x_labels:
            raise ValueError(
                'x_context_labels needs x_labels: the flow has to have '
                'something left to model')

        n_model = len(self.x_labels or [])
        n_x_context = len(self.x_context_labels)
        # Modelled columns come first in `x`, conditioned ones after.
        self.register_buffer('x_target_idx',
                             torch.arange(n_model, dtype=torch.long),
                             persistent=False)
        self.register_buffer('x_context_idx',
                             torch.arange(n_model, n_model + n_x_context,
                                          dtype=torch.long),
                             persistent=False)

        target_size = sum(dims[f] for f in self.target_fields) - n_x_context
        context_size = (sum(dims[f] for f in self.context_fields)
                        + n_x_context)

        if self.context_embedding_args:
            args = dict(self.context_embedding_args)
            act = get_activation(
                args.pop('act_name', 'relu'), args.pop('act_args', None))
            self.context_embedding = MLP(
                input_size=context_size, act=act, **args)
            flow_context_size = self.context_embedding.output_size
            print(f"[NLE] Context MLP {context_size} -> {flow_context_size} "
                  f"with {sum(p.numel() for p in self.context_embedding.parameters()):,}"
                  f" parameters")
        else:
            self.context_embedding = nn.Identity()
            flow_context_size = context_size

        self.flow = build_flow(target_size, flow_context_size, **self.flows_args)
        self.is_continuous = isinstance(self.flow, FlowMatching)
        print(f"[NLE] Modeling p({','.join(self.target_fields)} | "
              f"{','.join(self.context_order)})")
        print(f"[NLE] Flow built with"
              f" {sum(p.numel() for p in self.flow.parameters()):,} parameters")

    # -- normalization / assembly ------------------------------------------

    def _normalize(self, batch: StarBatch) -> StarBatch:
        """Push `x` through the stage chain and scale the context fields.

        Everything lives here so every caller gets it -- training through
        `_prepare_batch`, inference through `log_prob` and `sample` -- with
        no way for the two paths to drift apart.
        """
        def norm(field, value):
            if value is None:
                return None
            return (value - getattr(self, f'{field}_loc')) / getattr(
                self, f'{field}_scale')

        x = batch.x
        if x is not None:
            if self.x_stages is not None:
                x = self.x_stages(x)
            else:
                if self.track is not None:
                    x = self.track.project(x)
                x = norm('x', x)

        return StarBatch(
            x=x,
            xerr=norm('xerr', batch.xerr),
            theta=norm('theta', batch.theta),
            cond=norm('cond', batch.cond) if self.has_cond else None,
        )

    def _denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of the `x` half of `_normalize`, for flow samples."""
        if self.x_stages is not None:
            return self.x_stages.inverse(x)

        loc = torch.cat([getattr(self, f'{f}_loc') for f in self.target_fields])
        scale = torch.cat(
            [getattr(self, f'{f}_scale') for f in self.target_fields])
        x = x * scale + loc
        return x if self.track is None else self.track.unproject(x)

    def _assemble(self, batch: StarBatch, fields) -> torch.Tensor:
        cols = []
        for field in fields:
            value = getattr(batch, field)
            if value is None:
                raise ValueError(f'batch.{field} is required but was None')
            cols.append(value)
        return torch.cat(cols, dim=1)

    def _target_context(self, norm_batch: StarBatch):
        """Split a normalized batch into (target, embedded context).

        The observable columns named in `x_context_labels` cross from
        the target into the context here, which is the only place the
        split is applied.
        """
        x = self._assemble(norm_batch, self.target_fields)
        target = (x[:, self.x_target_idx] if self.x_context_idx.numel()
                  else x)
        return target, self._context(norm_batch)

    def _context(self, norm_batch: StarBatch) -> torch.Tensor:
        """Assemble and embed the context, observables included.

        The one place the context is built, so `sample` and
        `_target_context` cannot disagree about whether the columns in
        `x_context_labels` are part of it.
        """
        head = ['theta'] + (['cond'] if self.has_cond else [])
        cols = [self._assemble(norm_batch, head)]
        if self.x_context_idx.numel():
            x = self._assemble(norm_batch, self.target_fields)
            cols.append(x[:, self.x_context_idx])
        cols.append(self._assemble(norm_batch, ['xerr']))
        return self.context_embedding(torch.cat(cols, dim=1))

    def _flow_log_prob(self, norm_batch: StarBatch, steps: Optional[int] = None) -> torch.Tensor:
        """Per-star log p(target | context) for a normalized StarBatch."""
        target, context = self._target_context(norm_batch)
        if self.is_continuous:
            return self.flow.log_prob(target, context, steps=steps)
        return self.flow(context).log_prob(target)

    # -- training -----------------------------------------------------------

    def _prepare_batch(self, batch: StarBatch) -> StarBatch:
        """Augment (add measurement noise) then normalize a raw batch."""
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)
        return self._normalize(batch)

    def _loss(self, norm_batch: StarBatch, steps: Optional[int] = None) -> torch.Tensor:
        """Training objective: velocity regression, or the NLL."""
        if self.is_continuous:
            return self.flow.loss(*self._target_context(norm_batch), steps=steps)
        return -self._flow_log_prob(norm_batch).mean()

    def training_step(self, batch, batch_idx):
        loss = self._loss(self._prepare_batch(batch))
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch.x.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        norm_batch = self._prepare_batch(batch)
        loss = self._loss(norm_batch)
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch.x.shape[0], sync_dist=True)

        # `val/loss` is the objective the run is early-stopped and
        # checkpointed on; `val/nll` is the same number for the discrete
        # flows but a separate ODE solve for flow matching, and is what
        # makes the two families comparable.
        if not self.is_continuous:
            nll = loss
        elif batch_idx < self.val_nll_batches:
            nll = -self._flow_log_prob(norm_batch).mean()
        else:
            return loss
        self.log(
            'val/nll', nll, on_step=False, on_epoch=True, prog_bar=False,
            logger=True, batch_size=batch.x.shape[0], sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler.

        An epoch-denominated schedule is resolved to steps here,
        because this is the first point where the trainer knows how
        long an epoch is.  `estimated_stepping_batches` is Lightning's
        own count for the whole run, so it already accounts for
        accumulate_grad_batches, drop_last and device sharding, none
        of which len(dataloader) would.
        """
        steps_per_epoch = 0
        trainer = getattr(self, '_trainer', None)
        if trainer is not None and trainer.max_epochs:
            steps_per_epoch = max(
                1, trainer.estimated_stepping_batches // trainer.max_epochs)

        scheduler_args = resolve_schedule(
            self.scheduler_args, steps_per_epoch)
        if scheduler_args is not self.scheduler_args and \
                scheduler_args != self.scheduler_args:
            print(f'[LR] {steps_per_epoch} steps/epoch -> '
                  f'decay {scheduler_args.get("decay_steps")}, '
                  f'warmup {scheduler_args.get("warmup_steps")}')
        return configure_optimizers(
            self.parameters(), self.optimizer_args, scheduler_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    # -- inference (observed data in physical units) -----------------------

    @torch.no_grad()
    def log_prob(self, batch: StarBatch, steps: Optional[int] = None) -> torch.Tensor:
        """Per-star log p(observables | theta, cond) for a physical batch.

        `batch` holds the observed per-star fields (`x`, `xerr`) plus the
        candidate `theta` and `cond`, all broadcast to one row per star by
        the caller. No augmentation is applied. Returns a (n_stars,)
        tensor; summing over stars, priors, and MCMC are the caller's job.
        """
        self.eval()
        return self._flow_log_prob(self._normalize(batch.to(self.device)), steps=steps)

    @torch.no_grad()
    def sample(self, batch: StarBatch, num_samples: int = 1, steps: Optional[int] = None) -> torch.Tensor:
        """Sample observables from p(target | context) for each star.

        Uses the context assembled from `batch` (theta, cond, and the
        conditioning observables). Returns physical-unit samples of shape
        (n_stars, num_samples, target_dim), with the track added back
        when the run uses one.
        """
        self.eval()
        norm_batch = self._normalize(batch.to(self.device))
        context = self._context(norm_batch)
        if self.is_continuous:
            samples = self.flow.sample(context, num_samples, steps=steps)
        else:
            samples = self.flow(context).sample((num_samples,)).transpose(0, 1)

        # The stage chain -- the track shear above all -- is defined on
        # the full set of columns, so the conditioned ones have to be
        # put back before inverting.  That also fixes which phi1 the
        # track is undone against: the sampled one when phi1 is
        # modelled, the conditioned one when it is not.
        shape = samples.shape
        flat = samples.reshape(-1, shape[-1])
        if self.x_context_idx.numel():
            full = torch.empty(flat.shape[0], len(self.x_all_labels),
                               dtype=flat.dtype, device=flat.device)
            full[:, self.x_target_idx] = flat
            held = norm_batch.x[:, self.x_context_idx]
            full[:, self.x_context_idx] = held.repeat_interleave(
                num_samples, dim=0)
            flat = self._denormalize_x(full)[:, self.x_target_idx]
        else:
            flat = self._denormalize_x(flat)
        return flat.reshape(shape)
