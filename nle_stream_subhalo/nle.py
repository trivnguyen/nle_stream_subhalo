"""Neural Likelihood Estimation (NLE) module for per-star stream data.

The model learns the single-star likelihood p(observables | theta, cond)
with a conditional normalizing flow, and can sample observables from it.
`theta` are the perturber parameters of the star's host stream, broadcast
to each of its member stars. It deliberately does ONLY what it is trained
on (per-star density + sampling) -- assumptions like i.i.d. stars, priors,
fixing/sampling `cond`, and MCMC live outside the model.

"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko

from .flow_matching import FlowMatching
from .transforms import StarBatch
from .utils import MLP, configure_optimizers, get_activation


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
    ):
        super().__init__()
        self.norm_dict = norm_dict
        self.flows_args = flows_args or {}
        self.context_embedding_args = context_embedding_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.val_nll_batches = val_nll_batches

        self.save_hyperparameters(ignore=['pre_transforms'])

        self._register_norm(norm_dict)
        self._setup_model()

    # -- setup --------------------------------------------------------------

    def _register_norm(self, norm_dict: dict):
        self.has_cond = 'cond_loc' in norm_dict
        fields = ['x', 'xerr', 'theta']
        if self.has_cond:
            fields.append('cond')
        for field in fields:
            self.register_buffer(
                f'{field}_loc',
                torch.tensor(norm_dict[f'{field}_loc'], dtype=torch.float32))
            self.register_buffer(
                f'{field}_scale',
                torch.tensor(norm_dict[f'{field}_scale'], dtype=torch.float32))

    def _setup_model(self):
        """Set up the field layout, optional context MLP, and flow."""
        dims = {f: len(self.norm_dict[f'{f}_loc'])
                for f in ('x', 'xerr', 'theta')}
        if self.has_cond:
            dims['cond'] = len(self.norm_dict['cond_loc'])

        context = ['theta'] + (['cond'] if self.has_cond else []) + ['xerr']
        self.target_fields = ['x', ]
        self.context_fields = context

        target_size = sum(dims[f] for f in self.target_fields)
        context_size = sum(dims[f] for f in self.context_fields)

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
              f"{','.join(self.context_fields)})")
        print(f"[NLE] Flow built with"
              f" {sum(p.numel() for p in self.flow.parameters()):,} parameters")

    # -- normalization / assembly ------------------------------------------

    def _normalize(self, batch: StarBatch) -> StarBatch:
        """Normalize every present field to the training scale."""
        def norm(field, value):
            if value is None:
                return None
            return (value - getattr(self, f'{field}_loc')) / getattr(
                self, f'{field}_scale')

        return StarBatch(
            x=norm('x', batch.x),
            xerr=norm('xerr', batch.xerr),
            theta=norm('theta', batch.theta),
            cond=norm('cond', batch.cond) if self.has_cond else None,
        )

    def _assemble(self, batch: StarBatch, fields) -> torch.Tensor:
        cols = []
        for field in fields:
            value = getattr(batch, field)
            if value is None:
                raise ValueError(f'batch.{field} is required but was None')
            cols.append(value)
        return torch.cat(cols, dim=1)

    def _target_context(self, norm_batch: StarBatch):
        """Split a normalized batch into (target, embedded context)."""
        target = self._assemble(norm_batch, self.target_fields)
        context = self.context_embedding(
            self._assemble(norm_batch, self.context_fields))
        return target, context

    def _flow_log_prob(self, norm_batch: StarBatch) -> torch.Tensor:
        """Per-star log p(target | context) for a normalized StarBatch."""
        target, context = self._target_context(norm_batch)
        if self.is_continuous:
            return self.flow.log_prob(target, context)
        return self.flow(context).log_prob(target)

    # -- training -----------------------------------------------------------

    def _prepare_batch(self, batch: StarBatch) -> StarBatch:
        """Augment (add measurement noise) then normalize a raw batch."""
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)
        return self._normalize(batch)

    def _loss(self, norm_batch: StarBatch) -> torch.Tensor:
        """Training objective: velocity regression, or the NLL."""
        if self.is_continuous:
            return self.flow.loss(*self._target_context(norm_batch))
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
        """Initialize optimizer and LR scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    # -- inference (observed data in physical units) -----------------------

    @torch.no_grad()
    def log_prob(self, batch: StarBatch) -> torch.Tensor:
        """Per-star log p(observables | theta, cond) for a physical batch.

        `batch` holds the observed per-star fields (`x`, `xerr`) plus the
        candidate `theta` and `cond`, all broadcast to one row per star by
        the caller. No augmentation is applied. Returns a (n_stars,)
        tensor; summing over stars, priors, and MCMC are the caller's job.
        """
        self.eval()
        return self._flow_log_prob(self._normalize(batch.to(self.device)))

    @torch.no_grad()
    def sample(self, batch: StarBatch, num_samples: int = 1) -> torch.Tensor:
        """Sample observables from p(target | context) for each star.

        Uses the context assembled from `batch` (theta, cond, and the
        conditioning observables). Returns physical-unit samples of shape
        (n_stars, num_samples, target_dim).
        """
        self.eval()
        norm_batch = self._normalize(batch.to(self.device))
        context = self.context_embedding(
            self._assemble(norm_batch, self.context_fields))
        if self.is_continuous:
            samples = self.flow.sample(context, num_samples)
        else:
            samples = self.flow(context).sample((num_samples,)).transpose(0, 1)

        loc = torch.cat([getattr(self, f'{f}_loc') for f in self.target_fields])
        scale = torch.cat(
            [getattr(self, f'{f}_scale') for f in self.target_fields])
        return samples * scale + loc
