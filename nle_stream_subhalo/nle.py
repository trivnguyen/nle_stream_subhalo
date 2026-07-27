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

from .transforms import StarBatch
from .utils import MLP, configure_optimizers, get_activation


def build_flow(
    features: int,
    context: int,
    flow_type: str = 'spline',
    num_transforms: int = 3,
    hidden_features=(64, 64),
    num_bins: int = 8,
    activation: str = None,
):
    """Build a conditional zuko flow for p(target | context)."""
    kwargs = dict(transforms=num_transforms, hidden_features=hidden_features)
    if activation is not None:
        kwargs['activation'] = get_activation(activation, return_instance=False)

    if flow_type == 'spline':
        return zuko.flows.NSF(features, context, bins=num_bins, **kwargs)
    if flow_type == 'maf':
        return zuko.flows.MAF(features, context, **kwargs)
    raise ValueError(f"Unknown flow_type: {flow_type!r} (use 'spline'/'maf')")


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
    """

    def __init__(
        self,
        norm_dict: dict,
        flows_args: dict = None,
        context_embedding_args: dict = None,
        optimizer_args: dict = None,
        scheduler_args: dict = None,
        pre_transforms=None,
    ):
        super().__init__()
        self.norm_dict = norm_dict
        self.flows_args = flows_args or {}
        self.context_embedding_args = context_embedding_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms

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

    def _flow_log_prob(self, norm_batch: StarBatch) -> torch.Tensor:
        """Per-star log p(target | context) for a normalized StarBatch."""
        target = self._assemble(norm_batch, self.target_fields)
        context = self.context_embedding(
            self._assemble(norm_batch, self.context_fields))
        return self.flow(context).log_prob(target)

    # -- training -----------------------------------------------------------

    def _prepare_batch(self, batch: StarBatch) -> StarBatch:
        """Augment (add measurement noise) then normalize a raw batch."""
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)
        return self._normalize(batch)

    def training_step(self, batch, batch_idx):
        loss = -self._flow_log_prob(self._prepare_batch(batch)).mean()
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch.x.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = -self._flow_log_prob(self._prepare_batch(batch)).mean()
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
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
        samples = self.flow(context).sample((num_samples,))

        loc = torch.cat([getattr(self, f'{f}_loc') for f in self.target_fields])
        scale = torch.cat(
            [getattr(self, f'{f}_scale') for f in self.target_fields])
        samples = samples * scale + loc
        return samples.transpose(0, 1)
