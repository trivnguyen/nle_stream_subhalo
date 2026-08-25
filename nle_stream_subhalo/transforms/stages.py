"""Invertible stages applied to `x` before the flow sees it.

The flow does not model the physical coordinates directly. `x` first goes
through a fixed chain of invertible maps, and the flow models the result.
Each stage is

  * **invertible**, so flow samples map back to physical units;
  * **volume aware** -- it reports `log_det`, needed to compare
    representations (see `TRANSFORMS.md`), though not to run a posterior:
    every stage here is independent of `theta`, so the Jacobian cancels in
    posterior ratios and only shifts log Z;
  * **fitted once**, on unperturbed streams (`TrackShear`) or on the
    prior-pooled training set (the rest), never per-`theta`.

`StreamTrack` in `track.py` implements the same interface, so it chains
here like any other stage.

Serialization mirrors `norm_dict`: `fit_x_stages` returns plain lists, the
model rebuilds buffers from them, and everything rides along in
`save_hyperparameters` and the JSON config snapshot.
"""

import torch
import torch.nn as nn

from .track import StreamTrack


class AffineUnit(nn.Module):
    """Midrange/half-range onto [-1, 1], per coordinate.

    What the pipeline did before stages existed: exact on the training
    range, but it cannot bound anything it was not fitted on, and it
    leaves the bulk as a narrow spike because the scale is set by the
    extremes.
    """

    kind = 'affine_unit'

    def __init__(self, state: dict):
        super().__init__()
        self.register_buffer(
            'loc', torch.tensor(state['loc'], dtype=torch.float32))
        self.register_buffer(
            'scale', torch.tensor(state['scale'], dtype=torch.float32))

    @staticmethod
    def fit(x: torch.Tensor) -> dict:
        low, high = x.min(0)[0], x.max(0)[0]
        return {
            'kind': AffineUnit.kind,
            'loc': ((high + low) / 2).tolist(),
            'scale': ((high - low) / 2).clamp_min(1e-6).tolist(),
        }

    def forward(self, x):
        return (x - self.loc) / self.scale

    def inverse(self, u):
        return u * self.scale + self.loc

    def log_det(self, x):
        return -torch.log(self.scale).sum().expand(x.shape[0])

    def extra_repr(self):
        return f'features={self.loc.numel()}'


class MarginalUniform(nn.Module):
    """Monotone quantile map onto [-1, 1], per coordinate.

    u_i = 2 F_i(x_i) - 1, with F_i the empirical CDF over the
    prior-pooled training stars, realized as `n_knots` quantiles with
    linear interpolation onto an even grid. Monotone by construction --
    quantiles are sorted -- so invertibility needs no constraint.

    Every interval carries equal probability mass but a different physical
    width, so dense regions are stretched and sparse ones compressed, and
    the marginal comes out flat. That is the shape a spline flow
    represents with the fewest bins: zuko's NSF learns bin widths through
    a softmax, and a spiked marginal forces nearly all the mass into a
    tiny sub-interval before anything can be modelled.

    It is diagonal, so it fixes marginals only, never the joint -- put it
    after `TrackShear`, which is what removes the phi1 dependence.

    Beyond the outer knots it continues with the end slope, staying a
    bijection on all of R rather than saturating, so an unseen extreme
    degrades linearly instead of collapsing onto the boundary.
    """

    kind = 'marginal_uniform'

    def __init__(self, state: dict):
        super().__init__()
        knots = torch.tensor(state['knots'], dtype=torch.float32)
        self.register_buffer('knots', knots)                     # (K, D)
        grid = torch.linspace(-1, 1, knots.shape[0])
        self.register_buffer('grid', grid)                       # (K,)
        self.register_buffer(
            'slope',
            (grid[1:] - grid[:-1]).unsqueeze(-1) / (knots[1:] - knots[:-1]))

    @staticmethod
    def fit(x: torch.Tensor, n_knots: int = 256) -> dict:
        """Measure the quantile knots, one column at a time.

        Sorting rather than `torch.quantile`: that caps at 2**24 input
        elements and a real training set is an order of magnitude past it
        (10 dataset files is ~18M stars x 6 columns = 108M). Sorting each
        column separately also keeps the peak allocation to one column.
        """
        n_stars = x.shape[0]
        position = torch.linspace(0, 1, n_knots, dtype=torch.float64) \
            * (n_stars - 1)
        low = position.floor().long()
        high = position.ceil().long()
        weight = (position - low).to(x.dtype)

        knots = torch.stack([
            torch.lerp(ordered[low], ordered[high], weight)
            for ordered in (torch.sort(x[:, j]).values
                            for j in range(x.shape[1]))
        ], dim=1)
        # Reason: tied values give a zero-width interval, whose slope is
        # infinite and whose inverse does not exist. Nudge them apart by a
        # fraction of the range, which is far below the measurement noise.
        floor = 1e-6 * (knots[-1] - knots[0]).clamp_min(1e-6)
        for k in range(1, n_knots):
            knots[k] = torch.maximum(knots[k], knots[k - 1] + floor)
        return {'kind': MarginalUniform.kind, 'knots': knots.tolist()}

    def _interval(self, values, edges):
        """Per-column interval index of `values` within `edges`."""
        return torch.stack([
            torch.searchsorted(edges[:, j].contiguous() if edges.dim() == 2
                               else edges.contiguous(),
                               values[:, j].contiguous(), right=True) - 1
            for j in range(values.shape[1])], dim=1
        ).clamp(0, self.knots.shape[0] - 2)

    def _grid_2d(self):
        return self.grid.unsqueeze(-1).expand_as(self.knots)

    def forward(self, x):
        idx = self._interval(x, self.knots)
        return (torch.gather(self._grid_2d(), 0, idx)
                + torch.gather(self.slope, 0, idx)
                * (x - torch.gather(self.knots, 0, idx)))

    def inverse(self, u):
        """Inverse of `forward`.

        Exact in real arithmetic, but the round trip is limited by how
        much `u` can resolve: the outermost knot intervals are wide (the
        sparse tail of `vr` spans ~900 km/s in one bin), so a single
        float32 ulp of `u` near +-1 is worth ~7e-3 km/s there. That is
        ~1e-5 of the coordinate's range and far below the measurement
        noise, but it is a floor no amount of arithmetic precision moves.
        """
        idx = self._interval(u, self.grid)
        return (torch.gather(self.knots, 0, idx)
                + (u - torch.gather(self._grid_2d(), 0, idx))
                / torch.gather(self.slope, 0, idx))

    def log_det(self, x):
        return torch.log(
            torch.gather(self.slope, 0, self._interval(x, self.knots))).sum(-1)

    def extra_repr(self):
        return f'knots={self.knots.shape[0]}, features={self.knots.shape[1]}'


# `kind` -> class, for rebuilding a chain from its serialized form.
STAGE_TYPES = {
    StreamTrack.kind: StreamTrack,
    AffineUnit.kind: AffineUnit,
    MarginalUniform.kind: MarginalUniform,
}


class StagePipeline(nn.Module):
    """A chain of stages, applied to `x` in order.

    Holds them in a `ModuleList` so their buffers ride along in
    `state_dict` and follow the model across devices.
    """

    def __init__(self, stage_dicts, x_labels=None):
        super().__init__()
        stages = []
        for state in stage_dicts:
            kind = state['kind']
            if kind not in STAGE_TYPES:
                raise ValueError(
                    f'unknown stage {kind!r}; known: '
                    f'{sorted(STAGE_TYPES)}')
            cls = STAGE_TYPES[kind]
            # StreamTrack needs the column order, since it stores its
            # coordinates by name.
            stages.append(cls(state, x_labels) if cls is StreamTrack
                          else cls(state))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def inverse(self, u):
        for stage in reversed(self.stages):
            u = stage.inverse(u)
        return u

    def log_det(self, x):
        total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for stage in self.stages:
            total = total + stage.log_det(x)
            x = stage(x)
        return total

    def describe(self) -> str:
        return ' -> '.join(stage.kind for stage in self.stages)


def fit_x_stages(
    stage_names, x: torch.Tensor, track_dict: dict = None, x_labels=None,
    n_knots: int = 256,
) -> list:
    """Fit a stage chain on training stars; return its serialized form.

    Each stage is fitted on the output of the ones before it, so a tail
    stage is calibrated for what actually reaches it.

    Args:
        stage_names: Stage `kind` strings, in order.
        x: (n, D) physical stars, already through the observation
            pipeline -- the stats have to describe what the model sees.
        track_dict: Required if `track_shear` is among the stages.
        x_labels: Column order of `x`, required with `track_shear`.
        n_knots: Quantile knots for `marginal_uniform`.

    Returns:
        List of per-stage state dicts, each carrying its own `kind`.
    """
    fitted = []
    current = x
    for name in stage_names:
        if name == StreamTrack.kind:
            if track_dict is None:
                raise ValueError(
                    f"stage {name!r} needs a track: set config.track_path")
            state = dict(track_dict, kind=name)
            stage = StreamTrack(state, x_labels)
        elif name == AffineUnit.kind:
            state = AffineUnit.fit(current)
            stage = AffineUnit(state)
        elif name == MarginalUniform.kind:
            state = MarginalUniform.fit(current, n_knots=n_knots)
            stage = MarginalUniform(state)
        else:
            raise ValueError(
                f'unknown stage {name!r}; known: {sorted(STAGE_TYPES)}')

        fitted.append(state)
        with torch.no_grad():
            current = stage.to(current.device)(current)

    return fitted
