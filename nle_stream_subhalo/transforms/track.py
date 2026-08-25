"""Unperturbed stream track, evaluated in torch.

Consumes the `.npz` produced by `stream_sims.track` (in the
`sbi_stream_pipeline` repo) -- the track is a property of the simulator,
so it is fitted there and only read here.

Every per-star coordinate is a tight function of `phi1`, and the
perturber signal is the departure from that relation. Subtracting the
track leaves the flow modelling only the departure:

    Delta y = y - mu(phi1)

That map is a shear -- its Jacobian is lower triangular with a unit
diagonal -- so it is **volume preserving** and `log_prob` needs no
correction term. It also commutes with the measurement model, because
`phi1` carries no uncertainty: mu(phi1) is the same before and after
noise is added, and the recorded sigmas are unchanged. If `phi1` ever
gains an uncertainty, both of those stop holding and the projection has
to move after the noise, using the observed `phi1`.
"""

import json

import numpy as np
import torch
import torch.nn as nn


def load_track_dict(path: str) -> dict:
    """Read a `stream_sims.track` .npz into plain lists.

    Lists rather than arrays so the result survives
    `save_hyperparameters` and the JSON config snapshot unchanged, the
    same way `norm_dict` does.
    """
    with np.load(path, allow_pickle=False) as data:
        return {
            'knots': data['knots'].tolist(),
            'coef': data['coef'].tolist(),
            'end_value': data['end_value'].tolist(),
            'end_slope': data['end_slope'].tolist(),
            'coords': [str(name) for name in data['coords']],
            'meta': json.loads(str(data['meta'])),
        }


class StreamTrack(nn.Module):
    """Piecewise-cubic track of an unperturbed stream against `phi1`.

    An `nn.Module` so its arrays ride along in `state_dict` and follow
    the model across devices; it holds no parameters and never trains.

    Implements the stage interface in `stages.py` (`forward`, `inverse`,
    `log_det`), so it chains with the normalization stages.

    Parameters
    ----------
    track_dict : dict
        From `load_track_dict`.
    x_labels : sequence of str
        Column order of `x`. Needed because the track stores coordinates
        by name and the simulator's own order
        (phi1, phi2, dist, pm1, pm2, vr) is not the order the configs
        use (phi1, phi2, vr, pm1, pm2, dist).
    """

    kind = 'track_shear'

    def __init__(self, track_dict: dict, x_labels):
        super().__init__()
        x_labels = list(x_labels)
        if 'phi1' not in x_labels:
            raise ValueError(
                "x_labels has no 'phi1'; the track is indexed by it")

        coords = list(track_dict['coords'])
        missing = [name for name in x_labels
                   if name != 'phi1' and name not in coords]
        if missing:
            raise ValueError(
                f'track has no entry for {missing}; it was fitted on '
                f'{coords}. Refit it with --coords covering every '
                f'modelled coordinate.')

        # Keep only the tracked coordinates that x actually carries, in
        # the track's own row order, alongside where each one lives in x.
        used = [(index, name) for index, name in enumerate(coords)
                if name in x_labels]
        rows = [index for index, _ in used]

        self.coords = [name for _, name in used]
        self.x_labels = x_labels
        self.meta = dict(track_dict.get('meta', {}))

        as_tensor = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer('knots', as_tensor(track_dict['knots']))
        self.register_buffer('coef', as_tensor(track_dict['coef'])[rows])
        self.register_buffer(
            'end_value', as_tensor(track_dict['end_value'])[rows])
        self.register_buffer(
            'end_slope', as_tensor(track_dict['end_slope'])[rows])
        self.register_buffer(
            'columns',
            torch.tensor([x_labels.index(name) for name in self.coords],
                         dtype=torch.long))
        self.register_buffer(
            'phi1_column', torch.tensor(x_labels.index('phi1'),
                                        dtype=torch.long))

    def extra_repr(self) -> str:
        return (f'coords={self.coords}, bins={self.knots.numel()}, '
                f'phi1=[{self.knots[0]:.1f}, {self.knots[-1]:.1f}]')

    def evaluate(self, phi1: torch.Tensor) -> torch.Tensor:
        """Track values at `phi1`, as (n, n_coords) in `self.coords` order.

        Beyond the knots the track continues along its endpoint slope.
        Holding it constant there instead would turn the coordinates'
        real curvature into a fake offset, in the unperturbed stream as
        much as in a perturbed one.
        """
        low, high = self.knots[0], self.knots[-1]
        clamped = phi1.clamp(low, high)
        interval = (
            torch.searchsorted(self.knots, clamped.contiguous(), right=True)
            - 1).clamp(0, self.knots.numel() - 2)

        delta = (clamped - self.knots[interval]).unsqueeze(-1)
        c = self.coef[:, :, interval].permute(2, 0, 1)   # (n, n_coords, 4)
        value = (((c[..., 0] * delta + c[..., 1]) * delta + c[..., 2])
                 * delta + c[..., 3])

        value = torch.where(
            (phi1 < low).unsqueeze(-1),
            self.end_value[:, 0] + self.end_slope[:, 0]
            * (phi1 - low).unsqueeze(-1),
            value)
        return torch.where(
            (phi1 > high).unsqueeze(-1),
            self.end_value[:, 1] + self.end_slope[:, 1]
            * (phi1 - high).unsqueeze(-1),
            value)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Replace each tracked column of `x` by its offset from the track."""
        out = x.clone()
        out[:, self.columns] = (x[:, self.columns]
                                - self.evaluate(x[:, self.phi1_column]))
        return out

    def unproject(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of `project`.

        Uses each row's own `phi1`. For flow samples that is the
        *sampled* `phi1`, not the one conditioned on -- `phi1` is one of
        the modelled coordinates, so the drawn value is the one the
        offsets belong to.
        """
        out = x.clone()
        out[:, self.columns] = (x[:, self.columns]
                                + self.evaluate(x[:, self.phi1_column]))
        return out

    # -- stage interface ----------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return self.unproject(u)

    def log_det(self, x: torch.Tensor) -> torch.Tensor:
        """Zero: the shear's Jacobian is triangular with a unit diagonal."""
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
