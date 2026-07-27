"""Core per-star container for NLE.

The container holds physical observable fields only. Feature extraction,
the modeled-vs-conditioned split, and normalization are the NLE model's
job, not the data pipeline's.
"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class StarBatch:
    """A batch of stream stars for per-star NLE.

    Each row is one star. Fields are physical (unnormalized). `xerr` is the
    measurement sigma set by `UncertaintySampler` (training) or supplied
    from real data (inference), and is always used as conditioning context.

    Attributes:
        x: Observed features, (B, n_features).
        xerr: Measurement sigma, (B, n_err), one column per feature that
            actually has an uncertainty -- NOT one per feature. Columns
            follow the order the `UncertaintySampler`s run, i.e. the order
            of `uncertainty_args`. None until set.
        theta: Per-star inferred parameters, (B, n_params) -- the host
            stream's parameters, repeated for each of its stars. None on
            real observations.
        cond: Per-star measured conditioning variables, (B, n_cond).
            Distinct from `theta`: measured (fixed or MC-sampled at
            inference), not inferred. None when unused.
    """

    x: Tensor
    xerr: Tensor | None = None
    theta: Tensor | None = None
    cond: Tensor | None = None

    def clone(self) -> "StarBatch":
        """Return a deep copy so transforms never mutate their input."""
        return StarBatch(
            x=self.x.clone(),
            xerr=(None if self.xerr is None
                  else self.xerr.clone()),
            theta=None if self.theta is None else self.theta.clone(),
            cond=None if self.cond is None else self.cond.clone(),
        )

    def to(self, device) -> "StarBatch":
        """Move all present tensor fields to `device`."""
        return StarBatch(
            x=self.x.to(device),
            xerr=(None if self.xerr is None
                  else self.xerr.to(device)),
            theta=None if self.theta is None else self.theta.to(device),
            cond=None if self.cond is None else self.cond.to(device),
        )
