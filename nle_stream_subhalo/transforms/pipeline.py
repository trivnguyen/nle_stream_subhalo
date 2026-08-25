"""Compose the per-star NLE observation pipeline and measure field stats."""

from tqdm import tqdm
import torch

from .basic import StarBatch
from .stages import fit_x_stages
from .uncertainty import UncertaintySampler


class Compose:
    """Chain transforms, threading a StarBatch through each in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, batch: StarBatch) -> StarBatch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


def build_transformation(
    apply_uncertainty: bool = False,
    uncertainty_args=None,
) -> Compose:
    """
    Build the observation pipeline (measurement uncertainty).

    This is the stochastic, per-batch augmentation the NLE model applies in
    `_prepare_batch` during training. It produces physical fields only;
    normalization and the modeled-vs-conditioned split happen in the model.

    `uncertainty_args` accepts either a single dict or a list of dicts to
    chain multiple `UncertaintySampler` transforms, each perturbing its own
    feature (`feature_idx`) with its own `distribution_type` and parameters.
    `batch.xerr` gets one column per entry, in this order; features not
    listed here have no uncertainty and no column.
    """
    transforms = []

    if apply_uncertainty:
        if uncertainty_args is None:
            raise ValueError(
                '`uncertainty_args` must be provided when '
                '`apply_uncertainty` is True.')
        # a single mapping (dict/ConfigDict) or a list of them, one per
        # perturbed velocity component
        if not isinstance(uncertainty_args, (list, tuple)):
            uncertainty_args = [uncertainty_args]
        for args in uncertainty_args:
            transforms.append(UncertaintySampler(**args))

    return Compose(transforms)


def compute_field_norm(
    batch: StarBatch, batch_size: int = 4096, track=None,
    x_stages=None, track_dict=None, x_labels=None, n_knots: int = 256,
    max_stars: int = None, seed: int = 0,
    **pre_transform_kwargs
) -> dict:
    """Measure per-field normalization stats for the NLE model.

    Streams `batch` through the observation pipeline built from
    `pre_transform_kwargs`, chunk by chunk, then computes per-component
    location/scale for `x` and `xerr`. `theta` stats come from its
    midrange/half-range (mapped to ~[-1, 1]). Scales are floored at a small
    epsilon so constant components don't divide by zero.

    Args:
        batch: Raw StarBatch with `x` (and `theta`) set.
        batch_size: Number of stars per streaming chunk.
        max_stars: Measure the `x`/`xerr` stats on at most this many stars,
            drawn uniformly at random. These stats are approximate however
            many stars go in -- the pipeline resamples the measurement
            noise on every call, so no finite sample pins them down -- and
            the cost is not: the full training `x` is copied through the
            pipeline, concatenated, and copied again per fitted stage, so
            223M stars run to ~30 GB of transient allocation for numbers a
            few million already resolve. `theta` and `cond` stats are
            always exact; they are min/max over an existing tensor and
            allocate nothing.
        seed: Draws the subsample, so a rerun measures the same stats.
        track: Optional `transforms.StreamTrack`. When given, the `x`
            stats are measured on track offsets rather than on raw
            coordinates -- they have to describe what the model actually
            normalizes, and the model projects before it scales.
            Ignored when `x_stages` is given, since the track is then one
            of the stages.
        x_stages: Optional sequence of stage `kind` strings, e.g.
            `('track_shear', 'marginal_uniform')`. When given, the `x`
            side of the returned dict is an `x_stages` chain fitted by
            `stages.fit_x_stages` instead of `x_loc`/`x_scale`.
        track_dict: Required if `'track_shear'` is among `x_stages`.
        x_labels: Column order of `x`, required with `'track_shear'`.
        n_knots: Quantile knots for a `marginal_uniform` stage.
        **pre_transform_kwargs: Same kwargs as `build_transformation`.

    Returns:
        Dict with `<field>_loc`/`<field>_scale` lists for x, theta, and cond
        (cond only if `batch.cond` is set), plus xerr if the pipeline
        samples any uncertainty. `xerr` stats have one entry per perturbed
        feature, so they are generally shorter than the `x` stats.
    """
    pipeline = build_transformation(**pre_transform_kwargs)

    source = batch.x
    n = source.shape[0]
    if max_stars is not None and n > max_stars:
        # Reason: with replacement, so this stays O(max_stars) -- a
        # randperm over 223M rows would itself allocate 1.8 GB. At these
        # ratios repeats are ~1 in 10^5 and move no statistic.
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randint(n, (max_stars,), generator=generator)
        source = source[rows.sort().values]   # sorted: one ordered gather
        n = max_stars
        print(f'Field normalization on {n:,} of {batch.x.shape[0]:,} stars')

    x_parts, xerr_parts = [], []
    for start in tqdm(range(0, n, batch_size), desc='Computing field normalization'):
        stop = start + batch_size
        chunk = StarBatch(x=source[start:stop])
        out = pipeline(chunk)
        # With stages the chain is fitted on the observed coordinates and
        # applies the track itself, so nothing is projected here. Without
        # them, projection and the measurement model commute (phi1 is
        # noiseless), but this mirrors the model's own order anyway rather
        # than relying on that.
        if x_stages is not None or track is None:
            x_parts.append(out.x)
        else:
            x_parts.append(track.project(out.x))
        if out.xerr is not None:
            xerr_parts.append(out.xerr)

    x_all = torch.cat(x_parts, dim=0)

    _floor = lambda scale: scale.clamp_min(1e-6)  # avoid divide-by-zero

    if x_stages is not None:
        norm = {'x_stages': fit_x_stages(
            x_stages, x_all, track_dict=track_dict, x_labels=x_labels,
            n_knots=n_knots)}
    else:
        x_min, x_max = x_all.min(dim=0)[0], x_all.max(dim=0)[0]
        norm = {
            'x_loc': ((x_max + x_min) / 2).tolist(),
            'x_scale': _floor((x_max - x_min) / 2).tolist(),
        }

    if xerr_parts:
        err_all = torch.cat(xerr_parts, dim=0)
        err_min, err_max = err_all.min(dim=0)[0], err_all.max(dim=0)[0]
        norm['xerr_loc'] = ((err_max + err_min) / 2).tolist()
        norm['xerr_scale'] = _floor((err_max - err_min) / 2).tolist()

    # theta and cond both normalized to ~[-1, 1] via midrange/half-range.
    # x and xerr are deliberately not in this loop: their stats come from
    # the streamed chunks above, which carry the measurement noise and the
    # track projection. Reading them off `batch` here would take the raw,
    # unobserved, unprojected values instead.
    for name in ('theta', 'cond'):
        field = getattr(batch, name)
        if field is None:
            continue
        f_min = field.min(dim=0)[0]
        f_max = field.max(dim=0)[0]
        norm[f'{name}_loc'] = ((f_max + f_min) / 2).tolist()
        norm[f'{name}_scale'] = _floor((f_max - f_min) / 2).tolist()

    return norm
