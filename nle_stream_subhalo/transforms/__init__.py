"""Observation pipeline components for per-star NLE."""

from .basic import StarBatch
from .uncertainty import UncertaintySampler
from .pipeline import Compose, build_transformation, compute_field_norm
from .track import StreamTrack, load_track_dict
from .stages import (
    AffineUnit,
    MarginalUniform,
    STAGE_TYPES,
    StagePipeline,
    fit_x_stages,
)

__all__ = [
    'StarBatch',
    'UncertaintySampler',
    'Compose',
    'build_transformation',
    'compute_field_norm',
    'StreamTrack',
    'load_track_dict',
    'AffineUnit',
    'MarginalUniform',
    'STAGE_TYPES',
    'StagePipeline',
    'fit_x_stages',
]
