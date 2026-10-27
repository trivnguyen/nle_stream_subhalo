"""Derived training labels, borrowed from the simulator that made them.

A dataset stores the parameters its prior drew.  Some of those are a
poor representation for a network -- notably the impact azimuths,
which are drawn on [0, 360) so that 359 and 1 sit at opposite ends
of the input range despite being two degrees apart.  `gd1_nle`
defines the alternatives in `sim.labels`, as components rather than
polar pairs, and this module makes them available to a config.

The definitions live in `gd1_nle` and not here on purpose: they have
to match the geometry `sim.perturber.subhalo_w_at_impact` applies,
and that repository has the test which pins them against it.  A copy
here would be free to drift, and a network conditioned on a geometry
the simulator did not use fails silently.

`gd1_nle` is not installable, so it is found by, in order:

    1. `$GD1_NLE_ROOT`
    2. a `gd1_nle` directory beside this repository

Nothing is required unless a config actually names a derived label.
A run whose `theta_labels` are all stored proceeds without `gd1_nle`
on the machine at all.
"""

import os
import sys
from pathlib import Path


def _candidate_roots():
    """Where `gd1_nle` might be, best guess first.

    Returns:
        List of Path.
    """
    roots = []
    env = os.environ.get('GD1_NLE_ROOT')
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve().parent.parent
    roots.append(here.parent / 'gd1_nle')
    return roots


def load_sim_labels():
    """Import `sim.labels` from wherever `gd1_nle` lives.

    Returns:
        The imported module.

    Raises:
        ImportError: If no candidate root holds it, naming the
            places tried and how to fix it.
    """
    try:
        from sim import labels
        return labels
    except ImportError:
        pass
    tried = []
    for root in _candidate_roots():
        tried.append(str(root))
        if not (root / 'sim' / 'labels.py').is_file():
            continue
        sys.path.insert(0, str(root))
        try:
            from sim import labels
            return labels
        except ImportError:
            sys.path.pop(0)
    raise ImportError(
        'could not import sim.labels from gd1_nle, which defines the '
        'derived labels. Tried: ' + ', '.join(tried)
        + '. Set GD1_NLE_ROOT to the gd1_nle checkout, or put it '
          'beside this repository.')


def expand(graph_feats, wanted=()):
    """Add every derivable label to a dataset's graph features.

    Args:
        graph_feats: Mapping of stored label name -> (n_streams,)
            array, as `datasets.read_datasets` returns it.
        wanted: The labels this run asks for.  Used only to decide
            whether a missing `gd1_nle` is fatal: a run naming only
            stored labels does not need it.

    Returns:
        The mapping, with the derived labels added.  Returned
        unchanged when nothing derived is needed and `gd1_nle` is
        not available.

    Raises:
        ImportError: If a wanted label is not stored and `gd1_nle`
            cannot be found to derive it.
    """
    try:
        labels = load_sim_labels()
    except ImportError:
        missing = [n for n in wanted if n not in graph_feats]
        if missing:
            raise
        return graph_feats
    return labels.expand(graph_feats)
