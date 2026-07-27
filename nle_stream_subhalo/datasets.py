"""HDF5 I/O and per-star dataloaders for NLE.

Each training example is a single star: the model learns the likelihood
p(x | theta), where theta is the star's host-stream parameters. Stars are
kept flat (no per-stream grouping) except when splitting train/val, where
all stars of a stream are forced into the same set so the two never share a
stream.
"""

import os
import warnings
import h5py

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from tqdm import tqdm

from .transforms import StarBatch, compute_field_norm


def read_graph_dataset(path, features_list=None, concat=False, to_array=True):
    """Read a graph dataset from an HDF5 file.

    Parameters
    ----------
    path          : str   path to the HDF5 file
    features_list : list  features to read; reads all if empty / None
    concat        : bool  if True, concatenate all node features into one array
    to_array      : bool  if True (and not concat), wrap node-feature lists in
                          a numpy object array

    Returns
    -------
    node_features  : dict
    graph_features : dict
    headers        : dict
    """
    if features_list is None:
        features_list = []

    with h5py.File(path, 'r') as f:
        headers = dict(f.attrs)

        if len(features_list) == 0:
            features_list = headers['all_features']

        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                graph_features[key] = f[key][:]

    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, graph_features, headers


def read_datasets(
    root, name, num_datasets=100, init=0, is_directory=True, concat=True, ext='.h5'
):
    """Read and concatenate multiple HDF5 dataset files."""
    if ext[0] != '.':
        ext = '.' + ext

    if is_directory:
        node_feats, graph_feats = {}, {}

        for i in tqdm(range(init, init + num_datasets)):
            data_path = os.path.join(root, name, f'data.{i}{ext}')
            if not os.path.exists(data_path):
                print(f'Warning: {data_path} does not exist. Skipping...')
                continue
            nodes, graphs, _ = read_graph_dataset(data_path, concat=concat)

            for k in nodes:
                node_feats.setdefault(k, []).append(nodes[k])
            for k in graphs:
                graph_feats.setdefault(k, []).append(graphs[k])

        if not node_feats or not graph_feats:
            raise ValueError(
                f'No valid datasets found in {root}/{name} with '
                f'init={init} and num_datasets={num_datasets}.')

        node_feats = {k: np.concatenate(v) for k, v in node_feats.items()}
        graph_feats = {k: np.concatenate(v) for k, v in graph_feats.items()}
    else:
        data_path = os.path.join(root, name + ext)
        node_feats, graph_feats, _ = read_graph_dataset(
            data_path, concat=concat)

    return node_feats, graph_feats


# ---------------------------------------------------------------------------
# Per-star tensor assembly
# ---------------------------------------------------------------------------

class StarDataset(Dataset):
    """Flat per-star dataset, indexed a whole batch at a time.

    `__getitem__` takes the index list a `BatchSampler` produces (see
    `_build_loader`) rather than a single star, so assembling a batch is
    one advanced-indexing op per field instead of B per-star reads plus a
    `torch.stack`. Both x and theta already live in memory as dense
    tensors, so there is nothing to gain from going star by star.
    """

    def __init__(
        self, x: torch.Tensor, theta: torch.Tensor, cond: torch.Tensor = None):
        self.x = x
        self.theta = theta
        self.cond = cond

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> StarBatch:
        return StarBatch(
            x=self.x[idx],
            theta=self.theta[idx],
            cond=None if self.cond is None else self.cond[idx],
        )


def _passthrough(batch: StarBatch) -> StarBatch:
    """Collate for batch-indexed loaders: the batch is already a StarBatch."""
    return batch


def _build_loader(
    dataset: StarDataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    """Wrap `dataset` in a DataLoader that fetches whole batches.

    `batch_size=None` turns off DataLoader's own per-sample fetch and
    collate, handing the `BatchSampler`'s index list straight to
    `StarDataset.__getitem__`. The sampling order is unchanged from
    `DataLoader(..., shuffle=shuffle)`: it is the same sampler under the
    same batching rule (`drop_last=False`), only driven a batch at a time.
    """
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(sampler, batch_size, drop_last=False),
        num_workers=num_workers,
        collate_fn=_passthrough,
        pin_memory=False,
    )


def _broadcast_graph_features(graph_feats, keys, num_stars):
    """Broadcast per-stream features `keys` to per-star rows."""
    per_stream = np.column_stack(
        [graph_feats[k][:len(num_stars)] for k in keys])
    return torch.tensor(
        np.repeat(per_stream, num_stars, axis=0), dtype=torch.float32)


def _build_star_tensors(
    node_feats, graph_feats, x_labels, labels, cond_labels=None, max_graphs=None):
    """Assemble flat per-star x/theta[/cond] tensors.

    `x_labels` fixes the column order of `x`, which is what the
    `feature_idx` of each `UncertaintySampler` indexes into.
    """

    num_particles = np.asarray(graph_feats['num_particles'])
    if max_graphs is not None:
        num_particles = num_particles[:max_graphs]
    n_total = int(num_particles.sum())


    # stack(dim=1), not vstack(...).T: same values, but contiguous rows, so
    # a batch of stars is one memcpy rather than a strided gather.
    x = torch.stack(
        [torch.tensor(node_feats[k][:n_total], dtype=torch.float32)
         for k in x_labels], dim=1)
    theta = _broadcast_graph_features(graph_feats, labels, num_particles)
    cond = (_broadcast_graph_features(graph_feats, cond_labels, num_particles)
            if cond_labels else None)

    return x, theta, cond, num_particles


def _split_by_stream(num_particles, train_frac, seed):
    """Split stars into train/val so no stream spans both sets.

    streams (not stars) are permuted and partitioned; the split is then
    expanded to a per-star boolean mask via each star's stream id.
    """
    rng = np.random.default_rng(seed)
    num_streams = len(num_particles)
    perm = rng.permutation(num_streams)
    n_train = int(train_frac * num_streams)

    is_train_stream = np.zeros(num_streams, dtype=bool)
    is_train_stream[perm[:n_train]] = True

    stream_id = np.repeat(np.arange(num_streams), num_particles)
    train_mask = is_train_stream[stream_id]
    return train_mask, ~train_mask


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _compute_norm(x, theta, cond, pre_transform_kwargs):
    """Compute per-field normalization stats from training stars.

    Delegates to `transforms.compute_field_norm`, which streams the stars
    through the observation pipeline (uncertainty) and measures
    per-component location/scale for x, xerr, theta, and cond.
    The NLE model consumes this dict and applies the normalization itself.
    """
    return compute_field_norm(
        StarBatch(x=x, theta=theta, cond=cond), **pre_transform_kwargs)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def _mask(t, m):
    """Row-select a tensor by a boolean mask, passing None through."""
    return None if t is None else t[m]


def prepare_dataloaders(
    node_feats, graph_feats, x_labels, labels, cond_labels=None, train_frac=0.8,
    train_batch_size=1024, eval_batch_size=1024, num_workers=1,
    norm_dict=None, seed=0, pre_transform_kwargs=None
):
    """Prepare per-star train/val dataloaders from stream-frame features.

    Stars are split by stream (`_split_by_stream`) so train and val never
    share a stream. Loaders yield fully raw physical StarBatches (`x` in
    the units of `x_labels`, physical theta/cond, no `xerr` yet); the NLE
    model applies the observation pipeline (uncertainty) and all
    normalization. `cond_labels` are the measured conditioning variables,
    kept separate from the inferred `labels`.
    `pre_transform_kwargs` are the same kwargs passed to
    `transforms.build_transformation`; they're only needed when `norm_dict`
    isn't supplied, since the field stats are measured by streaming the
    training stars through that pipeline.
    """
    pre_transform_kwargs = pre_transform_kwargs or {}

    x, theta, cond, num_particles = _build_star_tensors(
        node_feats, graph_feats, x_labels, labels, cond_labels)
    tr, va = _split_by_stream(num_particles, train_frac, seed)

    if norm_dict is None:
        print('Computing norm_dict from training stars...')
        norm_dict = _compute_norm(
            x[tr], theta[tr], _mask(cond, tr), pre_transform_kwargs)

    train_loader = _build_loader(
        StarDataset(x[tr], theta[tr], _mask(cond, tr)),
        train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _build_loader(
        StarDataset(x[va], theta[va], _mask(cond, va)),
        eval_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    node_feats, graph_feats, x_labels, labels, cond_labels=None, batch_size=1024,
    num_workers=1, norm_dict=None, max_graphs=None, pre_transform_kwargs=None
):
    """Prepare a per-star test dataloader from stream-frame features."""
    pre_transform_kwargs = pre_transform_kwargs or {}

    x, theta, cond, num_particles = _build_star_tensors(
        node_feats, graph_feats, x_labels, labels, cond_labels,
        max_graphs=max_graphs)

    if norm_dict is None:
        norm_dict = _compute_norm(x, theta, cond, pre_transform_kwargs)

    loader = _build_loader(
        StarDataset(x, theta, cond), batch_size, shuffle=False,
        num_workers=num_workers)

    return loader, norm_dict
