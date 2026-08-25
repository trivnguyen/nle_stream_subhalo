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
                # Reason: the files store float64 but every consumer builds
                # float32 tensors, so carrying the wider dtype as far as the
                # concatenate doubles the largest resident block (10.5 GB
                # rather than 5.3 GB over 100 files) to be thrown away.
                array = nodes[k]
                if array.dtype == np.float64:
                    array = array.astype(np.float32)
                node_feats.setdefault(k, []).append(array)
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
    `torch.stack`.

    Two things are stored compressed, because at 400 files the dense form
    is most of the job's memory and none of its information:

    * `theta` is a property of the *stream*, identical for every one of
      its ~233 stars. Broadcasting it per-star is 9 floats per star --
      33.5 GB at 400 files to hold 144 MB -- so it is kept per stream and
      gathered per batch through `stream_id`.
    * `index` selects this split's rows out of a *shared* `x`, so train
      and val are two index arrays over one tensor rather than two
      advanced-index copies of it.

    Neither changes what a batch contains. `index` is ascending, so row
    `i` of this dataset is the same star as row `i` of the boolean-masked
    copy it replaces, and the gathered `theta` is the same value the
    broadcast one held.

    Attributes:
        x: (n_all, D) shared star features; not restricted to this split.
        theta_stream: (n_streams, P) per-stream parameters.
        stream_id: (n_all,) int32 index into `theta_stream`.
        index: (n_split,) int32 rows of `x` in this split, ascending.
        cond_stream: (n_streams, C) per-stream conditioning, or None.
    """

    def __init__(
        self, x: torch.Tensor, theta_stream: torch.Tensor,
        stream_id: torch.Tensor, index: torch.Tensor = None,
        cond_stream: torch.Tensor = None):
        self.x = x
        self.theta_stream = theta_stream
        self.stream_id = stream_id
        self.index = index
        self.cond_stream = cond_stream

    def __len__(self) -> int:
        return (self.x.shape[0] if self.index is None
                else self.index.shape[0])

    def _rows(self, idx) -> torch.Tensor:
        """Rows of the shared `x` for this split-relative index."""
        if self.index is None:
            return torch.as_tensor(idx, dtype=torch.long)
        return self.index[idx].long()

    def __getitem__(self, idx) -> StarBatch:
        rows = self._rows(idx)
        sid = self.stream_id[rows].long()
        return StarBatch(
            x=self.x[rows],
            theta=self.theta_stream[sid],
            cond=(None if self.cond_stream is None
                  else self.cond_stream[sid]),
        )

    def to_batch(self, rows=None) -> StarBatch:
        """Materialize a dense StarBatch, for fitting the normalization.

        Args:
            rows: Split-relative indices to materialize; None takes the
                whole split.

        Returns:
            A dense StarBatch of the requested rows.
        """
        if rows is None:
            rows = torch.arange(len(self))
        return self[rows]

    def stream_index(self) -> torch.Tensor:
        """The streams this split covers, ascending and deduplicated."""
        rows = (torch.arange(self.x.shape[0]) if self.index is None
                else self.index.long())
        return torch.unique(self.stream_id[rows].long())


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


def _stack_graph_features(graph_feats, keys, num_streams):
    """Per-stream features `keys` as one (n_streams, len(keys)) tensor.

    Deliberately *not* broadcast to stars: `StarDataset` gathers per batch
    instead. The broadcast form was 233x larger and carried no more
    information.
    """
    per_stream = np.column_stack(
        [graph_feats[k][:num_streams] for k in keys])
    return torch.tensor(per_stream, dtype=torch.float32)


def _build_star_tensors(
    node_feats, graph_feats, x_labels, labels, cond_labels=None,
    max_graphs=None, consume=False):
    """Assemble flat per-star x/theta[/cond] tensors.

    `x_labels` fixes the column order of `x`, which is what the
    `feature_idx` of each `UncertaintySampler` indexes into.

    Args:
        consume: Drop each `node_feats` column from the caller's dict as it
            is copied in. The source outweighs the result at this scale
            (5.3 GB over 100 files), so holding both to the end of the
            loop is the peak this function is responsible for. Leave False
            when the caller still needs `node_feats`.
    """

    num_particles = np.asarray(graph_feats['num_particles'])
    if max_graphs is not None:
        num_particles = num_particles[:max_graphs]
    n_total = int(num_particles.sum())

    # Columns are written into one preallocated (n, D) block rather than
    # stacked from a list: same contiguous rows -- so a batch of stars is
    # one memcpy rather than a strided gather -- without holding D
    # single-column tensors alongside the result.
    x = torch.empty((n_total, len(x_labels)), dtype=torch.float32)
    for j, k in enumerate(x_labels):
        column = node_feats.pop(k) if consume else node_feats[k]
        x[:, j] = torch.from_numpy(np.asarray(column[:n_total]))
        del column

    n_streams = len(num_particles)
    theta = _stack_graph_features(graph_feats, labels, n_streams)
    cond = (_stack_graph_features(graph_feats, cond_labels, n_streams)
            if cond_labels else None)
    # int32: 4M streams at 400 files, so the ids fit with room to spare,
    # and this array is one per star.
    stream_id = torch.from_numpy(
        np.repeat(np.arange(n_streams, dtype=np.int32), num_particles))

    return x, theta, cond, stream_id, num_particles


def _split_by_stream(num_particles, train_frac, seed):
    """Split stars into train/val so no stream spans both sets.

    Streams (not stars) are permuted and partitioned; the split is then
    expanded per star via each star's stream id.

    Returns ascending *index* arrays rather than boolean masks. Indexing
    `x` with either selects the same rows in the same order -- a boolean
    mask and `flatnonzero` of it are equivalent selections -- but the
    index form lets both splits share one `x` instead of each taking a
    copy of their half. The RNG draws are untouched, so the split itself
    is identical to the mask version.
    """
    rng = np.random.default_rng(seed)
    num_streams = len(num_particles)
    perm = rng.permutation(num_streams)
    n_train = int(train_frac * num_streams)

    is_train_stream = np.zeros(num_streams, dtype=bool)
    is_train_stream[perm[:n_train]] = True

    stream_id = np.repeat(np.arange(num_streams), num_particles)
    train_mask = is_train_stream[stream_id]
    return (torch.from_numpy(np.flatnonzero(train_mask).astype(np.int32)),
            torch.from_numpy(np.flatnonzero(~train_mask).astype(np.int32)))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _compute_norm(dataset, pre_transform_kwargs, track=None,
                  norm_kwargs=None, max_stars=None, seed=0):
    """Compute per-field normalization stats from training stars.

    Delegates to `transforms.compute_field_norm`, which streams the stars
    through the observation pipeline (uncertainty) and measures
    per-component location/scale for x, xerr, theta, and cond.
    The NLE model consumes this dict and applies the normalization itself.
    `track`, when given, is applied to `x` first, since that is the order
    the model uses.

    The `max_stars` subsample is drawn *here* and handed over already
    materialized, with `max_stars=None` passed on, because `dataset` no
    longer holds a dense per-star `x` for `compute_field_norm` to slice.
    The draw below mirrors that function's own -- same generator, same
    seed, same `randint`-then-sort -- so it selects the identical rows in
    the identical order; **the two must be changed together.**

    `theta` and `cond` go in per *stream*, not per star. Their stats are
    `min`/`max` over the field (see `compute_field_norm`), and
    broadcasting a value to its stars only adds duplicates, which move
    neither. Passing the per-stream form is therefore exact, not an
    approximation -- and it is what keeps this off the 33 GB the
    broadcast form would cost.
    """
    n = len(dataset)
    rows = None
    if max_stars is not None and n > max_stars:
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randint(n, (max_stars,), generator=generator)
        rows = rows.sort().values
        print(f'Field normalization on {max_stars:,} of {n:,} stars')

    x_batch = dataset.to_batch(rows)
    streams = dataset.stream_index()
    batch = StarBatch(
        x=x_batch.x,
        theta=dataset.theta_stream[streams],
        cond=(None if dataset.cond_stream is None
              else dataset.cond_stream[streams]))
    return compute_field_norm(
        batch, track=track, max_stars=None, seed=seed,
        **(norm_kwargs or {}),
        **pre_transform_kwargs)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    node_feats, graph_feats, x_labels, labels, cond_labels=None, train_frac=0.8,
    train_batch_size=1024, eval_batch_size=1024, num_workers=1,
    norm_dict=None, seed=0, pre_transform_kwargs=None, track=None,
    norm_kwargs=None, norm_max_stars=None
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
    training stars through that pipeline. `track` is likewise only needed
    then: the loaders yield raw physical stars either way, and the model
    owns the projection. `norm_max_stars` caps how many training stars
    that measurement sees (see `transforms.compute_field_norm`).

    Consumes `node_feats`: its columns are dropped as they are copied into
    `x`, so it is empty on return.
    """
    pre_transform_kwargs = pre_transform_kwargs or {}

    x, theta, cond, stream_id, num_particles = _build_star_tensors(
        node_feats, graph_feats, x_labels, labels, cond_labels, consume=True)
    tr, va = _split_by_stream(num_particles, train_frac, seed)

    # Both splits index into the *same* `x` and the same per-stream
    # `theta`. The previous version handed each split its own
    # advanced-index copy, which doubled the largest tensors at exactly
    # the moment both existed.
    train = StarDataset(x, theta, stream_id, index=tr, cond_stream=cond)
    val = StarDataset(x, theta, stream_id, index=va, cond_stream=cond)

    if norm_dict is None:
        print('Computing norm_dict from training stars...')
        norm_dict = _compute_norm(
            train, pre_transform_kwargs, track=track,
            norm_kwargs=norm_kwargs, max_stars=norm_max_stars)

    train_loader = _build_loader(
        train, train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _build_loader(
        val, eval_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    node_feats, graph_feats, x_labels, labels, cond_labels=None, batch_size=1024,
    num_workers=1, norm_dict=None, max_graphs=None, pre_transform_kwargs=None
):
    """Prepare a per-star test dataloader from stream-frame features."""
    pre_transform_kwargs = pre_transform_kwargs or {}

    x, theta, cond, stream_id, _ = _build_star_tensors(
        node_feats, graph_feats, x_labels, labels, cond_labels,
        max_graphs=max_graphs)
    dataset = StarDataset(x, theta, stream_id, cond_stream=cond)

    if norm_dict is None:
        norm_dict = _compute_norm(dataset, pre_transform_kwargs)

    loader = _build_loader(
        dataset, batch_size, shuffle=False, num_workers=num_workers)

    return loader, norm_dict
