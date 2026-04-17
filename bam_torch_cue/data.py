"""Graph data structures and data loaders for PyTorch-based ML potentials.

Replaces bam_omat24/data_nnx.py (jraph.GraphsTuple → GraphData dataclass,
JAX sharding → standard PyTorch DataLoader).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


# =============================================================================
# Graph Data Structure
# =============================================================================

@dataclass
class GraphData:
    """Atomistic graph data, analogous to jraph.GraphsTuple.

    All arrays are numpy at rest; call .to(device) to get torch tensors.
    """
    n_node: np.ndarray    # (n_graphs,) int32 — atoms per graph
    n_edge: np.ndarray    # (n_graphs,) int32 — edges per graph
    nodes: Dict           # 'species': (N,), 'positions': (N,3), 'forces': (N,3)|None
    edges: Dict           # 'Sij': (E,3) cell-shift vectors (periodic) or {}
    senders: np.ndarray   # (E,) int32
    receivers: np.ndarray # (E,) int32
    globals: Dict         # 'energy':(G,), 'cell':(G,3,3), 'volume':(G,),
                          # 'stress':(G,6), 'cutoff':(G,)

    def to(self, device: torch.device) -> "TorchGraphData":
        """Convert numpy arrays to torch tensors on device."""
        def _t(x, dtype=None):
            if x is None:
                return None
            t = torch.as_tensor(x, device=device)
            return t.to(dtype) if dtype else t

        nodes = {
            'species':   _t(self.nodes['species'], torch.long),
            'positions': _t(self.nodes['positions'], torch.float32),
        }
        if self.nodes.get('forces') is not None:
            nodes['forces'] = _t(self.nodes['forces'], torch.float32)

        edges = {}
        if 'Sij' in self.edges:
            edges['Sij'] = _t(self.edges['Sij'], torch.float32)

        globs = {
            'energy':  _t(self.globals['energy'], torch.float32)
        }
        if self.globals.get('cutoff') is not None:
            globs['cutoff'] = _t(self.globals['cutoff'], torch.float32)
        if 'cell' in self.globals:
            globs['cell']   = _t(self.globals['cell'], torch.float32)
        if 'volume' in self.globals:
            globs['volume'] = _t(self.globals['volume'], torch.float32)
        if 'stress' in self.globals:
            globs['stress'] = _t(self.globals['stress'], torch.float32)

        return TorchGraphData(
            n_node=_t(self.n_node, torch.long),
            n_edge=_t(self.n_edge, torch.long),
            nodes=nodes,
            edges=edges,
            senders=_t(self.senders, torch.long),
            receivers=_t(self.receivers, torch.long),
            globals=globs,
        )


@dataclass
class TorchGraphData:
    """GraphData with all arrays as torch.Tensor (on a specific device)."""
    n_node: torch.Tensor
    n_edge: torch.Tensor
    nodes: Dict
    edges: Dict
    senders: torch.Tensor
    receivers: torch.Tensor
    globals: Dict


# =============================================================================
# Batching Utilities
# =============================================================================

def batch_graphs(graphs: List[GraphData], periodic: bool = True) -> GraphData:
    """Concatenate a list of single-graph GraphData into one batched GraphData.

    Args:
        periodic: If True, include Sij, cell, volume, stress (bulk crystals).
                  If False, omit them (molecules / non-periodic systems).
    """
    n_node = np.concatenate([g.n_node for g in graphs])
    n_edge = np.concatenate([g.n_edge for g in graphs])

    # Offset senders/receivers by cumulative node count
    offsets = np.concatenate([[0], np.cumsum(n_node[:-1])])
    senders_list = []
    receivers_list = []
    for g, off in zip(graphs, offsets):
        senders_list.append(g.senders + off)
        receivers_list.append(g.receivers + off)

    nodes = {
        'species':   np.concatenate([g.nodes['species']   for g in graphs]),
        'positions': np.concatenate([g.nodes['positions'] for g in graphs]),
    }
    if graphs[0].nodes.get('forces') is not None:
        nodes['forces'] = np.concatenate([g.nodes['forces'] for g in graphs])

    edges: Dict = {}
    if periodic and 'Sij' in graphs[0].edges:
        edges['Sij'] = np.concatenate([g.edges['Sij'] for g in graphs])

    globs = {
        'energy':  np.concatenate([g.globals['energy']  for g in graphs])
    }
    if periodic:
        globs['cutoff']  = np.concatenate([g.globals['cutoff']  for g in graphs])
        globs['cell']    = np.concatenate([g.globals['cell']    for g in graphs])
        globs['volume']  = np.concatenate([g.globals['volume']  for g in graphs])
        globs['stress']  = np.concatenate([g.globals['stress']  for g in graphs])

    return GraphData(
        n_node=n_node,
        n_edge=n_edge,
        nodes=nodes,
        edges=edges,
        senders=np.concatenate(senders_list),
        receivers=np.concatenate(receivers_list),
        globals=globs,
    )


# =============================================================================
# Dataset
# =============================================================================

class GraphDataset(TorchDataset):
    """Dataset that loads a list of GraphData from a pickle file."""

    def __init__(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.graphs: List[GraphData] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        return self.graphs[idx]

    def get_stats(self) -> dict:
        n_nodes = [int(g.n_node[0]) for g in self.graphs]
        n_edges = [int(g.n_edge[0]) for g in self.graphs]
        return {
            'count': len(self.graphs),
            'nodes': {'min': min(n_nodes), 'max': max(n_nodes),
                      'mean': float(np.mean(n_nodes))},
            'edges': {'min': min(n_edges), 'max': max(n_edges),
                      'mean': float(np.mean(n_edges))},
        }


# =============================================================================
# Bucketed DataLoader
# =============================================================================

def create_size_buckets(graphs: List[GraphData], n_buckets: int = 8) -> List[List[int]]:
    """Group graph indices by node-count into n_buckets."""
    sizes = np.array([int(g.n_node[0]) for g in graphs])
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(sizes, percentiles)
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for i, s in enumerate(sizes):
        idx = min(int(np.searchsorted(boundaries[1:], s, side='right')), n_buckets - 1)
        buckets[idx].append(i)
    return buckets


class BucketedDataLoader:
    """DataLoader that batches graphs of similar sizes.

    Analogue of bam_omat24/data_nnx.py:BucketedDataLoader.

    For distributed training, set ``rank`` and ``world_size`` so that each
    process receives a disjoint shard of graphs.  Call :meth:`set_epoch`
    at the start of every epoch to re-seed the shuffle (analogous to
    ``torch.utils.data.distributed.DistributedSampler.set_epoch``).
    """

    def __init__(
        self,
        dataset: GraphDataset,
        batch_size: int = 10,
        n_buckets: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        max_nodes_per_batch: int = 512,
        seed: int = 0,
        periodic: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_nodes_per_batch = max_nodes_per_batch
        self.periodic = periodic
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)

        self.buckets = create_size_buckets(dataset.graphs, n_buckets)

    def set_epoch(self, epoch: int):
        """Set epoch to ensure different shuffling each epoch across all ranks."""
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + epoch)

    def __iter__(self) -> Iterator[GraphData]:
        # Optionally shuffle within each bucket
        if self.shuffle:
            bucket_orders = [
                self.rng.permutation(b).tolist() for b in self.buckets
            ]
        else:
            bucket_orders = [list(b) for b in self.buckets]

        for bucket in bucket_orders:
            # Distributed sharding: each rank takes a disjoint slice
            if self.world_size > 1:
                bucket = bucket[self.rank::self.world_size]

            pos = 0
            while pos < len(bucket):
                remaining = len(bucket) - pos
                if self.drop_last and remaining < self.batch_size:
                    break
                n_take = min(self.batch_size, remaining)
                graphs = []
                n_nodes = 0
                for j in range(n_take):
                    g = self.dataset.graphs[bucket[pos + j]]
                    n_nodes += int(g.n_node[0])
                    if n_nodes >= self.max_nodes_per_batch and graphs:
                        break
                    graphs.append(g)
                if not graphs:
                    break
                pos += len(graphs)
                yield batch_graphs(graphs, periodic=self.periodic)

    def __len__(self) -> int:
        total = 0
        for b in self.buckets:
            n = len(b) // self.world_size if self.world_size > 1 else len(b)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total
