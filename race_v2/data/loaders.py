import pickle
from pathlib import Path
from typing import Callable, List, Tuple, Iterator, Optional, Literal
from dataclasses import dataclass

#import ase
#import ase.io
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import jraph
#import matscipy.neighbours
import numpy as np
from flax import nnx
from tqdm import tqdm


# =============================================================================
# Host <-> device (multi-process data parallel)
# =============================================================================

def to_host(x) -> np.ndarray:
    """NumPy value of an array; for a replicated array spanning several processes (multi-node data
    parallel) the local copy is read, since such an array is not fully addressable."""
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        x = x.addressable_data(0)
    return np.asarray(x)


def global_from_host(x, sharding: NamedSharding) -> jax.Array:
    """Global array with the given sharding from a host value that is identical on every process
    (each process places only the shards of its own devices)."""
    x = np.asarray(to_host(x))
    x = x.astype(jax.dtypes.canonicalize_dtype(x.dtype))
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])


def create_size_buckets(
    graphs: List[jraph.GraphsTuple],
    n_buckets: int = 8
) -> List[List[int]]:
    """Group graph indices into buckets by size.

    Graphs in the same bucket have similar sizes, reducing padding waste.
    """
    sizes = np.array([int(g.n_node[0]) for g in graphs])

    # Use quantiles for bucket boundaries
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(sizes, percentiles)

    buckets = [[] for _ in range(n_buckets)]
    for i, size in enumerate(sizes):
        # Find which bucket this graph belongs to
        bucket_idx = min(
            np.searchsorted(boundaries[1:], size, side='right'),
            n_buckets - 1
        )
        buckets[bucket_idx].append(i)

    return buckets


def nearest_bucket(x: int, bucket_size: int = 32) -> int:
    """Round up to nearest bucket boundary.

    More memory-efficient than power-of-2 for intermediate sizes.
    E.g., with bucket_size=32: 65 -> 96, 100 -> 128
    """
    return ((x + bucket_size - 1) // bucket_size) * bucket_size


def default_collate_fn(
    graphs: List[jraph.GraphsTuple],
    batch_size: int,
    node_bucket_size: int = 128,
    edge_bucket_size: int = 4096
) -> jraph.GraphsTuple:

    if len(graphs) == 0:
        raise ValueError("Cannot collate empty list of graphs")

    # Batch graphs
    batched = jraph.batch_np(graphs)

    # Compute current sizes
    n_nodes = int(np.sum(batched.n_node))
    n_edges = int(np.sum(batched.n_edge))

    pad_nodes = nearest_bucket(n_nodes + 1, node_bucket_size)
    pad_edges = nearest_bucket(n_edges + 1, edge_bucket_size)
    pad_graphs = batch_size + 1

    return jraph.pad_with_graphs(batched, pad_nodes, pad_edges, pad_graphs)


def fixed_collate_fn(
    graphs: List[jraph.GraphsTuple],
    batch_size: int,
    n_node_pad: int,
    n_edge_pad: int,
) -> jraph.GraphsTuple:
    """Pad every batch to ONE fixed shape (n_node_pad nodes, n_edge_pad edges, batch_size + 1 graphs), so the
    jitted step compiles once.  The loader must pack graphs under these budgets
    (BucketedDataLoader max_nodes_per_batch=n_node_pad, max_edges_per_batch=n_edge_pad)."""
    if len(graphs) == 0:
        raise ValueError("Cannot collate empty list of graphs")
    batched = jraph.batch_np(graphs)
    n_nodes = int(np.sum(batched.n_node)); n_edges = int(np.sum(batched.n_edge))
    if n_nodes >= n_node_pad or n_edges > n_edge_pad or len(graphs) > batch_size:
        raise ValueError(f"batch ({len(graphs)} graphs, {n_nodes} nodes, {n_edges} edges) exceeds the fixed shape "
                         f"({batch_size} graphs, {n_node_pad} nodes, {n_edge_pad} edges)")
    return jraph.pad_with_graphs(batched, n_node_pad, n_edge_pad, batch_size + 1)


# =============================================================================
# Dynamic Batching (Size-Aware)
# =============================================================================

def compute_batch_stats(graphs: List[jraph.GraphsTuple]) -> dict:
    """Compute statistics about graph sizes for optimal batching."""
    n_nodes = [int(g.n_node[0]) for g in graphs]
    n_edges = [int(g.n_edge[0]) for g in graphs]

    return {
        'count': len(graphs),
        'nodes': {
            'min': min(n_nodes),
            'max': max(n_nodes),
            'mean': np.mean(n_nodes),
            'std': np.std(n_nodes),
            'total': sum(n_nodes),
        },
        'edges': {
            'min': min(n_edges),
            'max': max(n_edges),
            'mean': np.mean(n_edges),
            'std': np.std(n_edges),
            'total': sum(n_edges),
        }
    }


# =============================================================================
# Dataset Class
# =============================================================================

class Dataset(nnx.Object):
    """NNX-based dataset that reads xyz files and returns jraph.GraphsTuple."""

    def __init__(
        self,
        file_path: str = None
    ):
        with open(file_path, "rb") as f:
            graphs = pickle.load(f)

        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        return self.graphs[idx]

    def get_stats(self) -> dict:
        """Get statistics about the dataset."""
        return compute_batch_stats(self.graphs)


# =============================================================================
# Advanced: Bucketed DataLoader for Minimum Padding
# =============================================================================

class BucketedDataLoader(nnx.Object):
    """DataLoader that groups graphs by size into buckets for minimal padding.

    This loader:
    1. Groups graphs into size buckets
    2. Samples batches from within buckets
    3. Results in much less padding waste than random batching
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 10,
        n_buckets: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        rngs: nnx.Rngs | None = None,
        max_nodes_per_batch=256,
        max_edges_per_batch=None,
    ):
        self.dataset = dataset
        self.max_edge_per_batch = max_edges_per_batch
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = default_collate_fn
        self.max_node_per_batch = max_nodes_per_batch

        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs

        # Create size buckets
        self.buckets = create_size_buckets(dataset.graphs, n_buckets)
        self.bucket_sizes = [len(b) for b in self.buckets]

        # Calculate total batches
        self.n_batches = sum(
            size // batch_size if drop_last else (size + batch_size - 1) // batch_size
            for size in self.bucket_sizes if size > 0
        )

        self._reset()

    def _reset(self):
        """Reset for new epoch."""
        self.current_bucket = 0
        self.bucket_idx = 0

        # Shuffle within each bucket
        if self.shuffle:
            key = self.rngs.params()
            self.bucket_orders = []
            for bucket in self.buckets:
                if len(bucket) > 0:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, len(bucket))
                    self.bucket_orders.append([bucket[i] for i in np.asarray(perm)])
                else:
                    self.bucket_orders.append([])
        else:
            self.bucket_orders = [list(b) for b in self.buckets]

    def __iter__(self):
        self._reset()
        return self

    def __next__(self) -> jraph.GraphsTuple:
        # Find next non-empty bucket with remaining samples
        while self.current_bucket < self.n_buckets:
            bucket = self.bucket_orders[self.current_bucket]
            remaining = len(bucket) - self.bucket_idx

            if remaining <= 0:
                self.current_bucket += 1
                self.bucket_idx = 0
                continue

            if self.drop_last and remaining < self.batch_size:
                self.current_bucket += 1
                self.bucket_idx = 0
                continue

            # Collect batch from current bucket
            n_to_take = min(self.batch_size, remaining)
            graphs = []
            n_nodes = 0
            n_edges = 0
            curr_idx = self.bucket_idx
            for _ in range(n_to_take):
                idx = bucket[curr_idx]
                curr_graph = self.dataset.graphs[idx]
                n_nodes += curr_graph.n_node[0]
                n_edges += curr_graph.n_edge[0]
                if n_nodes >= self.max_node_per_batch:
                    break
                if self.max_edge_per_batch is not None and n_edges > self.max_edge_per_batch:
                    break
                curr_idx += 1
                graphs.append(curr_graph)

            if len(graphs) > 0:
                self.bucket_idx = curr_idx
                return self.collate_fn(graphs, self.batch_size)
            else:
                # one graph alone reaches max_nodes_per_batch: ending the epoch here would silently
                # skip every remaining graph
                g0 = self.dataset.graphs[bucket[curr_idx]]
                raise ValueError(f"graph {bucket[curr_idx]} has {int(g0.n_node[0])} nodes / {int(g0.n_edge[0])} edges, "
                                 f"over max_nodes_per_batch={self.max_node_per_batch} / "
                                 f"max_edges_per_batch={self.max_edge_per_batch}")

        raise StopIteration

    def __len__(self) -> int:
        return self.n_batches


# =============================================================================
# Multi-Device Batch Info Dataclass
# =============================================================================

@dataclass
class MultiDeviceBatchInfo:
    """Information about a multi-device batch."""
    n_devices: int
    valid_device_count: int
    n_graphs_per_device: List[int]
    n_nodes_per_device: List[int]
    n_edges_per_device: List[int]
    padded_n_nodes: int
    padded_n_edges: int
    padded_n_graphs: int

    def __repr__(self):
        return (
            f"MultiDeviceBatchInfo(\n"
            f"  valid_devices={self.valid_device_count}/{self.n_devices}\n"
            f"  graphs_per_device={self.n_graphs_per_device}\n"
            f"  nodes_per_device={self.n_nodes_per_device}\n"
            f"  edges_per_device={self.n_edges_per_device}\n"
            f"  padded_shape=(nodes={self.padded_n_nodes}, edges={self.padded_n_edges}, graphs={self.padded_n_graphs})\n"
            f"  efficiency={100 * sum(self.n_nodes_per_device) / (self.padded_n_nodes * self.n_devices):.1f}%\n"
            f")"
        )



def stack_batches_for_devices(
    batches: List[jraph.GraphsTuple],
    mesh: Mesh,
) -> Tuple[jraph.GraphsTuple, MultiDeviceBatchInfo]:
    """Stack multiple batches and shard across devices.

    Use this after collecting batches from BucketedDataLoader:

        batches = [next(loader) for _ in range(n_devices)]
        sharded_batch, info = stack_batches_for_devices(batches, mesh)

    Args:
        batches: List of GraphsTuple from BucketedDataLoader
        mesh: JAX device mesh for sharding

    Returns:
        Tuple of (sharded_batch, batch_info)
    """
    n_devices = len(batches)

    # Helper to get actual array sizes
    def get_node_size(b):
        if isinstance(b.nodes, dict):
            first_key = next(iter(b.nodes.keys()))
            return b.nodes[first_key].shape[0]
        return b.nodes.shape[0]

    def get_edge_size(b):
        return b.senders.shape[0]

    # Collect stats before padding
    stats = {
        'n_graphs': [],
        'n_nodes': [],
        'n_edges': [],
    }
    for b in batches:
        # Real counts (not padded)
        real_graphs = jraph.get_graph_padding_mask(b).sum()
        real_nodes = jraph.get_node_padding_mask(b).sum()
        real_edges = jraph.get_edge_padding_mask(b).sum()
        stats['n_graphs'].append(real_graphs)
        stats['n_nodes'].append(real_nodes)
        stats['n_edges'].append(real_edges)
    # Find max actual array sizes
    node_sizes = [get_node_size(b) for b in batches]
    edge_sizes = [get_edge_size(b) for b in batches]
    graph_sizes = [b.n_node.shape[0] for b in batches]

    max_nodes = max(node_sizes)
    max_edges = max(edge_sizes)
    max_graphs = max(graph_sizes)

    # Pad all batches to same size
    padded_batches = []
    for b, n_nodes, n_edges, n_graphs in zip(batches, node_sizes, edge_sizes, graph_sizes):
        if n_nodes < max_nodes or n_edges < max_edges or n_graphs < max_graphs:
            ub = jraph.unpad_with_graphs (b)
            b = jraph.pad_with_graphs(ub, max_nodes, max_edges, max_graphs)
        padded_batches.append(b)
    sharding = NamedSharding(mesh, P('dp'))
    if jax.process_count() == 1:
        # Stack along device dimension
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *padded_batches)
        # Shard across devices
        sharded = jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)
    else:
        # multi-process: every process collates the same global group (identical loaders and
        # seeds) and places only the slices of its own devices
        stacked = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *padded_batches)
        sharded = jax.tree.map(lambda x: global_from_host(x, sharding), stacked)

    info = MultiDeviceBatchInfo(
        n_devices=n_devices,
        valid_device_count=n_devices,  # All valid in this approach
        n_graphs_per_device=stats['n_graphs'],
        n_nodes_per_device=stats['n_nodes'],
        n_edges_per_device=stats['n_edges'],
        padded_n_nodes=max_nodes,
        padded_n_edges=max_edges,
        padded_n_graphs=max_graphs,
    )

    return sharded, info


def multi_device_iterator(
    loader,  # BucketedDataLoader
    n_devices: int,
    mesh: Mesh,
    drop_incomplete: bool = True,
) -> Iterator[Tuple[jraph.GraphsTuple, MultiDeviceBatchInfo]]:
    """Wrap a BucketedDataLoader to yield multi-device batches.

    Example:
        loader = BucketedDataLoader(dataset, batch_size=20)
        for batch, info in multi_device_iterator(loader, n_devices, mesh):
            loss = train_step(batch)

    Args:
        loader: BucketedDataLoader instance
        n_devices: Number of devices
        mesh: JAX device mesh
        drop_incomplete: If True, skip final batch if < n_devices batches available

    Yields:
        Tuple of (sharded_batch, batch_info)
    """
    batches = []

    for batch in loader:
        batches.append(batch)

        if len(batches) == n_devices:
            yield stack_batches_for_devices(batches, mesh)
            batches = []

    # Handle remaining batches
    if batches and not drop_incomplete:
        # Pad with duplicates of last batch
        valid_count = len(batches)
        while len(batches) < n_devices:
            batches.append(batches[-1])

        sharded, info = stack_batches_for_devices(batches, mesh)
        # Adjust valid count
        info.valid_device_count = valid_count
        yield sharded, info


class MultiDeviceDataLoader(nnx.Object):
    """Wrapper that converts any single-device loader to multi-device.

    This wraps BucketedDataLoader (or any iterator) and collects
    n_devices batches before stacking and sharding.

    Example:
        base_loader = BucketedDataLoader(dataset, batch_size=20)
        loader = MultiDeviceDataLoader(base_loader, n_devices, mesh)

        for batch, info in loader:
            loss = train_step(batch)
    """

    def __init__(
        self,
        base_loader,
        n_devices: int,
        mesh: Mesh,
        drop_incomplete: bool = True,
    ):
        self.base_loader = base_loader
        self.n_devices = n_devices
        self.mesh = mesh
        self.drop_incomplete = drop_incomplete

        # Estimate number of multi-device batches
        if hasattr(base_loader, '__len__'):
            self.n_batches = len(base_loader) // n_devices
        else:
            self.n_batches = None

    def __iter__(self):
        return multi_device_iterator(
            iter(self.base_loader),
            self.n_devices,
            self.mesh,
            self.drop_incomplete,
        )

    def __len__(self):
        if self.n_batches is not None:
            return self.n_batches
        raise TypeError("Base loader doesn't support len()")




def print_bucket_batch_info(bucket_batch_info: dict,
                            title: str = "Bucket Batch Statistics"):
    """Pretty print bucket batch statistics."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"\n{'Unique (node_pad, edge_pad) shapes:':<40} {len(bucket_batch_info)}")
    print(f"\n{'Shape (nodes, edges)':<25} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*45}")

    total_batches = sum(bucket_batch_info.values())
    sorted_items = sorted(bucket_batch_info.items(), key=lambda x: -x[1])

    for (n_nodes, n_edges), count in sorted_items:
        pct = 100 * count / total_batches
        print(f"({n_nodes:>6}, {n_edges:>6})       {count:<10} {pct:>6.1f}%")

    print(f"{'-'*45}")
    print(f"{'Total batches:':<25} {total_batches}")
    print(f"{'='*60}\n")


# =============================================================================
# Device Mesh Setup
# =============================================================================

def setup_mesh() -> Tuple[Mesh, int]:
    """Setup device mesh for data parallelism (all devices of all processes)."""
    devices = jax.devices()
    n_devices = len(devices)
    if jax.process_index() == 0:
        print(f"Setting up mesh with {n_devices} device(s)")
        for i, d in enumerate(devices):
            print(f"  Device {i}: {d}")

    device_mesh = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(device_mesh, axis_names=('dp',))  # dp = data parallel
    return mesh, n_devices


# =============================================================================
# Test / Example
# =============================================================================

if __name__ == "__main__":
    from natsort import natsorted

    mesh, n_devices = setup_mesh()

    train_path = Path('/home/willow/Python/BAM-nequip-main/race_v2')
    train_files_pkl = natsorted(list(train_path.glob('*.pkl')))

    rngs = nnx.Rngs(42)
    batch_size = 20

    # Compare padding strategies
    print("Comparing padding strategies...")
    train_dataset = Dataset(file_path=train_files_pkl[0])

    # Print dataset stats
    stats = train_dataset.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total graphs: {stats['count']}")
    print(f"  Nodes: min={stats['nodes']['min']}, max={stats['nodes']['max']}, "
          f"mean={stats['nodes']['mean']:.1f}, std={stats['nodes']['std']:.1f}")
    print(f"  Edges: min={stats['edges']['min']}, max={stats['edges']['max']}, "
          f"mean={stats['edges']['mean']:.1f}, std={stats['edges']['std']:.1f}")

    # Test MultiDeviceBucketedDataLoader
    print("\n\nTesting BucketedDataLoader...")
    base_loader = BucketedDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        n_buckets=8,
        shuffle=True,
        drop_last=True,
        rngs=rngs,
        max_nodes_per_batch=256,
    )

    loader = MultiDeviceDataLoader( base_loader,
                                    n_devices=n_devices,
                                    mesh=mesh)

    bucket_batch_info = {}

    total_real_nodes = 0
    total_padded_nodes = 0

    print("\nIterating through batches...")
    for batch_idx, (batch, info) in enumerate(tqdm(loader)):

        # Track padding statistics
        key = (info.padded_n_nodes, info.padded_n_edges)
        bucket_batch_info[key] = bucket_batch_info.get(key, 0) + 1

        # Calculate padding efficiency
        real_nodes = sum(info.n_nodes_per_device)
        padded_nodes = info.padded_n_nodes * n_devices
        total_real_nodes += real_nodes
        total_padded_nodes += padded_nodes

        # Print detailed info for first few batches
        if batch_idx < 5:
            print(f"\nBatch {batch_idx}:")
            print(info)
            # Handle both dict and array node features
            if isinstance(batch.nodes, dict):
                first_key = next(iter(batch.nodes.keys()))
                node_shape = batch.nodes[first_key].shape
                print(f"  Batch shape: n_node={batch.n_node.shape}, nodes['{first_key}']={node_shape}")
            else:
                print(f"  Batch shape: n_node={batch.n_node.shape}, nodes={batch.nodes.shape}")
            print(f"  Padding efficiency: {100 * real_nodes / padded_nodes:.1f}%")

    # Print summary statistics
    print_bucket_batch_info(bucket_batch_info)


    # Print overall padding efficiency
    print(f"\nOverall Padding Efficiency:")
    print(f"  Total real nodes: {total_real_nodes}")
    print(f"  Total padded nodes: {total_padded_nodes}")
    print(f"  Efficiency: {100 * total_real_nodes / total_padded_nodes:.1f}%")
