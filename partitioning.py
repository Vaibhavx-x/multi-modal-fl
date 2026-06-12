import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import datasets as hf_datasets
from flwr_datasets.partitioner import DirichletPartitioner


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.01,
    seed: Optional[int] = 42,
    min_samples_per_client: int = 1,
    exclude_indices: Optional[List[int]] = None,
    original_indices: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    """
    Partition a label array across clients using Flower's Dirichlet partitioner.

    Args:
        labels: Full label array (length = total dataset size).
        num_clients: Number of FL clients.
        alpha: Dirichlet concentration parameter (lower = more heterogeneous).
        seed: RNG seed for reproducibility.
        min_samples_per_client: Minimum samples guaranteed per client.
        exclude_indices: Row positions to exclude before partitioning (e.g. server val set).
        original_indices: Full-dataset positions corresponding to `labels`, if `labels`
            is already a subset. When None, positions 0..len(labels)-1 are assumed.

    Returns:
        Dict mapping client_id -> list of full-dataset row indices.
    """
    # Map each element's position in `labels` to its full-dataset index
    orig = np.asarray(original_indices, dtype=np.int64) if original_indices is not None \
        else np.arange(len(labels), dtype=np.int64)

    # Exclude val indices so no client trains on server-held validation samples
    if exclude_indices:
        exclude_set = set(exclude_indices)
        keep_mask = np.array([int(i) not in exclude_set for i in orig], dtype=bool)
        n_excluded = int((~keep_mask).sum())
        orig = orig[keep_mask]
        labels = labels[keep_mask]
        print(f"[Partitioner] Excluded {n_excluded} val-set samples.")

    if len(labels) == 0:
        raise ValueError("No samples remain after excluding val indices.")

    # Build an in-memory HuggingFace Dataset carrying original indices through
    df = pd.DataFrame({"_original_idx": orig, "label": labels.astype(np.int64)})
    hf_ds = hf_datasets.Dataset.from_pandas(df, preserve_index=False)

    # Configure and assign partitions
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=min_samples_per_client,
        self_balancing=True,
        shuffle=True,
        seed=seed,
    )
    partitioner.dataset = hf_ds

    unique_classes = np.unique(labels)
    print(f"\n{'='*60}")
    print(f"  DirichletPartitioner | alpha={alpha} | clients={num_clients}")
    if exclude_indices:
        print(f"  Val samples excluded: {len(exclude_indices)}")
    print(f"{'='*60}")

    client_indices: Dict[int, List[int]] = {}
    total = 0
    for cid in range(num_clients):
        partition = partitioner.load_partition(partition_id=cid)
        indices: List[int] = partition["_original_idx"]
        client_indices[cid] = indices

        n = len(indices)
        total += n
        client_labels = labels[np.isin(orig, indices)]
        class_counts = {int(c): int(np.sum(client_labels == c)) for c in unique_classes}
        print(f"  Client {cid}: {n:>5} samples  |  per-class: {class_counts}")

    print(f"{'-'*60}")
    print(f"  Total assigned: {total}  (available: {len(labels)})")
    print(f"{'='*60}\n")

    return client_indices


# Smoke-test: python partitioning.py
if __name__ == "__main__":
    from dataset import stratified_split

    rng = np.random.default_rng(0)
    fake_labels = rng.choice(6, size=2299)  # 6 classes, PAD-UFES-20 size

    train_idx, val_idx = stratified_split(fake_labels, val_fraction=0.15, seed=42)
    train_labels = fake_labels[train_idx]

    partitions = dirichlet_partition(
        train_labels,
        num_clients=5,
        alpha=0.01,
        original_indices=train_idx,
    )

    val_set = set(val_idx)
    for cid, idxs in partitions.items():
        leaked = [i for i in idxs if i in val_set]
        assert not leaked, f"Client {cid} has {len(leaked)} val samples leaked!"

    all_indices = sorted(idx for idxs in partitions.values() for idx in idxs)
    assert len(all_indices) == len(set(all_indices)), "Duplicate indices found!"
    print("OK -- No val leakage and no duplicate indices.")
