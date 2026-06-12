import argparse
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import filelock
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as flwr
from flwr.client import NumPyClient, start_client

from dataset import PADUFES20Dataset, NUM_CLASSES, prepare_train_val_split
from partitioning import dirichlet_partition
from model import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _acquire_gpu_slot(lock_dir: str, max_slots: int) -> filelock.FileLock:
    """
    Block until one of `max_slots` GPU slot lock files can be acquired.

    Slots are implemented as file locks on a shared Docker volume, so this
    works across separate containers without any network coordination.
    Returns the held FileLock — caller MUST release it after GPU work.
    """
    os.makedirs(lock_dir, exist_ok=True)
    while True:
        for i in range(max_slots):
            lock = filelock.FileLock(os.path.join(lock_dir, f"gpu_slot_{i}.lock"), timeout=0)
            try:
                lock.acquire()
                return lock
            except filelock.Timeout:
                continue
        time.sleep(0.5)


class FlowerClient(NumPyClient):
    """
    Federated client that trains for one epoch per FL round.

    Evaluation is done centrally on the server's held-out val set; clients
    have no local val split. GPU access is throttled via a file-based
    semaphore so that at most `max_gpu_slots` containers use the GPU at once.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        gpu_lock_dir: Optional[str] = None,
        max_gpu_slots: int = 3,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.client_id    = client_id
        self.gpu_lock_dir = gpu_lock_dir
        self.max_gpu_slots = max_gpu_slots
        self.criterion    = nn.CrossEntropyLoss()
        # LR is injected by the server each round via the fit() config dict
        self.optimizer    = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model weights as CPU numpy arrays."""
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load aggregated weights received from the server."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Apply server-managed LR for this round
        lr = float(config.get("lr", self.optimizer.param_groups[0]["lr"]))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.set_parameters(parameters)

        # Acquire a GPU slot before touching the GPU
        slot: Optional[filelock.FileLock] = None
        if self.gpu_lock_dir and device.type == "cuda":
            print(f"[Client {self.client_id}] waiting for GPU slot...")
            slot = _acquire_gpu_slot(self.gpu_lock_dir, self.max_gpu_slots)
            print(f"[Client {self.client_id}] GPU slot acquired.")

        try:
            self.model.to(device).train()
            running_loss, total, correct = 0.0, 0, 0

            for image, tabular, label in self.train_loader:
                image   = image.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                label   = label.to(device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(image, tabular)
                loss   = self.criterion(logits, label)
                loss.backward()
                self.optimizer.step()

                b             = image.size(0)
                running_loss += loss.item() * b
                total        += b
                correct      += (logits.argmax(dim=1) == label).sum().item()

        finally:
            self.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if slot is not None:
                slot.release()
                print(f"[Client {self.client_id}] GPU slot released.")

        avg_loss = running_loss / total if total > 0 else 0.0
        acc      = correct / total      if total > 0 else 0.0
        print(f"[Client {self.client_id}] loss={avg_loss:.4f}  acc={acc:.4f}  lr={lr:.2e}  n={total}")

        return self.get_parameters(config={}), total, {
            "train_loss": float(avg_loss),
            "train_accuracy": float(acc),
            "lr": float(lr),
        }

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        # Federated evaluation is disabled (fraction_evaluate=0.0) when
        # centralized val evaluation is active. Stub kept for interface compliance.
        return 0.0, 0, {}


def build_client(
    client_id: int,
    archive_root: str,
    num_clients: int = 5,
    alpha: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    gpu_lock_dir: Optional[str] = None,
    max_gpu_slots: int = 3,
) -> FlowerClient:
    """
    Build a FlowerClient whose full local partition is used for training.

    Pipeline:
        1. Reproduce the global 85/15 train/val split with the same seed as
           the server to get `train_indices`, `val_indices`, and `numerical_stats`
           fitted exclusively on train rows.
        2. Partition `train_indices` across clients with Dirichlet.
        3. Give this client its full Dirichlet share as training data.
        4. Build the dataset with the pre-fitted `numerical_stats` so
           z-score normalisation is identical on every client and the server.
    """
    print(f"\n[Client {client_id}] device={device}")

    # Step 1: reproduce the global train/val split (same seed → same split as server)
    global_train_indices, global_val_indices, numerical_stats, _ = prepare_train_val_split(
        archive_root=archive_root, val_fraction=0.15, seed=seed
    )

    # Step 2: Dirichlet partition over train indices only
    full_ds = PADUFES20Dataset(
        archive_root=archive_root, indices=None, train=False, numerical_stats=numerical_stats
    )
    partitions = dirichlet_partition(
        labels=full_ds.get_all_labels()[global_train_indices],
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        original_indices=global_train_indices,
    )

    train_idx = partitions[client_id]

    # Sanity check: ensure zero overlap with the server's val set
    leaked = set(train_idx) & set(global_val_indices)
    if leaked:
        raise RuntimeError(
            f"[Client {client_id}] DATA LEAK: {len(leaked)} samples overlap with server val set!"
        )
    print(f"[Client {client_id}] {len(train_idx)} training samples  "
          f"(val set stays on server — {len(global_val_indices)} samples)")

    # Step 3: Dataset and DataLoader
    train_ds = PADUFES20Dataset(
        archive_root=archive_root, indices=train_idx, train=True, numerical_stats=numerical_stats
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    # Step 4: Build model on CPU to save VRAM; moved to GPU only during training
    model = build_model(tabular_in_dim=train_ds.feature_dim, num_classes=NUM_CLASSES, device="cpu")

    return FlowerClient(
        model=model, train_loader=train_loader, client_id=client_id,
        gpu_lock_dir=gpu_lock_dir, max_gpu_slots=max_gpu_slots,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAD-UFES-20 FL client")
    p.add_argument("--client-id",      type=int,   required=True, choices=range(5), metavar="[0-4]")
    p.add_argument("--archive",        type=str,   default="archive")
    p.add_argument("--server-address", type=str,   default="127.0.0.1:8080")
    p.add_argument("--num-clients",    type=int,   default=5)
    p.add_argument("--alpha",          type=float, default=0.01)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42,
                   help="RNG seed — must match --seed on the server for identical train/val splits.")
    p.add_argument("--gpu-lock-dir",   type=str,   default=None,
                   help="Shared directory for GPU slot lock files (mount in all client containers).")
    p.add_argument("--max-gpu-slots",  type=int,   default=3,
                   help="Max clients on GPU concurrently (default 3).")
    return p.parse_args()


if __name__ == "__main__":
    args         = parse_args()
    archive_root = os.path.join(os.path.dirname(__file__), args.archive)

    client = build_client(
        client_id=args.client_id,
        archive_root=archive_root,
        num_clients=args.num_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_lock_dir=args.gpu_lock_dir,
        max_gpu_slots=args.max_gpu_slots,
    )
    start_client(server_address=args.server_address, client=client.to_client())
