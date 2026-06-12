import argparse
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from dataset import PADUFES20Dataset, NumericalStats, prepare_train_val_split
from model import build_model


class EarlyStopSignal(Exception):
    """Raised when patience is exhausted; caught in main() for a clean exit."""


# --- Metric aggregation --------------------------------------------------

def weighted_average_fit(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Sample-weighted average of train_loss, train_accuracy, and lr across clients."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        key: sum(n * m.get(key, 0.0) for n, m in metrics if key in m) / total
        for key in ("train_loss", "train_accuracy", "lr")
    }


def weighted_average_evaluate(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Sample-weighted accuracy across clients."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {"accuracy": sum(n * m.get("accuracy", 0.0) for n, m in metrics if "accuracy" in m) / total}


# --- Centralized validation ----------------------------------------------

def build_val_loader(
    archive_root: str,
    val_fraction: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, int, List[int], NumericalStats]:
    """
    Perform the global stratified train/val split, fit the numerical scaler
    on train rows only, and return a DataLoader over the val set.

    Returns:
        loader: DataLoader for centralized evaluation.
        feature_dim: Tabular feature vector width.
        val_indices: Row positions used for validation.
        numerical_stats: Scaler stats fitted on train rows only.
    """
    train_indices, val_indices, numerical_stats, feature_dim = prepare_train_val_split(
        archive_root=archive_root, val_fraction=val_fraction, seed=seed
    )
    val_ds = PADUFES20Dataset(
        archive_root=archive_root, indices=val_indices,
        train=False, numerical_stats=numerical_stats,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[Server] Val loader ready: {len(val_ds)} samples  (fraction={val_fraction:.2f}, seed={seed})")
    return loader, feature_dim, val_indices, numerical_stats


def make_evaluate_fn(
    val_loader: DataLoader,
    feature_dim: int,
    device: str = "cpu",
) -> Callable[[int, np.ndarray, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Return a centralized evaluate_fn compatible with Flower's FedAvg.

    The closure loads global parameters into a fresh model each round,
    runs inference over the server-held val set, and returns
    (loss, {"val_accuracy": ..., "val_loss": ...}).
    """
    criterion = nn.CrossEntropyLoss()

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = build_model(tabular_in_dim=feature_dim, device=device)
        params_tensors   = [torch.tensor(p) for p in parameters]
        state_dict_keys  = list(model.state_dict().keys())

        if len(state_dict_keys) != len(params_tensors):
            print(f"[Server] WARNING: parameter count mismatch "
                  f"({len(params_tensors)} received vs {len(state_dict_keys)} expected). "
                  f"Skipping eval this round.")
            return None

        model.load_state_dict(dict(zip(state_dict_keys, params_tensors)), strict=True)
        model.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
                logits = model(images, tabular)
                total_loss    += criterion(logits, labels).item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

        val_loss = total_loss / max(total_samples, 1)
        val_acc  = total_correct / max(total_samples, 1)
        print(f"[Round {server_round:>3}] CENTRAL VAL  "
              f"loss={val_loss:.4f}  acc={val_acc:.4f}  "
              f"({total_samples} samples)  {datetime.now().strftime('%H:%M:%S')}")
        return val_loss, {"val_accuracy": val_acc, "val_loss": val_loss}

    return evaluate_fn


# --- Strategy ------------------------------------------------------------

class EarlyStoppingFedAvg(FedAvg):
    """
    FedAvg extended with early stopping, best-model checkpointing, and a
    server-managed learning rate schedule.

    When a centralized evaluate_fn is provided, early stopping and
    checkpointing are driven by the centralized val accuracy (15% of
    the dataset, server-held). Federated evaluation is disabled in that
    mode because clients use their full partition for training.

    When no evaluate_fn is provided, the strategy falls back to federated
    aggregate accuracy from clients.

    LR reduction: after `lr_reduce_after` consecutive stale rounds, the LR
    is divided by 10 and the best checkpoint is restored so clients resume
    fine-tuning from the best known weights.
    """

    def __init__(
        self,
        patience: int = 30,
        checkpoint_path: str = "best_model.npz",
        initial_lr: float = 0.01,
        lr_reduce_after: int = 15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.lr_reduce_after = lr_reduce_after
        self.current_lr      = initial_lr

        self.best_accuracy   = 0.0
        self.best_loss       = float("inf")
        self.best_round      = 0
        self.stale_rounds    = 0

        self._lr_reduced_this_plateau = False
        self._latest_parameters: Optional[Parameters] = None
        self._pending_restore: bool = False

        using_central = kwargs.get("evaluate_fn") is not None
        print(f"\n[Server] EarlyStoppingFedAvg")
        print(f"  patience={patience}  lr÷10 after {lr_reduce_after} stale rounds")
        print(f"  initial_lr={initial_lr}  checkpoint={checkpoint_path}")
        print(f"  centralized eval={'YES' if using_central else 'NO (federated fallback)'}\n")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Inject the server-managed LR into every client's fit config.

        If a checkpoint restore is pending (triggered by LR reduction), the
        best-model weights are substituted for Flower's current global parameters.
        """
        if self._pending_restore and self._latest_parameters is not None:
            parameters = self._latest_parameters
            self._pending_restore = False
            print(f"[Round {server_round:>3}] [LR] Restored best-checkpoint weights broadcast to clients.")

        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, FitIns(fit_ins.parameters, {**fit_ins.config, "lr": self.current_lr}))
            for client, fit_ins in client_instructions
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params, metrics = aggregated
            self._latest_parameters = params
            print(f"[Round {server_round:>3}] FIT  clients={len(results)}"
                  f"  loss={metrics.get('train_loss', float('nan')):.4f}"
                  f"  acc={metrics.get('train_accuracy', float('nan')):.4f}"
                  f"  lr={self.current_lr:.2e}")
        return aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Intercept centralized eval result to drive early stopping and checkpointing."""
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
        central_loss, metrics = result
        self._check_and_checkpoint(
            server_round, metrics.get("val_accuracy", 0.0), central_loss,
            source="CENTRAL", parameters=parameters,
        )
        return central_loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return None
        fed_loss, metrics = aggregated
        fed_acc = metrics.get("accuracy", 0.0)
        print(f"[Round {server_round:>3}] FED EVAL  loss={fed_loss:.4f}  acc={fed_acc:.4f}")

        # Use federated accuracy for early stopping only when there is no centralized eval
        if self.evaluate_fn is None:
            self._check_and_checkpoint(
                server_round, fed_acc, fed_loss,
                source="FEDERATED", parameters=self._latest_parameters,
            )
        return fed_loss, metrics

    def _check_and_checkpoint(
        self,
        server_round: int,
        accuracy: float,
        loss: float,
        source: str,
        parameters: Optional[Parameters] = None,
    ) -> None:
        """
        Update best metrics and trigger checkpointing, LR reduction, or early stopping.

        - Improvement  → save checkpoint, reset stale counter.
        - Stale == lr_reduce_after → divide LR by 10, restore best checkpoint.
        - Stale == patience        → raise EarlyStopSignal.
        """
        if accuracy > self.best_accuracy:
            self.best_accuracy            = accuracy
            self.best_loss                = loss
            self.best_round               = server_round
            self.stale_rounds             = 0
            self._lr_reduced_this_plateau = False
            self._save_checkpoint(server_round, accuracy, loss, parameters)
            status = "IMPROVED"
        else:
            self.stale_rounds += 1
            if self.stale_rounds == self.lr_reduce_after and not self._lr_reduced_this_plateau:
                old_lr          = self.current_lr
                self.current_lr = old_lr / 10.0
                self._lr_reduced_this_plateau = True
                print(f"[Round {server_round:>3}] [LR] {self.lr_reduce_after} stale rounds "
                      f"→ LR {old_lr:.2e} → {self.current_lr:.2e}")
                self._restore_best_checkpoint(server_round)
            status = f"no change ({self.stale_rounds}/{self.patience})"

        print(f"[Round {server_round:>3}] [{source}]  "
              f"best_acc={self.best_accuracy:.4f} (round {self.best_round})  "
              f"lr={self.current_lr:.2e}  [{status}]")

        if self.stale_rounds >= self.patience:
            raise EarlyStopSignal(
                f"No improvement for {self.stale_rounds} rounds. "
                f"Best acc={self.best_accuracy:.4f} at round {self.best_round}."
            )

    def _save_checkpoint(
        self, server_round: int, accuracy: float, loss: float, parameters: Optional[Parameters]
    ) -> None:
        """Save current global parameters to a .npz file."""
        params = parameters or self._latest_parameters
        if params is None:
            print("[Server] WARNING: nothing to save yet.")
            return
        out_dir = os.path.dirname(self.checkpoint_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(self.checkpoint_path, *parameters_to_ndarrays(params))
        print(f"[Server] Checkpoint saved  round={server_round}  acc={accuracy:.4f}  → {self.checkpoint_path}")

    def _restore_best_checkpoint(self, server_round: int) -> None:
        """
        Reload the best-model checkpoint into self._latest_parameters.

        On the next configure_fit() call, these weights are broadcast to all
        clients, rolling back the global model to the best known state.
        """
        if not os.path.isfile(self.checkpoint_path):
            print(f"[Round {server_round:>3}] [LR] No checkpoint found at {self.checkpoint_path} — keeping current weights.")
            return
        data   = np.load(self.checkpoint_path)
        arrays = [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]
        self._latest_parameters = ndarrays_to_parameters(arrays)
        self._pending_restore   = True
        print(f"[Round {server_round:>3}] [LR] Best checkpoint loaded "
              f"(acc={self.best_accuracy:.4f} at round {self.best_round}) "
              f"→ broadcast to clients next round")


# --- Entry point ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAD-UFES-20 FL server")
    p.add_argument("--host",             type=str,   default="0.0.0.0")
    p.add_argument("--port",             type=int,   default=8080)
    p.add_argument("--rounds",           type=int,   default=300)
    p.add_argument("--patience",         type=int,   default=30)
    p.add_argument("--min-clients",      type=int,   default=5)
    p.add_argument("--checkpoint",       type=str,   default="checkpoints/best_model.npz")
    p.add_argument("--initial-lr",       type=float, default=0.01,
                   help="Starting LR broadcast to all clients.")
    p.add_argument("--lr-reduce-after",  type=int,   default=15,
                   help="Divide LR by 10 after this many consecutive stale rounds (must be < --patience).")
    p.add_argument("--data-root",        type=str,   default=None,
                   help="Path to the PAD-UFES-20 archive. Enables centralized val evaluation each round.")
    p.add_argument("--val-split",        type=float, default=0.15,
                   help="Fraction reserved for the server val set (default 0.15).")
    p.add_argument("--batch-size",       type=int,   default=32,
                   help="Batch size for centralized validation.")
    p.add_argument("--device",           type=str,   default="cpu",
                   help="Torch device for centralized eval (e.g. 'cpu' or 'cuda').")
    p.add_argument("--seed",             type=int,   default=42,
                   help="RNG seed for the train/val split (must match clients).")
    return p.parse_args()


def main() -> None:
    args           = parse_args()
    server_address = f"{args.host}:{args.port}"

    print(f"\n{'='*65}")
    print(f"  PAD-UFES-20 Federated Learning — Server")
    print(f"{'='*65}")
    print(f"  Address        : {server_address}")
    print(f"  Max rounds     : {args.rounds}")
    print(f"  Early stopping : patience={args.patience}  lr÷10 after {args.lr_reduce_after} stale rounds")
    print(f"  Initial LR     : {args.initial_lr}")
    print(f"  Min clients    : {args.min_clients}")
    print(f"  Checkpoint     : {args.checkpoint}")
    if args.data_root:
        print(f"  Data root      : {args.data_root}")
        print(f"  Val split      : {args.val_split:.0%}  |  Batch size: {args.batch_size}  |  Seed: {args.seed}")
    else:
        print(f"  Centralized eval: DISABLED (pass --data-root to enable)")
    print(f"{'='*65}\n")

    evaluate_fn = None
    if args.data_root:
        val_loader, feature_dim, _, _ = build_val_loader(
            archive_root=args.data_root,
            val_fraction=args.val_split,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        evaluate_fn = make_evaluate_fn(val_loader=val_loader, feature_dim=feature_dim, device=args.device)

    using_central_eval = evaluate_fn is not None
    strategy = EarlyStoppingFedAvg(
        patience=args.patience,
        checkpoint_path=args.checkpoint,
        initial_lr=args.initial_lr,
        lr_reduce_after=args.lr_reduce_after,
        fraction_fit=1.0,
        fraction_evaluate=0.0 if using_central_eval else 1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=0 if using_central_eval else args.min_clients,
        min_available_clients=args.min_clients,
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_evaluate,
        evaluate_fn=evaluate_fn,
    )

    try:
        fl.server.start_server(
            server_address=server_address,
            config=ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    except EarlyStopSignal as ess:
        print(f"\n{'='*65}")
        print(f"  EARLY STOPPING TRIGGERED")
        print(f"  {ess}")
        print(f"  Best model saved to: {args.checkpoint}")
        print(f"{'='*65}\n")
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user.")
    finally:
        print(f"[Server] Done.  Best accuracy={strategy.best_accuracy:.4f} at round {strategy.best_round}.")


if __name__ == "__main__":
    main()
