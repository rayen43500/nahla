"""
Training Script - IoT Network Intrusion Detection System

Phase 8.1: Complete training pipeline with:
- Early stopping (validation loss)
- Learning rate scheduling (ReduceLROnPlateau)
- Model checkpointing (best model)
- TensorBoard logging
- Final model save (.pt/.pth)
- Support for all model architectures (MLP, LSTM, CNN, Hybrid, Autoencoder)
"""

import argparse
import pathlib
import time
import json
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models import create_model, Autoencoder


# ============================================================================
# DATA LOADING
# ============================================================================

def load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Tuple[DataLoader, np.ndarray]:
    X = np.array(X.todense()) if hasattr(X, "todense") else np.array(X, dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    if y.dtype.kind in {"U", "S", "O"}:
        classes, y_idx = np.unique(y, return_inverse=True)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
    else:
        classes = np.unique(y)
        y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, classes


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if hasattr(optimizer, "_grad_clip_value") and optimizer._grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer._grad_clip_value)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def train_autoencoder_epoch(model: Autoencoder, loader: DataLoader, optimizer, device: torch.device) -> float:
    """Train autoencoder with MSE reconstruction loss."""
    model.train()
    total_loss = 0.0
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        x_hat = model(X_batch)
        loss = nn.functional.mse_loss(x_hat, X_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_autoencoder(model: Autoencoder, loader: DataLoader, device: torch.device) -> float:
    """Evaluate autoencoder reconstruction loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            x_hat = model(X_batch)
            loss = nn.functional.mse_loss(x_hat, X_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def compute_val_loss(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DL model on preprocessed data")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data/preprocessed"))
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "cnn", "autoencoder", "hybrid"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument("--class-weights", type=pathlib.Path, default=None,
                        help="Path to class_weights.joblib")
    parser.add_argument("--auto-class-weights", action="store_true",
                        help="Compute inverse-frequency class weights from y_train")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (set <= 0 to disable)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_files = ["train.npz", "val.npz", "test.npz"]
    missing = [name for name in required_files if not (args.data_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required file(s) in {args.data_dir}: {missing_str}. "
            "Run data preparation first or pass --data-dir to the folder containing train.npz/val.npz/test.npz."
        )

    # Load data
    X_train, y_train = load_npz(args.data_dir / "train.npz")
    X_val, y_val = load_npz(args.data_dir / "val.npz")
    X_test, y_test = load_npz(args.data_dir / "test.npz")

    train_loader, classes = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader, _ = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader, _ = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    input_dim = train_loader.dataset.tensors[0].shape[1]
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model_type} | Input dim: {input_dim} | Classes: {num_classes}")

    # Create model
    model_kwargs = {"hidden": args.hidden, "dropout": args.dropout}
    if args.model_type == "lstm":
        model_kwargs = {"hidden_dim": args.hidden, "num_layers": 2, "dropout": args.dropout, "bidirectional": True}
    elif args.model_type == "cnn":
        model_kwargs = {"num_filters": 64, "kernel_sizes": (3, 5, 7), "dropout": args.dropout}
    elif args.model_type == "autoencoder":
        model_kwargs = {"bottleneck_dim": 16, "hidden_dims": (128, 64), "dropout": args.dropout}
    elif args.model_type == "hybrid":
        model_kwargs = {"cnn_filters": 64, "lstm_hidden": 128, "lstm_layers": 2, "dropout": args.dropout}

    model = create_model(args.model_type, input_dim, num_classes, **model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    is_autoencoder = args.model_type == "autoencoder"

    # Loss with optional class weights
    if is_autoencoder:
        criterion = None  # Uses MSE internally
    else:
        if args.class_weights and args.class_weights.exists():
            weights_dict = joblib.load(args.class_weights)
            weight_tensor = torch.tensor([weights_dict.get(i, 1.0) for i in range(num_classes)],
                                         dtype=torch.float32).to(device)
            print(f"Using class weights from file: {args.class_weights}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        elif args.auto_class_weights:
            class_counts = np.bincount(train_loader.dataset.tensors[1].numpy(), minlength=num_classes).astype(np.float32)
            class_counts[class_counts == 0] = 1.0
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * num_classes
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Using auto class weights: {weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer._grad_clip_value = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience)

    # TensorBoard
    run_name = f"{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(args.log_dir / run_name))

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": [], "val_macro_f1": []}

    print(f"\nStarting training for {args.epochs} epochs (patience={args.patience})...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if is_autoencoder:
            train_loss = train_autoencoder_epoch(model, train_loader, optimizer, device)
            val_loss = eval_autoencoder(model, val_loader, device)
        else:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = compute_val_loss(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        val_macro_f1 = float("nan")
        if not is_autoencoder:
            val_preds, val_labels = eval_model(model, val_loader, device)
            val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        history["val_macro_f1"].append(val_macro_f1)

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        if not is_autoencoder:
            writer.add_scalar("F1/val_macro", val_macro_f1, epoch)

        # Checkpoint best model
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            improved = " *"
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, args.output_dir / f"{args.model_type}_checkpoint.pt")

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
              f"val_macro_f1={val_macro_f1:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s{improved}")

        # LR scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if early_stopper.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    if is_autoencoder:
        test_loss = eval_autoencoder(model, test_loader, device)
        print(f"Autoencoder test reconstruction loss: {test_loss:.6f}")
        report_dict = {"test_reconstruction_loss": test_loss}
    else:
        test_preds, test_labels = eval_model(model, test_loader, device)
        report = classification_report(test_labels, test_preds,
                                       target_names=[str(c) for c in classes])
        print(report)
        macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
        macro_recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)
        print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Macro Recall: {macro_recall:.4f}")
        report_dict = classification_report(test_labels, test_preds,
                                            target_names=[str(c) for c in classes],
                                            output_dict=True)
        report_dict["summary"] = {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_recall": float(macro_recall),
        }

    # Save final model
    save_path = args.output_dir / f"{args.model_type}_best.pt"
    save_data = {
        "model_state": model.state_dict(),
        "model_type": args.model_type,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes": classes,
        "model_kwargs": model_kwargs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(save_data, save_path)
    print(f"\nSaved best model to {save_path}")

    # Save training history
    history_path = args.output_dir / f"{args.model_type}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save metrics
    metrics_path = args.output_dir / f"{args.model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    writer.close()
    print(f"TensorBoard logs: {args.log_dir / run_name}")
    print("Training complete!")


if __name__ == "__main__":
    main()
