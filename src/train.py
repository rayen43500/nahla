import argparse
import pathlib
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from models import MLP


def load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    if len(X.shape) == 2:
        # ensure dense array
        X = np.array(X.todense()) if hasattr(X, "todense") else np.array(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # encode labels as integers if strings
    if y.dtype.kind in {"U", "S", "O"}:
        classes, y_idx = np.unique(y, return_inverse=True)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
    else:
        classes = np.unique(y)
        y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, classes


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on preprocessed data")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    train_path = args.data_dir / "train.npz"
    val_path = args.data_dir / "val.npz"
    test_path = args.data_dir / "test.npz"
    preprocessor_path = args.data_dir / "preprocessor.joblib"

    X_train, y_train = load_npz(train_path)
    X_val, y_val = load_npz(val_path)
    X_test, y_test = load_npz(test_path)

    train_loader, classes = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader, _ = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader, _ = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    input_dim = train_loader.dataset.tensors[0].shape[1]
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden=args.hidden, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    test_preds, test_labels = eval_model(model, test_loader, device)
    report = classification_report(test_labels, test_preds, target_names=[str(c) for c in classes])
    print(report)

    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "classes": classes,
    }, args.data_dir / "model.pt")
    joblib.dump(classes, args.data_dir / "classes.joblib")
    print("Saved model to", args.data_dir / "model.pt")


if __name__ == "__main__":
    main()
