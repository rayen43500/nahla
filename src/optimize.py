"""
Optimization and Hyperparameter Tuning - IoT Network Intrusion Detection System

Phase 11: 
- Hyperparameter tuning with Optuna (11.1)
- Architecture search (11.2)
- Model optimization: quantization, ONNX export (11.3)
"""

import argparse
import pathlib
import json
import time
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import create_model
from train import load_npz, make_loader, train_one_epoch, compute_val_loss, EarlyStopping


# ============================================================================
# OPTUNA HYPERPARAMETER TUNING (Phase 11.1)
# ============================================================================

def run_optuna_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    model_type: str = "mlp",
    n_trials: int = 30,
    max_epochs: int = 30,
    batch_size: int = 128,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for a given model type."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not installed. Install with: pip install optuna")
        return {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, classes = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader, _ = make_loader(X_val, y_val, batch_size, shuffle=False)
    input_dim = train_loader.dataset.tensors[0].shape[1]
    num_classes = len(classes)

    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model_kwargs = {}
        if model_type == "mlp":
            hidden = trial.suggest_categorical("hidden", [128, 256, 512])
            model_kwargs = {"hidden": hidden, "dropout": dropout}
        elif model_type == "lstm":
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 2, 4)
            bidirectional = trial.suggest_categorical("bidirectional", [True, False])
            model_kwargs = {"hidden_dim": hidden_dim, "num_layers": num_layers,
                           "dropout": dropout, "bidirectional": bidirectional}
        elif model_type == "cnn":
            num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
            model_kwargs = {"num_filters": num_filters, "kernel_sizes": (3, 5, 7), "dropout": dropout}
        elif model_type == "hybrid":
            cnn_filters = trial.suggest_categorical("cnn_filters", [32, 64, 128])
            lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
            model_kwargs = {"cnn_filters": cnn_filters, "lstm_hidden": lstm_hidden,
                           "lstm_layers": 2, "dropout": dropout}

        model = create_model(model_type, input_dim, num_classes, **model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping(patience=5)

        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = compute_val_loss(model, val_loader, criterion, device)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if early_stopper.step(val_loss):
                break

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    return {
        "best_params": best.params,
        "best_value": best.value,
        "n_trials": len(study.trials),
        "all_trials": [
            {"params": t.params, "value": t.value, "state": str(t.state)}
            for t in study.trials if t.value is not None
        ],
    }


# ============================================================================
# ARCHITECTURE SEARCH (Phase 11.2)
# ============================================================================

def architecture_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    model_types: list = None,
    max_epochs: int = 20,
    batch_size: int = 128,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Compare different architectures with default hyperparameters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_types is None:
        model_types = ["mlp", "lstm", "cnn", "hybrid"]

    train_loader, classes = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader, _ = make_loader(X_val, y_val, batch_size, shuffle=False)
    input_dim = train_loader.dataset.tensors[0].shape[1]
    num_classes = len(classes)

    results = {}

    for model_type in model_types:
        print(f"\nTesting architecture: {model_type}")

        model_kwargs = {}
        if model_type == "lstm":
            model_kwargs = {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3, "bidirectional": True}
        elif model_type == "cnn":
            model_kwargs = {"num_filters": 64, "kernel_sizes": (3, 5, 7), "dropout": 0.3}
        elif model_type == "hybrid":
            model_kwargs = {"cnn_filters": 64, "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.3}
        else:
            model_kwargs = {"hidden": 256, "dropout": 0.3}

        model = create_model(model_type, input_dim, num_classes, **model_kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        early_stopper = EarlyStopping(patience=7)

        best_val_loss = float("inf")
        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = compute_val_loss(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if early_stopper.step(val_loss):
                break

        train_time = time.time() - t0

        results[model_type] = {
            "best_val_loss": float(best_val_loss),
            "total_parameters": n_params,
            "training_time_seconds": float(train_time),
            "epochs_trained": epoch,
        }
        print(f"  val_loss={best_val_loss:.5f} | params={n_params:,} | time={train_time:.1f}s")

    # Rank by validation loss
    ranked = sorted(results.items(), key=lambda x: x[1]["best_val_loss"])
    results["ranking"] = [r[0] for r in ranked]

    return results


# ============================================================================
# MODEL OPTIMIZATION (Phase 11.3)
# ============================================================================

def export_onnx(model: nn.Module, input_dim: int, output_path: pathlib.Path,
                batch_size: int = 1) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(batch_size, input_dim)
    torch.onnx.export(
        model, dummy_input, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"ONNX model exported to {output_path}")


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization (CPU only, INT8)."""
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )
    return quantized


def benchmark_inference(model: nn.Module, input_dim: int,
                        n_samples: int = 1000, device: torch.device = None) -> Dict[str, float]:
    """Benchmark inference speed."""
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    dummy = torch.randn(n_samples, input_dim).to(device)

    # Warmup
    with torch.no_grad():
        _ = model(dummy[:10])

    # Benchmark
    t0 = time.time()
    with torch.no_grad():
        _ = model(dummy)
    elapsed = time.time() - t0

    return {
        "total_time_seconds": elapsed,
        "time_per_sample_ms": (elapsed / n_samples) * 1000,
        "samples_per_second": n_samples / elapsed,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize and tune models")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "cnn", "hybrid"])
    parser.add_argument("--mode", type=str, default="tune",
                        choices=["tune", "arch_search", "export_onnx", "quantize", "benchmark"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("tune", "arch_search"):
        X_train, y_train = load_npz(args.data_dir / "train.npz")
        X_val, y_val = load_npz(args.data_dir / "val.npz")

    if args.mode == "tune":
        print(f"Running Optuna tuning for {args.model_type} ({args.n_trials} trials)...")
        results = run_optuna_search(X_train, y_train, X_val, y_val,
                                     model_type=args.model_type, n_trials=args.n_trials, device=device)
        output_path = args.output_dir / f"{args.model_type}_optuna_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Best params: {results.get('best_params')}")
        print(f"Results saved to {output_path}")

    elif args.mode == "arch_search":
        print("Running architecture search...")
        results = architecture_search(X_train, y_train, X_val, y_val, device=device)
        output_path = args.output_dir / "architecture_search_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nRanking: {results['ranking']}")
        print(f"Results saved to {output_path}")

    elif args.mode == "export_onnx":
        if not args.model_path or not args.model_path.exists():
            print("Error: --model-path required for ONNX export")
            return
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model = create_model(args.model_type, checkpoint["input_dim"],
                            checkpoint["num_classes"], **checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state"])
        export_onnx(model, checkpoint["input_dim"],
                   args.output_dir / f"{args.model_type}.onnx")

    elif args.mode == "quantize":
        if not args.model_path or not args.model_path.exists():
            print("Error: --model-path required for quantization")
            return
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model = create_model(args.model_type, checkpoint["input_dim"],
                            checkpoint["num_classes"], **checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state"])

        print("Before quantization:")
        bm_before = benchmark_inference(model, checkpoint["input_dim"])
        print(f"  {bm_before['time_per_sample_ms']:.3f} ms/sample")

        q_model = quantize_model(model)
        print("After quantization:")
        bm_after = benchmark_inference(q_model, checkpoint["input_dim"])
        print(f"  {bm_after['time_per_sample_ms']:.3f} ms/sample")

        torch.save({"model": q_model, "input_dim": checkpoint["input_dim"],
                    "num_classes": checkpoint["num_classes"]},
                   args.output_dir / f"{args.model_type}_quantized.pt")

    elif args.mode == "benchmark":
        if not args.model_path or not args.model_path.exists():
            print("Error: --model-path required for benchmark")
            return
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model = create_model(args.model_type, checkpoint["input_dim"],
                            checkpoint["num_classes"], **checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state"])
        results = benchmark_inference(model, checkpoint["input_dim"], device=device)
        print(f"Inference: {results['time_per_sample_ms']:.3f} ms/sample")
        print(f"Throughput: {results['samples_per_second']:.0f} samples/s")


if __name__ == "__main__":
    main()
