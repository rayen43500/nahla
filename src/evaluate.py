"""
Evaluation Module - IoT Network Intrusion Detection System

Phase 9: Complete evaluation pipeline with:
- Precision, Recall, F1-score, Accuracy
- TPR / FPR per class
- Confusion matrix (raw + normalized)
- Classification report per class
- Evaluation per attack type (9.2)
- Zero-day analysis (9.3)
"""

import argparse
import pathlib
import json
import time
from typing import Tuple, Dict, Any, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from models import create_model, Autoencoder


# ============================================================================
# DATA LOADING
# ============================================================================

def load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[DataLoader, np.ndarray]:
    X = np.array(X, dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y.dtype.kind in {"U", "S", "O"}:
        classes, y_idx = np.unique(y, return_inverse=True)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
    else:
        classes = np.unique(y)
        y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, classes


# ============================================================================
# INFERENCE
# ============================================================================

def predict_dl(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions, true labels, and probabilities from a DL model."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y_batch.numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.vstack(all_probs)


def predict_ensemble_mlp_rf(
    mlp_checkpoint_path: pathlib.Path,
    rf_model_path: pathlib.Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Predict with weighted probability ensemble: alpha*MLP + (1-alpha)*RF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(mlp_checkpoint_path, map_location=device, weights_only=False)
    classes = np.array(checkpoint["classes"])
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]
    model_kwargs = checkpoint.get("model_kwargs", {})

    mlp_model = create_model("mlp", input_dim, num_classes, **model_kwargs).to(device)
    mlp_model.load_state_dict(checkpoint["model_state"])
    mlp_model.eval()

    rf_model = joblib.load(rf_model_path)

    test_loader, _ = make_loader(X_test, y_test, batch_size)

    t0 = time.time()
    _, y_true, mlp_probs = predict_dl(mlp_model, test_loader, device)
    rf_probs = rf_model.predict_proba(X_test)

    if rf_probs.shape[1] != mlp_probs.shape[1]:
        raise ValueError(
            "Ensemble class count mismatch: "
            f"MLP={mlp_probs.shape[1]} vs RF={rf_probs.shape[1]}"
        )

    ensemble_probs = alpha * mlp_probs + (1.0 - alpha) * rf_probs
    y_pred = np.argmax(ensemble_probs, axis=1)
    inference_time = time.time() - t0

    return y_pred, y_true, ensemble_probs, classes, inference_time


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_probs: Optional[np.ndarray], classes: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    n_classes = len(classes)
    class_names = [str(c) for c in classes]

    # Global metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Classification report per class
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    # Confusion matrix (raw + normalized)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_normalized"] = cm_normalized.tolist()

    # TPR and FPR per class
    tpr_per_class = {}
    fpr_per_class = {}
    for i, cls_name in enumerate(class_names):
        # Binary: this class vs all others
        y_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        tp = np.sum((y_binary == 1) & (y_pred_binary == 1))
        fn = np.sum((y_binary == 1) & (y_pred_binary == 0))
        fp = np.sum((y_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_binary == 0) & (y_pred_binary == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_per_class[cls_name] = float(tpr)
        fpr_per_class[cls_name] = float(fpr)

    metrics["tpr_per_class"] = tpr_per_class
    metrics["fpr_per_class"] = fpr_per_class
    metrics["tpr_mean"] = float(np.mean(list(tpr_per_class.values())))
    metrics["fpr_mean"] = float(np.mean(list(fpr_per_class.values())))

    # ROC AUC per class if probabilities available
    if y_probs is not None and n_classes > 1:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        auc_per_class = {}
        for i, cls_name in enumerate(class_names):
            if y_bin[:, i].sum() > 0:
                fpr_vals, tpr_vals, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                auc_per_class[cls_name] = float(auc(fpr_vals, tpr_vals))
            else:
                auc_per_class[cls_name] = 0.0
        metrics["auc_per_class"] = auc_per_class
        metrics["auc_macro"] = float(np.mean(list(auc_per_class.values())))

    return metrics


# ============================================================================
# PER ATTACK TYPE EVALUATION (Phase 9.2)
# ============================================================================

def evaluate_per_attack_type(y_true: np.ndarray, y_pred: np.ndarray,
                              classes: np.ndarray) -> Dict[str, Any]:
    """Evaluate metrics separately for each attack type."""
    class_names = [str(c) for c in classes]
    results = {}

    for i, cls_name in enumerate(class_names):
        mask = (y_true == i)
        if mask.sum() == 0:
            continue

        y_true_cls = y_true[mask]
        y_pred_cls = y_pred[mask]

        correct = (y_true_cls == y_pred_cls).sum()
        total = len(y_true_cls)

        results[cls_name] = {
            "total_samples": int(total),
            "correctly_classified": int(correct),
            "accuracy": float(correct / total) if total > 0 else 0.0,
            "misclassified_as": {},
        }

        # What were the misclassifications?
        wrong_mask = y_true_cls != y_pred_cls
        if wrong_mask.sum() > 0:
            wrong_preds = y_pred_cls[wrong_mask]
            unique_wrong, counts = np.unique(wrong_preds, return_counts=True)
            for uw, cnt in zip(unique_wrong, counts):
                target_name = class_names[uw] if uw < len(class_names) else str(uw)
                results[cls_name]["misclassified_as"][target_name] = int(cnt)

    # Sort by accuracy (hardest to detect first)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["accuracy"]))

    return {
        "per_class_results": results,
        "hardest_to_detect": list(sorted_results.keys())[:5],
        "easiest_to_detect": list(reversed(list(sorted_results.keys())))[:5],
    }


# ============================================================================
# ZERO-DAY ANALYSIS (Phase 9.3)
# ============================================================================

def zero_day_analysis(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    classes: np.ndarray,
    model_type: str = "mlp",
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Zero-day analysis: train on subset of attack types, test on unseen attack types.
    Measures model robustness to unknown attacks.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    if n_classes < 3:
        return {"error": "Need at least 3 classes for zero-day analysis"}

    results = {}

    # Leave-one-class-out: for each class, train without it, test on it
    for held_out_class in unique_classes:
        class_name = str(classes[held_out_class]) if held_out_class < len(classes) else str(held_out_class)

        # Train set: remove held-out class
        train_mask = y_train != held_out_class
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        # Re-map labels to be contiguous
        remaining_classes = np.unique(y_tr)
        label_map = {old: new for new, old in enumerate(remaining_classes)}
        y_tr_mapped = np.array([label_map[l] for l in y_tr])

        # Test on held-out class
        test_mask = y_test == held_out_class
        if test_mask.sum() == 0:
            results[class_name] = {"skipped": "no test samples for this class"}
            continue

        X_te_held = X_test[test_mask]
        n_held = len(X_te_held)

        # Also test on seen classes
        test_seen_mask = np.isin(y_test, remaining_classes)
        X_te_seen = X_test[test_seen_mask]
        y_te_seen = y_test[test_seen_mask]
        y_te_seen_mapped = np.array([label_map.get(l, -1) for l in y_te_seen])
        valid_seen = y_te_seen_mapped >= 0
        X_te_seen = X_te_seen[valid_seen]
        y_te_seen_mapped = y_te_seen_mapped[valid_seen]

        # Quick train with sklearn RF for zero-day (fast)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr_mapped)

        # Predict on held-out (unseen) class
        y_pred_held = clf.predict(X_te_held)
        # Check if predictions cluster or spread (uncertainty measure)
        pred_entropy = -np.sum(
            [np.mean(y_pred_held == c) * np.log(np.mean(y_pred_held == c) + 1e-10)
             for c in np.unique(y_pred_held)]
        )

        # Accuracy on seen classes
        if len(X_te_seen) > 0:
            y_pred_seen = clf.predict(X_te_seen)
            acc_seen = float(accuracy_score(y_te_seen_mapped, y_pred_seen))
        else:
            acc_seen = 0.0

        results[class_name] = {
            "held_out_samples": int(n_held),
            "prediction_entropy": float(pred_entropy),
            "accuracy_on_seen_classes": acc_seen,
            "prediction_distribution": {
                str(c): int(cnt) for c, cnt in
                zip(*np.unique(y_pred_held, return_counts=True))
            },
        }

    return {
        "zero_day_results": results,
        "n_classes_total": int(n_classes),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data/preprocessed"))
    parser.add_argument("--model-path", type=pathlib.Path, required=True,
                        help="Path to saved model (.pt for DL, .joblib for ML)")
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "cnn", "autoencoder", "hybrid",
                                 "random_forest", "svm", "xgboost", "gradient_boosting",
                                 "ensemble_mlp_rf"])
    parser.add_argument("--rf-model-path", type=pathlib.Path, default=pathlib.Path("models/random_forest_model.joblib"),
                        help="Random Forest model path for ensemble_mlp_rf")
    parser.add_argument("--ensemble-alpha", type=float, default=0.6,
                        help="Weight for MLP probs in ensemble: alpha*MLP + (1-alpha)*RF")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--zero-day", action="store_true", help="Run zero-day analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test, y_test = load_npz(args.data_dir / "test.npz")

    is_dl = args.model_type in {"mlp", "lstm", "cnn", "autoencoder", "hybrid"}

    if args.model_type == "ensemble_mlp_rf":
        if not (0.0 <= args.ensemble_alpha <= 1.0):
            raise ValueError("--ensemble-alpha must be between 0 and 1")
        y_pred, y_true, y_probs, classes, inference_time = predict_ensemble_mlp_rf(
            mlp_checkpoint_path=args.model_path,
            rf_model_path=args.rf_model_path,
            X_test=X_test,
            y_test=y_test,
            batch_size=args.batch_size,
            alpha=args.ensemble_alpha,
        )
    elif is_dl:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        classes = checkpoint["classes"]
        input_dim = checkpoint["input_dim"]
        num_classes = checkpoint["num_classes"]
        model_kwargs = checkpoint.get("model_kwargs", {})

        model = create_model(args.model_type, input_dim, num_classes, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])

        test_loader, _ = make_loader(X_test, y_test, args.batch_size)

        t0 = time.time()
        y_pred, y_true, y_probs = predict_dl(model, test_loader, device)
        inference_time = time.time() - t0
    else:
        model = joblib.load(args.model_path)
        if y_test.dtype.kind in {"U", "S", "O"}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_test = le.fit_transform(y_test)
            classes = le.classes_
        else:
            classes = np.unique(y_test)

        t0 = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - t0
        y_true = y_test
        y_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    print(f"Inference time: {inference_time:.3f}s ({(inference_time/len(y_true))*1000:.3f} ms/sample)")

    # Compute metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    metrics = compute_metrics(y_true, y_pred, y_probs, classes)
    metrics["inference_time_seconds"] = inference_time
    metrics["inference_time_per_sample_ms"] = (inference_time / len(y_true)) * 1000

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"Precision:     {metrics['precision_weighted']:.4f}")
    print(f"Recall:        {metrics['recall_weighted']:.4f}")
    print(f"Mean TPR:      {metrics['tpr_mean']:.4f}")
    print(f"Mean FPR:      {metrics['fpr_mean']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0))

    # Per attack type evaluation
    print("\n" + "=" * 70)
    print("PER ATTACK TYPE ANALYSIS")
    print("=" * 70)
    attack_analysis = evaluate_per_attack_type(y_true, y_pred, classes)
    metrics["per_attack_analysis"] = attack_analysis

    print(f"\nHardest to detect: {attack_analysis['hardest_to_detect']}")
    print(f"Easiest to detect: {attack_analysis['easiest_to_detect']}")

    for cls_name, res in attack_analysis["per_class_results"].items():
        print(f"  {cls_name}: {res['accuracy']:.4f} accuracy ({res['correctly_classified']}/{res['total_samples']})")

    # Zero-day analysis
    if args.zero_day:
        print("\n" + "=" * 70)
        print("ZERO-DAY ANALYSIS")
        print("=" * 70)
        X_train, y_train = load_npz(args.data_dir / "train.npz")
        zd = zero_day_analysis(X_train, y_train, X_test, y_test if y_test.dtype != object else y_true,
                               classes, args.model_type)
        metrics["zero_day_analysis"] = zd
        for cls_name, res in zd.get("zero_day_results", {}).items():
            if "skipped" in res:
                print(f"  {cls_name}: skipped ({res['skipped']})")
            else:
                print(f"  {cls_name}: entropy={res['prediction_entropy']:.3f}, "
                      f"seen_acc={res['accuracy_on_seen_classes']:.4f}")

    # Save all metrics
    output_path = args.output_dir / f"{args.model_type}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {output_path}")

    # Save confusion matrix as numpy
    cm_path = args.output_dir / f"{args.model_type}_confusion_matrix.npy"
    np.save(cm_path, np.array(metrics["confusion_matrix"]))
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
