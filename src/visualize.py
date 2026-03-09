"""
Visualization Module - IoT Network Intrusion Detection System

Phase 10: Complete visualization and analysis pipeline:
- ROC curves (One-vs-Rest)
- Precision-Recall curves
- Confusion matrix heatmaps
- Distribution of predictions per class
- Training loss/accuracy curves
- Learning curves
- Feature importance
- Comparison report DL vs ML
"""

import argparse
import pathlib
import json
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix,
)
from sklearn.preprocessing import label_binarize


# ============================================================================
# ROC CURVES (Phase 10.1)
# ============================================================================

def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    class_names: List[str], output_path: pathlib.Path) -> None:
    """Plot One-vs-Rest ROC curves for all classes."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {output_path}")


# ============================================================================
# PRECISION-RECALL CURVES (Phase 10.1)
# ============================================================================

def plot_pr_curves(y_true: np.ndarray, y_probs: np.ndarray,
                   class_names: List[str], output_path: pathlib.Path) -> None:
    """Plot Precision-Recall curves for all classes."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(recall, precision, color=color, lw=2, label=f"{cls_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curves saved to {output_path}")


# ============================================================================
# CONFUSION MATRIX HEATMAP (Phase 10.1)
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          output_path: pathlib.Path, normalize: bool = True) -> None:
    """Plot confusion matrix as heatmap."""
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


# ============================================================================
# PREDICTION DISTRIBUTION (Phase 10.1)
# ============================================================================

def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: List[str], output_path: pathlib.Path) -> None:
    """Plot distribution of predictions per class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # True label distribution
    true_classes, true_counts = np.unique(y_true, return_counts=True)
    axes[0].barh([class_names[c] for c in true_classes], true_counts, color="steelblue")
    axes[0].set_title("True Label Distribution", fontsize=12)
    axes[0].set_xlabel("Count")

    # Predicted label distribution
    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    axes[1].barh([class_names[c] for c in pred_classes], pred_counts, color="coral")
    axes[1].set_title("Predicted Label Distribution", fontsize=12)
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction distribution saved to {output_path}")


# ============================================================================
# TRAINING CURVES (Phase 10.2)
# ============================================================================

def plot_training_curves(history_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Plot training and validation loss/accuracy curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate curve
    if "lr" in history:
        axes[1].plot(epochs, history["lr"], "g-", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


# ============================================================================
# FEATURE IMPORTANCE (Phase 10.2)
# ============================================================================

def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                            output_path: pathlib.Path, top_k: int = 20) -> None:
    """Plot top-K feature importances."""
    indices = np.argsort(importances)[-top_k:]
    top_importances = importances[indices]
    top_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    ax.barh(range(len(top_importances)), top_importances, color="steelblue")
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance saved to {output_path}")


# ============================================================================
# COMPARISON REPORT (Phase 10.3)
# ============================================================================

def generate_comparison_report(results_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    """Generate comparison table: DL vs ML classic."""
    metrics_files = list(results_dir.glob("*_metrics.json"))

    if not metrics_files:
        print("No metrics files found for comparison.")
        return

    rows = []
    for mf in metrics_files:
        with open(mf, "r") as f:
            data = json.load(f)
        model_name = mf.stem.replace("_metrics", "")
        rows.append({
            "Model": model_name,
            "Accuracy": data.get("accuracy", data.get("test_accuracy", "N/A")),
            "F1 (weighted)": data.get("f1_weighted", data.get("test_f1_weighted", "N/A")),
            "Precision": data.get("precision_weighted", data.get("test_precision_weighted", "N/A")),
            "Recall": data.get("recall_weighted", data.get("test_recall_weighted", "N/A")),
            "Inference (ms/sample)": data.get("inference_time_per_sample_ms", "N/A"),
        })

    # Create comparison table
    report_lines = ["# Model Comparison Report\n"]
    report_lines.append("| Model | Accuracy | F1 (weighted) | Precision | Recall | Inference (ms/sample) |")
    report_lines.append("|-------|----------|---------------|-----------|--------|-----------------------|")

    for row in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        report_lines.append(
            f"| {row['Model']} | {fmt(row['Accuracy'])} | {fmt(row['F1 (weighted)'])} | "
            f"{fmt(row['Precision'])} | {fmt(row['Recall'])} | {fmt(row['Inference (ms/sample)'])} |"
        )

    report_text = "\n".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nComparison report saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--models-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--history", type=pathlib.Path, default=None,
                        help="Path to training history JSON for loss curves")
    parser.add_argument("--comparison", action="store_true", help="Generate comparison report")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison report if requested
    if args.comparison:
        generate_comparison_report(args.results_dir, args.output_dir / "comparison_report.md")

    # Plot training curves if history available
    if args.history and args.history.exists():
        plot_training_curves(args.history, args.output_dir / "training_curves.png")

    # Auto-discover and plot confusion matrices
    for cm_file in args.results_dir.glob("*_confusion_matrix.npy"):
        model_name = cm_file.stem.replace("_confusion_matrix", "")
        cm = np.load(cm_file)

        # Try to load class names
        metrics_file = args.results_dir / f"{model_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            report = metrics.get("classification_report", {})
            class_names = [k for k in report.keys() if k not in
                          {"accuracy", "macro avg", "weighted avg"}]
        else:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]

        # Raw and normalized
        plot_confusion_matrix(cm, class_names,
                             args.output_dir / f"{model_name}_confusion_matrix.png", normalize=False)
        plot_confusion_matrix(cm, class_names,
                             args.output_dir / f"{model_name}_confusion_matrix_normalized.png", normalize=True)

    # Auto-discover feature importances from baseline models
    for model_file in args.models_dir.glob("random_forest_model.joblib"):
        import joblib
        model = joblib.load(model_file)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
            plot_feature_importance(importances, feature_names,
                                  args.output_dir / "feature_importance.png")

    print("\nAll visualizations generated!")


if __name__ == "__main__":
    main()
