"""
Baseline ML Training Script - IoT Network Intrusion Detection System

Phase 8.2 + 8.3: Train classical ML models for comparison with Deep Learning.
- Random Forest
- SVM (RBF kernel)
- XGBoost (Gradient Boosting)
- Stratified K-Fold Cross-Validation
- Model saving with joblib
"""

import argparse
import pathlib
import time
import json
from typing import Tuple, Dict, Any, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    if y.dtype.kind in {"U", "S", "O"}:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    return X, y


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_baselines(random_state: int = 42) -> Dict[str, Any]:
    """Create baseline classifiers."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
            max_iter=5000,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        # Fallback to sklearn GradientBoosting
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
        )

    return models


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    results = {}
    for metric in scoring:
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        results[f"cv_{metric}_mean"] = float(cv_results[test_key].mean())
        results[f"cv_{metric}_std"] = float(cv_results[test_key].std())
        results[f"cv_{metric}_train_mean"] = float(cv_results[train_key].mean())

    results["cv_fit_time_mean"] = float(cv_results["fit_time"].mean())
    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline ML models")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train = load_npz(args.data_dir / "train.npz")
    X_test, y_test = load_npz(args.data_dir / "test.npz")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes: {len(np.unique(y_train))}")

    models = create_baselines()
    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name.upper()}")
        print(f"{'='*70}")

        # Cross-validation
        if not args.skip_cv:
            print(f"\nRunning {args.cv_folds}-fold stratified cross-validation...")
            cv_results = run_cross_validation(model, X_train, y_train, n_folds=args.cv_folds)
            print(f"  CV F1 (weighted): {cv_results['cv_f1_weighted_mean']:.4f} "
                  f"(+/- {cv_results['cv_f1_weighted_std']:.4f})")
            print(f"  CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} "
                  f"(+/- {cv_results['cv_accuracy_std']:.4f})")
        else:
            cv_results = {}

        # Train on full training set
        print(f"\nTraining on full training set...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on test set
        t0 = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - t0

        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"\nTest Results:")
        print(classification_report(y_test, y_pred))

        # Store results
        results = {
            "model": name,
            "train_time_seconds": train_time,
            "inference_time_seconds": inference_time,
            "inference_time_per_sample_ms": (inference_time / len(y_test)) * 1000,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "test_precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
            "test_recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
            "classification_report": report,
            **cv_results,
        }
        all_results[name] = results

        # Save model
        model_path = args.output_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")

    # Save all results
    results_path = args.output_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'F1':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Time(s)':>8}")
    print("-" * 70)
    for name, res in all_results.items():
        print(f"{name:<22} {res['test_f1_weighted']:>8.4f} {res['test_accuracy']:>10.4f} "
              f"{res['test_precision_weighted']:>10.4f} {res['test_recall_weighted']:>10.4f} "
              f"{res['train_time_seconds']:>8.1f}")


if __name__ == "__main__":
    main()
