import argparse
import pathlib
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def resolve_label_column(df: pd.DataFrame, requested_label: str) -> str:
    """Resolve label column name robustly (exact, trimmed, and case-insensitive)."""
    if requested_label in df.columns:
        return requested_label

    requested_norm = requested_label.strip().lower()

    # Match when CSV headers contain leading/trailing spaces.
    for col in df.columns:
        if col.strip() == requested_label.strip():
            return col

    # Match case-insensitively after stripping spaces.
    for col in df.columns:
        if col.strip().lower() == requested_norm:
            return col

    available = ", ".join(df.columns.tolist())
    raise ValueError(
        f"Label column '{requested_label}' not found. Available columns: {available}"
    )


def sanitize_feature_values(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Sanitize feature values to handle common CICIDS issues (spaces, inf strings)."""
    cleaned = df.copy()
    feature_cols = [c for c in cleaned.columns if c != label_col]

    for col in feature_cols:
        series = cleaned[col]

        if series.dtype == object:
            stripped = series.astype(str).str.strip()
            numeric_candidate = pd.to_numeric(stripped, errors="coerce")

            non_null_count = int(series.notna().sum())
            if non_null_count > 0:
                numeric_ratio = float(numeric_candidate.notna().sum()) / float(non_null_count)
                # Convert object columns that are mostly numeric strings.
                if numeric_ratio >= 0.95:
                    cleaned[col] = numeric_candidate

    replacement_tokens = [np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"]
    cleaned[feature_cols] = cleaned[feature_cols].replace(replacement_tokens, np.nan)
    return cleaned


def merge_webattack_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Merge sparse WebAttack sub-classes into a single WebAttack class."""
    merged = df.copy()

    def _map_label(v: Any) -> Any:
        s = str(v).strip().lower()
        if "web attack" in s or "sql injection" in s or s == "xss" or "brute force" in s:
            return "WebAttack"
        return str(v).strip()

    merged[label_col] = merged[label_col].map(_map_label)
    return merged


def simplify_ids_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Simplify labels to 4 IDS buckets: BENIGN, PortScan, DDoS, WebAttack."""
    simplified = df.copy()

    def _simplify(v: Any) -> str:
        s = str(v).strip().lower()
        if "benign" in s:
            return "BENIGN"
        if "portscan" in s or "port scan" in s:
            return "PortScan"
        if "ddos" in s:
            return "DDoS"
        return "WebAttack"

    simplified[label_col] = simplified[label_col].map(_simplify)
    return simplified


def filter_rare_classes(df: pd.DataFrame, label_col: str, min_samples: int) -> pd.DataFrame:
    """Drop classes with fewer than min_samples examples."""
    if min_samples <= 1:
        return df

    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= min_samples].index
    filtered = df[df[label_col].isin(valid_classes)].copy()
    return filtered


def detect_missing_values(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """Detect and report missing values statistics."""
    feature_cols = [c for c in df.columns if c != label_col]
    missing_stats = {}
    
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            missing_stats[col] = {
                "count": missing_count,
                "percentage": missing_pct,
                "dtype": str(df[col].dtype)
            }
    
    return missing_stats


def analyze_class_distribution(y: pd.Series) -> Dict[str, Any]:
    """Analyze class distribution and imbalance."""
    class_counts = y.value_counts().to_dict()
    class_dist = (y.value_counts(normalize=True) * 100).to_dict()
    
    if len(class_counts) > 0:
        max_class_count = max(class_counts.values())
        min_class_count = min(class_counts.values())
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 0
    else:
        imbalance_ratio = 0
    
    return {
        "class_counts": class_counts,
        "class_distribution_pct": class_dist,
        "imbalance_ratio": imbalance_ratio,
        "num_classes": len(class_counts)
    }


def build_preprocessor(df: pd.DataFrame, label_col: str, use_robust_scaler: bool = True) -> ColumnTransformer:
    """Build preprocessing pipeline with RobustScaler for outlier handling."""
    feature_cols = [c for c in df.columns if c != label_col]
    cat_cols = [c for c in feature_cols if df[c].dtype == object or str(df[c].dtype).startswith("category") or str(df[c].dtype).startswith("string") or str(df[c].dtype) == "str"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    # Use RobustScaler if specified (robust to outliers), otherwise StandardScaler
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    preprocessor.cat_cols_ = cat_cols  # type: ignore[attr-defined]
    preprocessor.num_cols_ = num_cols  # type: ignore[attr-defined]
    return preprocessor


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42,
                preferred_k_neighbors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE for class imbalance handling."""
    # Check if all classes have enough samples for SMOTE (requires at least k_neighbors)
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    if preferred_k_neighbors is not None:
        k_neighbors = min(max(1, preferred_k_neighbors), max(1, min_count - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    elif min_count < 6:
        # Reduce k_neighbors if class is too small
        k_neighbors = max(1, min_count - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    else:
        smote = SMOTE(random_state=random_state)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Warning: SMOTE failed ({e}), returning original data")
        return X, y


def apply_pca(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, 
              variance_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=variance_ratio, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA: Reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} dimensions")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_val_pca, X_test_pca, pca


def split_data(df: pd.DataFrame, label_col: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    y = df[label_col]
    X = df.drop(columns=[label_col])
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size + val_size, stratify=y, random_state=seed)
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=relative_test, stratify=y_tmp, random_state=seed)
    train = X_train.copy(); train[label_col] = y_train
    val = X_val.copy(); val[label_col] = y_val
    test = X_test.copy(); test[label_col] = y_test
    return train, val, test


def transform_and_save(pre: ColumnTransformer, df: pd.DataFrame, label_col: str, out_path: pathlib.Path,
                       apply_smote_flag: bool = False, pca_model: Optional[PCA] = None) -> Dict[str, Any]:
    """Transform and save data with optional SMOTE and PCA."""
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    X_t = pre.transform(X)
    
    # Apply SMOTE if requested (only for training set)
    if apply_smote_flag:
        X_t, y = apply_smote(X_t, y)
    
    # Apply PCA if model provided
    if pca_model is not None:
        X_t = pca_model.transform(X_t)
    
    np.savez_compressed(out_path, X=X_t, y=y)
    
    return {
        "samples": len(y),
        "features": X_t.shape[1],
        "classes": len(np.unique(y))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset: split, encode, scale, handle imbalance, reduce dims")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Input CSV file")
    parser.add_argument("--label", type=str, default="label", help="Label column name")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"), help="Output directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE for class imbalance (train only)")
    parser.add_argument("--smote-k-neighbors", type=int, default=None,
                        help="SMOTE k_neighbors override (e.g., 3 for small classes)")
    parser.add_argument("--pca", type=float, default=None, help="Apply PCA with variance ratio (e.g., 0.95)")
    parser.add_argument("--class-weights", action="store_true", help="Compute class weights for imbalance handling")
    parser.add_argument("--min-samples-per-class", type=int, default=1,
                        help="Drop classes with fewer than this number of samples")
    parser.add_argument("--merge-webattacks", action="store_true",
                        help="Merge Web Attack sub-classes (SQLi/XSS/Brute Force) into 'WebAttack'")
    parser.add_argument("--simplify-labels", action="store_true",
                        help="Map labels to 4 classes: BENIGN, PortScan, DDoS, WebAttack")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    label_col = resolve_label_column(df, args.label)
    df = sanitize_feature_values(df, label_col)
    if label_col != args.label:
        print(f"Resolved label column '{args.label}' -> '{label_col}'")

    if args.simplify_labels:
        before_classes = int(df[label_col].nunique())
        df = simplify_ids_labels(df, label_col)
        after_classes = int(df[label_col].nunique())
        print(f"Simplified labels to IDS buckets: {before_classes} -> {after_classes} classes")

    if args.merge_webattacks:
        before_classes = int(df[label_col].nunique())
        df = merge_webattack_labels(df, label_col)
        after_classes = int(df[label_col].nunique())
        print(f"Merged WebAttack sub-classes: {before_classes} -> {after_classes} classes")

    if args.min_samples_per_class > 1:
        before_size = len(df)
        before_classes = int(df[label_col].nunique())
        df = filter_rare_classes(df, label_col, args.min_samples_per_class)
        after_size = len(df)
        after_classes = int(df[label_col].nunique())
        removed = before_size - after_size
        print(
            f"Filtered rare classes (<{args.min_samples_per_class} samples): "
            f"removed {removed} rows, classes {before_classes} -> {after_classes}"
        )

    if df.empty:
        raise ValueError("No samples left after class filtering. Lower --min-samples-per-class.")

    # Detect and report missing values
    print("=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    missing_stats = detect_missing_values(df, label_col)
    if missing_stats:
        print(f"Found missing values in {len(missing_stats)} columns:")
        for col, stats in missing_stats.items():
            print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
    else:
        print("No missing values detected!")
    
    # Analyze class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    class_analysis = analyze_class_distribution(df[label_col])
    print(f"Number of classes: {class_analysis['num_classes']}")
    print(f"Imbalance ratio: {class_analysis['imbalance_ratio']:.3f}x")
    print("Class distribution:")
    for cls, count in class_analysis['class_counts'].items():
        dist_pct = class_analysis['class_distribution_pct'][cls]
        print(f"  Class {cls}: {count} samples ({dist_pct:.2f}%)")

    weak_classes = [cls for cls, count in class_analysis["class_counts"].items() if count < 1000]
    if weak_classes:
        print("Warning: classes below 1000 samples detected:", ", ".join(map(str, weak_classes)))
    
    # Split data
    train_df, val_df, test_df = split_data(df, label_col=label_col, test_size=args.test_size, val_size=args.val_size)
    pre = build_preprocessor(train_df, label_col=label_col, use_robust_scaler=True)
    pre.fit(train_df.drop(columns=[label_col]))

    # Transform data
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION")
    print("=" * 60)
    train_stats = transform_and_save(pre, train_df, label_col, args.outdir / "train_raw.npz")
    val_stats = transform_and_save(pre, val_df, label_col, args.outdir / "val.npz")
    test_stats = transform_and_save(pre, test_df, label_col, args.outdir / "test.npz")
    
    # Load transformed training data for SMOTE and PCA
    train_data = np.load(args.outdir / "train_raw.npz", allow_pickle=True)
    X_train, y_train = train_data['X'].copy(), train_data['y'].copy()
    train_data.close()
    
    # Apply SMOTE if requested
    pca_model = None
    if args.smote:
        print("\n" + "=" * 60)
        print("APPLYING SMOTE FOR CLASS IMBALANCE")
        print("=" * 60)
        print(f"Train set before SMOTE: {len(y_train)} samples")
        X_train, y_train = apply_smote(X_train, y_train, preferred_k_neighbors=args.smote_k_neighbors)
        print(f"Train set after SMOTE: {len(y_train)} samples")
        
        # Update class distribution after SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution after SMOTE:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples")
    
    # Apply PCA if requested
    pca_model = None
    X_val = None
    X_test = None
    if args.pca:
        print("\n" + "=" * 60)
        print("APPLYING PCA FOR DIMENSIONALITY REDUCTION")
        print("=" * 60)
        val_data = np.load(args.outdir / "val.npz", allow_pickle=True)
        test_data = np.load(args.outdir / "test.npz", allow_pickle=True)
        X_val, y_val = val_data['X'].copy(), val_data['y'].copy()
        X_test, y_test = test_data['X'].copy(), test_data['y'].copy()
        val_data.close()
        test_data.close()
        
        X_train, X_val, X_test, pca_model = apply_pca(X_train, X_val, X_test, variance_ratio=args.pca)
        
        # Save PCA-transformed data
        np.savez_compressed(args.outdir / "train.npz", X=X_train, y=y_train)
        np.savez_compressed(args.outdir / "val.npz", X=X_val, y=y_val)
        np.savez_compressed(args.outdir / "test.npz", X=X_test, y=y_test)
        joblib.dump(pca_model, args.outdir / "pca_model.joblib")
    else:
        # Save SMOTE-transformed data (if applied)
        np.savez_compressed(args.outdir / "train.npz", X=X_train, y=y_train)
        # Remove temporary raw file
        (args.outdir / "train_raw.npz").unlink(missing_ok=True)
    
    # Compute and save class weights if requested
    if args.class_weights and len(np.unique(y_train)) > 1:
        from sklearn.utils.class_weight import compute_class_weight
        print("\n" + "=" * 60)
        print("COMPUTING CLASS WEIGHTS")
        print("=" * 60)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
        print("Class weights:")
        for cls, weight in class_weights_dict.items():
            print(f"  Class {cls}: {weight:.4f}")
        joblib.dump(class_weights_dict, args.outdir / "class_weights.joblib")

    # Save preprocessor
    joblib.dump(pre, args.outdir / "preprocessor.joblib")

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"✓ Saved artifacts to {args.outdir}")
    print(f"✓ Numeric columns: {len(pre.named_transformers_['num'].named_steps['imputer'].feature_names_in_)}")
    print(f"✓ Categorical columns: {len(getattr(pre, 'cat_cols_', []))}")
    if args.smote:
        print(f"✓ SMOTE applied to training set")
    if pca_model is not None:
        print(f"✓ PCA applied with {pca_model.n_components_} components")
    if args.class_weights:
        print(f"✓ Class weights saved")
    
    print(f"\nFinal dataset shapes:")
    print(f"  Train: {train_stats} → {X_train.shape}")
    if X_val is not None:
        print(f"  Val: {val_stats['samples']} → {X_val.shape}")
    else:
        print(f"  Val: {val_stats['samples']} → not transformed")
    if X_test is not None:
        print(f"  Test: {test_stats['samples']} → {X_test.shape}")
    else:
        print(f"  Test: {test_stats['samples']} → not transformed")


if __name__ == "__main__":
    main()
