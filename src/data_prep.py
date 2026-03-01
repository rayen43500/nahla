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
    cat_cols = [c for c in feature_cols if df[c].dtype == object or str(df[c].dtype).startswith("category")]
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


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE for class imbalance handling."""
    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


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
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=relative_val, stratify=y_tmp, random_state=seed)
    train = X_train.copy(); train[label_col] = y_train
    val = X_val.copy(); val[label_col] = y_val
    test = X_test.copy(); test[label_col] = y_test
    return train, val, test


def build_preprocessor(df: pd.DataFrame, label_col: str) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != label_col]
    cat_cols = [c for c in feature_cols if df[c].dtype == object or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    preprocessor.cat_cols_ = cat_cols  # type: ignore[attr-defined]
    preprocessor.num_cols_ = num_cols  # type: ignore[attr-defined]
    return preprocessor


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
    parser.add_argument("--robust-scaler", action="store_false", default=True, help="Use RobustScaler (default: True, set to disable)")
    parser.add_argument("--pca", type=float, default=None, help="Apply PCA with variance ratio (e.g., 0.95)")
    parser.add_argument("--class-weights", action="store_true", help="Compute class weights for imbalance handling")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    if args.label not in df.columns:
        raise ValueError(f"Label column {args.label} not found")

    # Detect and report missing values
    print("=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    missing_stats = detect_missing_values(df, args.label)
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
    class_analysis = analyze_class_distribution(df[args.label])
    print(f"Number of classes: {class_analysis['num_classes']}")
    print(f"Imbalance ratio: {class_analysis['imbalance_ratio']:.3f}x")
    print("Class distribution:")
    for cls, count in class_analysis['class_counts'].items():
        dist_pct = class_analysis['class_distribution_pct'][cls]
        print(f"  Class {cls}: {count} samples ({dist_pct:.2f}%)")
    
    # Split data
    train_df, val_df, test_df = split_data(df, label_col=args.label, test_size=args.test_size, val_size=args.val_size)
    pre = build_preprocessor(train_df, label_col=args.label, use_robust_scaler=args.robust_scaler)
    pre.fit(train_df.drop(columns=[args.label]))

    # Transform data
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION")
    print("=" * 60)
    train_stats = transform_and_save(pre, train_df, args.label, args.outdir / "train_raw.npz")
    val_stats = transform_and_save(pre, val_df, args.label, args.outdir / "val.npz")
    test_stats = transform_and_save(pre, test_df, args.label, args.outdir / "test.npz")
    
    # Load transformed training data for SMOTE and PCA
    train_data = np.load(args.outdir / "train_raw.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    # Apply SMOTE if requested
    pca_model = None
    if args.smote:
        print("\n" + "=" * 60)
        print("APPLYING SMOTE FOR CLASS IMBALANCE")
        print("=" * 60)
        print(f"Train set before SMOTE: {len(y_train)} samples")
        X_train, y_train = apply_smote(X_train, y_train)
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
        val_data = np.load(args.outdir / "val.npz")
        test_data = np.load(args.outdir / "test.npz")
        X_val, y_val = val_data['X'], val_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        
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
