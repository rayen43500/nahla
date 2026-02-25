import argparse
import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def transform_and_save(pre: ColumnTransformer, df: pd.DataFrame, label_col: str, out_path: pathlib.Path) -> None:
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    X_t = pre.transform(X)
    np.savez_compressed(out_path, X=X_t, y=y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset: split, encode, scale")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Input CSV file")
    parser.add_argument("--label", type=str, default="label", help="Label column name")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"), help="Output directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    if args.label not in df.columns:
        raise ValueError(f"Label column {args.label} not found")

    train_df, val_df, test_df = split_data(df, label_col=args.label, test_size=args.test_size, val_size=args.val_size)
    pre = build_preprocessor(train_df, label_col=args.label)
    pre.fit(train_df.drop(columns=[args.label]))

    joblib.dump(pre, args.outdir / "preprocessor.joblib")
    transform_and_save(pre, train_df, args.label, args.outdir / "train.npz")
    transform_and_save(pre, val_df, args.label, args.outdir / "val.npz")
    transform_and_save(pre, test_df, args.label, args.outdir / "test.npz")

    print("Saved artifacts to", args.outdir)
    print("Numeric cols:", getattr(pre, "num_cols_", []))
    print("Categorical cols:", getattr(pre, "cat_cols_", []))


if __name__ == "__main__":
    main()
