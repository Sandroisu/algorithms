#!/usr/bin/env python3
"""
ML engineering mini-experiment:
- dataset: sklearn breast_cancer
- preprocessing: numeric scaling (only), later you can add categorical, imputation, etc.
- model: LogisticRegression (default) or RandomForestClassifier
- evaluation: Stratified K-Fold CV + final holdout
- artifacts: model.pkl, metrics.json, confusion_matrix.png, cv_results.csv
- logging, reproducibility, CLI via argparse

Usage examples:
  python train_experiment.py --model logreg --seed 42
  python train_experiment.py --model rf --n-estimators 300 --max-depth 6
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings

# -----------------------------
# utils
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# data
# -----------------------------
def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Берём удобный табличный датасет рака груди (классификация 2 классов).
    Возвращаем X (DataFrame), y (Series).
    """
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    X = df.drop(columns=[ds.target.name])
    y = df[ds.target.name]
    return X, y

# -----------------------------
# pipelines
# -----------------------------
def build_pipeline(model_name: str, args: argparse.Namespace, feature_names) -> Pipeline:
    """
    Строим end-to-end pipeline: препроцессинг -> модель.
    Сейчас все фичи числовые, поэтому просто StandardScaler.
    Если будут категориальные — расширишь ColumnTransformer.
    """
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), feature_names),
        ],
        remainder="drop",
        n_jobs=None,
    )

    if model_name == "logreg":
        # Линейная модель — хорошая базовая линия; регуляризация C настраивается.
        model = LogisticRegression(
            C=args.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=args.max_iter,
            n_jobs=None,
            random_state=args.seed,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
            random_state=args.seed,
        )
    else:
        raise ValueError("Unknown model: choose 'logreg' or 'rf'")

    pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])
    return pipe

# -----------------------------
# evaluation
# -----------------------------
def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int, folds: int = 5) -> Dict[str, Any]:
    """
    Стратифицированная кросс-валидация: считаем accuracy, f1, roc_auc.
    Возвращаем среднее и std по фолдам + сырые результаты в DataFrame (для сохранения).
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
    }
    result = cross_validate(
        pipe,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    # агрегаты
    agg = {
        metric: {"mean": float(np.mean(result[f"test_{metric}"])),
                 "std": float(np.std(result[f"test_{metric}"]))}
        for metric in scoring.keys()
    }
    return {"cv_metrics": agg, "raw": pd.DataFrame(result)}

def evaluate_holdout(pipe: Pipeline, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None
    preds = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }
    if proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    return metrics

def plot_confusion_matrix(y_true, y_pred, out_path: str) -> None:
    """
    Рисуем confusion matrix и сохраняем как png.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["neg", "pos"])
    plt.yticks(tick_marks, ["neg", "pos"])
    # подписи ячеек
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini ML engineering experiment runner")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg", help="which model to use")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--test-size", type=float, default=0.2, help="holdout test size")
    parser.add_argument("--cv-folds", type=int, default=5, help="number of StratifiedKFold folds")

    # logreg hyperparams
    parser.add_argument("--C", type=float, default=1.0, help="inverse regularization strength")
    parser.add_argument("--max-iter", type=int, default=200, help="max iterations for solver")

    # random forest hyperparams
    parser.add_argument("--n-estimators", type=int, default=300, help="number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="max depth of each tree")
    parser.add_argument("--min-samples-split", type=int, default=2, help="min samples to split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="min samples in leaf")

    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="where to save outputs")
    return parser.parse_args()

def main():
    warnings.filterwarnings("ignore")
    setup_logging()
    args = parse_args()

    run_id = f"{args.model}_{timestamp()}"
    out_dir = os.path.join(args.artifacts_dir, run_id)
    ensure_dir(out_dir)
    logging.info(f"Artifacts will be saved to: {out_dir}")

    # 1) Data
    X, y = load_data()
    logging.info(f"Data shape: X={X.shape}, y={y.shape}")

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 3) Pipeline
    pipe = build_pipeline(args.model, args, feature_names=X.columns)

    # 4) CV evaluation
    t0 = time.time()
    cv_res = evaluate_cv(pipe, X_train, y_train, seed=args.seed, folds=args.cv_folds)
    cv_time = time.time() - t0
    logging.info(f"CV done in {cv_time:.2f}s | accuracy={cv_res['cv_metrics']['accuracy']['mean']:.4f} "
                 f"+/- {cv_res['cv_metrics']['accuracy']['std']:.4f}")

    # 5) Final holdout
    hold_metrics = evaluate_holdout(pipe, X_train, X_test, y_train, y_test)
    logging.info(f"Holdout metrics: {hold_metrics}")

    # 6) Save artifacts
    # 6.1 model
    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(pipe, model_path)
    # 6.2 metrics
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "cv_metrics": cv_res["cv_metrics"],
        "holdout_metrics": hold_metrics,
        "cv_time_sec": cv_time,
    }
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    # 6.3 cv raw table
    cv_res["raw"].to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    # 6.4 confusion matrix on holdout
    y_pred_hold = pipe.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_hold, os.path.join(out_dir, "confusion_matrix.png"))

    # 7) short summary to console
    logging.info(f"Saved: {model_path}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
