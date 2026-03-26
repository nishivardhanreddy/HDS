from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model_path(filename: str) -> str:
    """
    Return the absolute path to a model artifact under artifacts/models.

    Path resolution is anchored to this file, so it remains stable regardless
    of the current working directory in local runs, Docker, or deployment.
    """
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    return os.path.abspath(os.path.join(project_root, "artifacts", "models", filename))


def save_artifact(artifact: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        if y_proba is None:
            metrics["roc_auc"] = float("nan")
            return metrics

        try:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


def summarize_metric_folds(metric_folds: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not metric_folds:
        return summary

    metric_names = metric_folds[0].keys()
    for metric_name in metric_names:
        values = np.asarray([fold[metric_name] for fold in metric_folds], dtype=float)
        summary[metric_name] = float(np.nanmean(values))
        summary[f"{metric_name}_std"] = float(np.nanstd(values))
    summary["n_folds"] = float(len(metric_folds))
    return summary


def get_safe_n_splits(target, max_splits: int = 5) -> int:
    values, counts = np.unique(np.asarray(target), return_counts=True)
    if len(counts) == 0:
        return 2
    min_class_count = int(np.min(counts))
    return max(2, min(max_splits, min_class_count))


def confusion_matrix_payload(y_true, y_pred, labels=None) -> dict[str, Any]:
    labels_array = np.asarray(labels if labels is not None else np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        matrix = confusion_matrix(y_true, y_pred, labels=labels_array)
    return {
        "labels": labels_array.tolist(),
        "matrix": matrix.tolist(),
    }
