from __future__ import annotations

import json
import os
from pathlib import Path

from models.diabetes_model import train_diabetes_model
from models.heart_model import train_heart_model
from models.lifestyle_risk_model import train_lifestyle_risk_model
from models.symptom_model import train_symptom_model


def _run_with_progress(label: str, trainer, **kwargs) -> dict[str, float]:
    print(f"[START] Training {label}...", flush=True)
    metrics = trainer(**kwargs)
    print(_format_metric_line(label, metrics), flush=True)
    return metrics


def _format_metric_line(name: str, metrics: dict[str, float]) -> str:
    accuracy = f"{metrics['accuracy']:.4f}"
    if "accuracy_std" in metrics:
        accuracy = f"{metrics['accuracy']:.4f}+/-{metrics['accuracy_std']:.4f}"
    if "cv_accuracy" in metrics and "cv_accuracy_std" in metrics:
        accuracy = f"{accuracy} (cv={metrics['cv_accuracy']:.4f}+/-{metrics['cv_accuracy_std']:.4f})"

    return (
        f"{name:18s} | "
        f"acc={accuracy} "
        f"prec={metrics['precision_weighted']:.4f} "
        f"rec={metrics['recall_weighted']:.4f} "
        f"f1={metrics['f1_weighted']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


def main() -> None:
    metrics: dict[str, dict[str, float]] = {}
    fast_train = os.getenv("FAST_TRAIN", "0") == "1"

    lifestyle_rows_default = "8000" if fast_train else "80000"
    lifestyle_trees_default = "25" if fast_train else "120"
    lifestyle_cv_default = "2" if fast_train else "5"

    diabetes_trees_default = "140" if fast_train else "320"
    diabetes_cv_default = "3" if fast_train else "5"

    heart_trees_default = "140" if fast_train else "280"
    heart_cv_default = "3" if fast_train else "5"

    model_n_jobs_default = "1"

    lifestyle_rows = int(os.getenv("LIFESTYLE_MAX_ROWS", lifestyle_rows_default))
    lifestyle_trees = int(os.getenv("LIFESTYLE_TREES", lifestyle_trees_default))
    lifestyle_cv_splits = int(os.getenv("LIFESTYLE_CV_SPLITS", lifestyle_cv_default))
    diabetes_trees = int(os.getenv("DIABETES_TREES", diabetes_trees_default))
    diabetes_cv_splits = int(os.getenv("DIABETES_CV_SPLITS", diabetes_cv_default))
    heart_trees = int(os.getenv("HEART_TREES", heart_trees_default))
    heart_cv_splits = int(os.getenv("HEART_CV_SPLITS", heart_cv_default))
    model_n_jobs = int(os.getenv("MODEL_N_JOBS", model_n_jobs_default))

    print(
        "[CONFIG] "
        f"fast_train={fast_train} "
        f"diabetes_trees={diabetes_trees} diabetes_cv={diabetes_cv_splits} "
        f"heart_trees={heart_trees} heart_cv={heart_cv_splits} "
        f"lifestyle_rows={lifestyle_rows} lifestyle_trees={lifestyle_trees} lifestyle_cv={lifestyle_cv_splits} "
        f"model_n_jobs={model_n_jobs}",
        flush=True,
    )

    metrics["symptom_model"] = _run_with_progress("symptom_model", train_symptom_model)
    metrics["diabetes_model"] = _run_with_progress(
        "diabetes_model",
        train_diabetes_model,
        n_estimators=diabetes_trees,
        cv_splits=diabetes_cv_splits,
        n_jobs=model_n_jobs,
    )
    metrics["heart_model"] = _run_with_progress(
        "heart_model",
        train_heart_model,
        n_estimators=heart_trees,
        cv_max_splits=heart_cv_splits,
        n_jobs=model_n_jobs,
    )
    metrics["lifestyle_risk_model"] = _run_with_progress(
        "lifestyle_risk_model",
        train_lifestyle_risk_model,
        max_rows_per_file=lifestyle_rows,
        n_estimators=lifestyle_trees,
        cv_splits=lifestyle_cv_splits,
        n_jobs=model_n_jobs,
    )

    output_path = Path("artifacts/metrics/model_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    print("\nModel Evaluation Metrics")
    print("-" * 95)
    for name, result in metrics.items():
        print(_format_metric_line(name, result))
    print("-" * 95)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
