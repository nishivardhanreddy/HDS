from __future__ import annotations

import json
import os
from pathlib import Path

from models.diabetes_model import train_diabetes_model
from models.heart_model import train_heart_model
from models.lifestyle_risk_model import train_lifestyle_risk_model
from models.symptom_model import train_symptom_model


def _format_metric_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:18s} | "
        f"acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision_weighted']:.4f} "
        f"rec={metrics['recall_weighted']:.4f} "
        f"f1={metrics['f1_weighted']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


def main() -> None:
    metrics: dict[str, dict[str, float]] = {}
    lifestyle_rows = int(os.getenv("LIFESTYLE_MAX_ROWS", "80000"))
    lifestyle_trees = int(os.getenv("LIFESTYLE_TREES", "120"))

    metrics["symptom_model"] = train_symptom_model()
    metrics["diabetes_model"] = train_diabetes_model()
    metrics["heart_model"] = train_heart_model()
    metrics["lifestyle_risk_model"] = train_lifestyle_risk_model(
        max_rows_per_file=lifestyle_rows,
        n_estimators=lifestyle_trees,
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
