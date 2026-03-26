from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from data.data_loader import load_brfss_lifestyle_dataset
from data.feature_engineering import prepare_lifestyle_features
from data.preprocessing import (
    build_structured_preprocessor,
    fit_transform_with_selection_and_smote,
    get_selected_feature_names,
    infer_feature_types,
    transform_with_preprocessor_and_selector,
)
from models.common import classification_metrics, save_artifact, summarize_metric_folds


def train_lifestyle_risk_model(
    data_paths: Sequence[str | Path] | None = None,
    output_path: str | Path = "artifacts/models/lifestyle_risk_model.joblib",
    max_rows_per_file: int = 80_000,
    n_estimators: int = 120,
    max_depth: int | None = 10,
    cv_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, float]:
    raw_df = load_brfss_lifestyle_dataset(paths=data_paths, max_rows_per_file=max_rows_per_file)
    df = prepare_lifestyle_features(raw_df)

    target_col = "target_health_risk"
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    num_features, cat_features = infer_feature_types(X.columns.tolist(), X)
    safe_splits = max(2, min(cv_splits, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        preprocessor = build_structured_preprocessor(
            numeric_features=num_features,
            categorical_features=cat_features,
            scaler="minmax",
        )
        X_resampled, y_resampled, selector = fit_transform_with_selection_and_smote(
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            variance_threshold=0.0,
            random_state=random_state,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        )
        model.fit(X_resampled, y_resampled)

        X_test_processed = transform_with_preprocessor_and_selector(preprocessor, selector, X_test)
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)
        fold_metrics.append(classification_metrics(y_test, y_pred, y_proba))

    metrics = summarize_metric_folds(fold_metrics)

    preprocessor = build_structured_preprocessor(
        numeric_features=num_features,
        categorical_features=cat_features,
        scaler="minmax",
    )
    X_resampled, y_resampled, selector = fit_transform_with_selection_and_smote(
        preprocessor=preprocessor,
        X_train=X,
        y_train=y,
        variance_threshold=0.0,
        random_state=random_state,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=6,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    model.fit(X_resampled, y_resampled)

    artifact = {
        "model_name": "random_forest",
        "modality": "lifestyle_risk",
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "feature_names": get_selected_feature_names(preprocessor, selector),
        "input_features": X.columns.tolist(),
        "target_label": "long_term_health_risk",
        "classes": model.classes_.tolist(),
    }
    save_artifact(artifact, output_path)
    return metrics

