from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.data_loader import load_heart_dataset
from data.preprocessing import (
    build_structured_preprocessor,
    fit_transform_with_selection_and_smote,
    get_selected_feature_names,
    infer_feature_types,
    transform_with_preprocessor_and_selector,
)
from models.common import (
    classification_metrics,
    confusion_matrix_payload,
    get_safe_n_splits,
    save_artifact,
    summarize_metric_folds,
)


def train_heart_model(
    data_path: str | Path = "heart.csv",
    output_path: str | Path = "artifacts/models/heart_model.joblib",
    n_estimators: int = 280,
    cv_max_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, float]:
    df = load_heart_dataset(data_path).drop_duplicates().reset_index(drop=True)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    num_features, cat_features = infer_feature_types(X.columns.tolist(), X)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    cv = StratifiedKFold(
        n_splits=get_safe_n_splits(y_trainval, max_splits=cv_max_splits),
        shuffle=True,
        random_state=random_state,
    )
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in cv.split(X_trainval, y_trainval):
        X_train = X_trainval.iloc[train_idx]
        X_val = X_trainval.iloc[val_idx]
        y_train = y_trainval.iloc[train_idx]
        y_val = y_trainval.iloc[val_idx]

        preprocessor = build_structured_preprocessor(
            numeric_features=num_features,
            categorical_features=cat_features,
            scaler="standard",
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
            max_depth=10,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        )
        model.fit(X_resampled, y_resampled)

        X_val_processed = transform_with_preprocessor_and_selector(preprocessor, selector, X_val)
        y_pred = model.predict(X_val_processed)
        y_proba = model.predict_proba(X_val_processed)
        fold_metrics.append(classification_metrics(y_val, y_pred, y_proba))

    cv_summary = summarize_metric_folds(fold_metrics)

    preprocessor = build_structured_preprocessor(
        numeric_features=num_features,
        categorical_features=cat_features,
        scaler="standard",
    )
    X_resampled, y_resampled, selector = fit_transform_with_selection_and_smote(
        preprocessor=preprocessor,
        X_train=X_trainval,
        y_train=y_trainval,
        variance_threshold=0.0,
        random_state=random_state,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    model.fit(X_resampled, y_resampled)
    X_test_processed = transform_with_preprocessor_and_selector(preprocessor, selector, X_test)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics.update({f"cv_{key}": value for key, value in cv_summary.items()})
    metrics["train_size"] = float(len(X_trainval))
    metrics["test_size"] = float(len(X_test))
    metrics["evaluation_protocol"] = "stratified_train_test_split_plus_cv"
    metrics["confusion_matrix"] = confusion_matrix_payload(y_test, y_pred, labels=model.classes_)

    artifact = {
        "model_name": "random_forest",
        "modality": "physiological_heart",
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "feature_names": get_selected_feature_names(preprocessor, selector),
        "input_features": X.columns.tolist(),
        "target_label": "heart_disease",
        "classes": model.classes_.tolist(),
    }
    save_artifact(artifact, output_path)
    return metrics

