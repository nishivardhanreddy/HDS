from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data.data_loader import load_diabetes_dataset
from data.preprocessing import (
    build_structured_preprocessor,
    fit_transform_with_selection_and_smote,
    get_selected_feature_names,
    infer_feature_types,
    transform_with_preprocessor_and_selector,
)
from models.common import classification_metrics, save_artifact


def train_diabetes_model(
    data_path: str | Path = "diabetes_prediction_dataset.csv",
    output_path: str | Path = "artifacts/models/diabetes_model.joblib",
    random_state: int = 42,
) -> dict[str, float]:
    df = load_diabetes_dataset(data_path)
    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    num_features, cat_features = infer_feature_types(X.columns.tolist(), X)
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
        n_estimators=320,
        max_depth=None,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(X_resampled, y_resampled)

    X_test_processed = transform_with_preprocessor_and_selector(preprocessor, selector, X_test)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)
    metrics = classification_metrics(y_test, y_pred, y_proba)

    artifact = {
        "model_name": "random_forest",
        "modality": "clinical_structured_diabetes",
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "feature_names": get_selected_feature_names(preprocessor, selector),
        "input_features": X.columns.tolist(),
        "target_label": "diabetes",
        "classes": model.classes_.tolist(),
    }
    save_artifact(artifact, output_path)
    return metrics

