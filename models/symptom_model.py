from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from data.data_loader import load_symptom_dataset, load_symptom_severity
from data.feature_engineering import build_symptom_documents
from data.preprocessing import build_symptom_vectorizer
from models.common import classification_metrics, save_artifact


def train_symptom_model(
    data_path: str | Path = "dataset.csv",
    severity_path: str | Path = "Symptom-severity.csv",
    output_path: str | Path = "artifacts/models/symptom_model.joblib",
    random_state: int = 42,
) -> dict[str, float]:
    df, symptom_cols = load_symptom_dataset(data_path)
    severity_map = load_symptom_severity(severity_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df[symptom_cols],
        df["Disease"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["Disease"],
    )

    train_docs, freq_meta = build_symptom_documents(
        X_train,
        symptom_columns=symptom_cols,
        severity_map=severity_map,
        fit_frequency=True,
    )
    test_docs, _ = build_symptom_documents(
        X_test,
        symptom_columns=symptom_cols,
        severity_map=severity_map,
        frequency_meta=freq_meta,
        fit_frequency=False,
    )

    vectorizer = build_symptom_vectorizer()
    X_train_vec = vectorizer.fit_transform(train_docs)
    X_test_vec = vectorizer.transform(test_docs)

    model = MultinomialNB(alpha=0.15)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)
    metrics = classification_metrics(y_test, y_pred, y_proba)

    artifact = {
        "model_name": "multinomial_naive_bayes",
        "modality": "symptom_text",
        "model": model,
        "vectorizer": vectorizer,
        "symptom_columns": symptom_cols,
        "severity_map": severity_map,
        "frequency_meta": freq_meta,
        "classes": model.classes_.tolist(),
    }
    save_artifact(artifact, output_path)
    return metrics

