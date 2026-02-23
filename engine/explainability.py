from __future__ import annotations

import numpy as np
import pandas as pd

from data.feature_engineering import build_symptom_documents


def _clean_feature_label(label: str) -> str:
    label = label.replace("num__", "").replace("cat__", "")
    label = label.replace("_", " ")
    return label


def explain_tree_model_prediction(artifact: dict, input_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    selector = artifact["selector"]
    feature_names = artifact["feature_names"]

    transformed = selector.transform(preprocessor.transform(input_df))
    if hasattr(transformed, "toarray"):
        row = transformed.toarray()[0]
    else:
        row = np.asarray(transformed)[0]

    importances = np.asarray(model.feature_importances_)
    contribution = np.abs(row) * importances
    top_idx = np.argsort(contribution)[::-1][:top_n]

    explanations = []
    for idx in top_idx:
        if contribution[idx] <= 0:
            continue
        explanations.append(
            {
                "feature": _clean_feature_label(str(feature_names[idx])),
                "score": float(contribution[idx]),
            }
        )
    return explanations


def explain_symptom_prediction(
    artifact: dict,
    symptom_tokens: list[str],
    predicted_label: str,
    top_n: int = 6,
) -> list[dict]:
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]
    severity_map = artifact["severity_map"]
    symptom_cols = artifact["symptom_columns"]
    frequency_meta = artifact["frequency_meta"]

    row_data = {col: symptom_tokens[idx] if idx < len(symptom_tokens) else "" for idx, col in enumerate(symptom_cols)}
    symptom_frame = pd.DataFrame([row_data])
    docs, _ = build_symptom_documents(
        symptom_frame=symptom_frame,
        symptom_columns=symptom_cols,
        severity_map=severity_map,
        frequency_meta=frequency_meta,
        fit_frequency=False,
    )
    vec = vectorizer.transform(docs)
    class_idx = int(np.where(model.classes_ == predicted_label)[0][0])

    token_idx = vec.indices
    token_values = vec.data
    feature_names = vectorizer.get_feature_names_out()
    class_log_prob = model.feature_log_prob_[class_idx]

    contributions = []
    for idx, value in zip(token_idx, token_values):
        term = feature_names[idx]
        if term.startswith("freqbin_") or "_freq_" in term or term.startswith("severity_"):
            continue
        contributions.append((term, float(value * class_log_prob[idx])))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": term.replace("_", " "), "score": score} for term, score in contributions[:top_n]]

