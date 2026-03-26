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
    predicted_label: str | list[str],
    top_n: int = 6,
) -> list[dict]:
    pipeline = artifact.get("pipeline")
    model = artifact["model"]
    symptom_cols = artifact["symptom_columns"]

    row_data = {col: symptom_tokens[idx] if idx < len(symptom_tokens) else "" for idx, col in enumerate(symptom_cols)}
    symptom_frame = pd.DataFrame([row_data])

    if pipeline is not None:
        document_builder = pipeline.named_steps["document_builder"]
        vectorizer = pipeline.named_steps["tfidf"]
        docs = document_builder.transform(symptom_frame)
        vec = vectorizer.transform(docs)
    else:
        vectorizer = artifact["vectorizer"]
        severity_map = artifact["severity_map"]
        frequency_meta = artifact["frequency_meta"]
        docs, _ = build_symptom_documents(
            symptom_frame=symptom_frame,
            symptom_columns=symptom_cols,
            severity_map=severity_map,
            frequency_meta=frequency_meta,
            fit_frequency=False,
        )
        vec = vectorizer.transform(docs)

    token_idx = vec.indices
    token_values = vec.data
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "estimators_") and artifact.get("multilabel"):
        target_labels = predicted_label if isinstance(predicted_label, list) else [predicted_label]
        classes = [str(label) for label in artifact.get("classes", [])]
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        target_indices = [class_to_idx[label] for label in target_labels if label in class_to_idx]

        if not target_indices:
            return []

        contribution_map: dict[str, float] = {}
        for class_idx in target_indices:
            estimator = model.estimators_[class_idx]
            if not hasattr(estimator, "feature_log_prob_"):
                continue

            class_log_prob = estimator.feature_log_prob_
            if class_log_prob.ndim == 2:
                positive_idx = int(np.where(estimator.classes_ == 1)[0][0]) if 1 in estimator.classes_ else -1
                class_weights = class_log_prob[positive_idx]
            else:
                class_weights = class_log_prob

            for idx, value in zip(token_idx, token_values):
                term = feature_names[idx]
                if term.startswith("freqbin_") or "_freq_" in term or term.startswith("severity_"):
                    continue
                contribution_map[term] = contribution_map.get(term, 0.0) + float(value * class_weights[idx])

        sorted_terms = sorted(contribution_map.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": term.replace("_", " "), "score": score} for term, score in sorted_terms[:top_n]]

    class_idx = int(np.where(model.classes_ == predicted_label)[0][0])
    class_log_prob = model.feature_log_prob_[class_idx]

    contributions = []
    for idx, value in zip(token_idx, token_values):
        term = feature_names[idx]
        if term.startswith("freqbin_") or "_freq_" in term or term.startswith("severity_"):
            continue
        contributions.append((term, float(value * class_log_prob[idx])))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": term.replace("_", " "), "score": score} for term, score in contributions[:top_n]]
