from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from data.data_loader import load_symptom_dataset, load_symptom_severity
from data.feature_engineering import build_symptom_documents
from data.preprocessing import build_symptom_vectorizer
from models.common import save_artifact, summarize_metric_folds


class SymptomDocumentTransformer(BaseEstimator, TransformerMixin):
    """
    Convert structured symptom columns into text documents.

    Frequency metadata is learned only on training folds via fit/fit_transform.
    During transform (validation/test/inference), no augmentation is applied.
    """

    def __init__(
        self,
        symptom_columns: list[str],
        severity_map: dict[str, int] | None = None,
        all_possible_symptoms: list[str] | None = None,
        realistic_mode: bool = True,
        simulate_noise: bool = False,
        symptom_dropout_rate: float = 0.0,
        noise_injection_rate: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.symptom_columns = symptom_columns
        self.severity_map = severity_map
        self.all_possible_symptoms = all_possible_symptoms or []
        self.realistic_mode = realistic_mode
        self.simulate_noise = simulate_noise
        self.symptom_dropout_rate = symptom_dropout_rate
        self.noise_injection_rate = noise_injection_rate
        self.random_state = random_state

    def fit(self, X, y=None):
        symptom_lists = self._extract_symptom_lists(X)
        _, self.frequency_meta_ = self._build_documents_from_symptom_lists(
            symptom_lists=symptom_lists,
            fit_frequency=True,
        )
        return self

    def transform(self, X):
        symptom_lists = self._extract_symptom_lists(X)
        documents, _ = self._build_documents_from_symptom_lists(
            symptom_lists=symptom_lists,
            frequency_meta=self.frequency_meta_,
            fit_frequency=False,
        )
        return documents

    def fit_transform(self, X, y=None, **fit_params):
        symptom_lists = self._extract_symptom_lists(X)

        # Training-only augmentation:
        # 1) dropout simulates missing symptom reporting
        # 2) noise injection simulates spurious/confounding reported symptoms
        # This helps generalization on less-deterministic real-world inputs.
        if self.realistic_mode and self.simulate_noise:
            symptom_lists = self._apply_training_noise(symptom_lists)

        documents, self.frequency_meta_ = self._build_documents_from_symptom_lists(
            symptom_lists=symptom_lists,
            fit_frequency=True,
        )
        return documents

    def _extract_symptom_lists(self, X) -> list[list[str]]:
        symptom_lists: list[list[str]] = []
        for _, row in X[self.symptom_columns].iterrows():
            seen: set[str] = set()
            tokens: list[str] = []
            for symptom in row.tolist():
                normalized = str(symptom).strip().lower()
                if not normalized or normalized == "nan" or normalized in seen:
                    continue
                tokens.append(normalized)
                seen.add(normalized)
            symptom_lists.append(tokens)
        return symptom_lists

    def _build_documents_from_symptom_lists(
        self,
        symptom_lists: list[list[str]],
        frequency_meta: dict | None = None,
        fit_frequency: bool = False,
    ) -> tuple[list[str], dict]:
        symptom_frame = {
            column: [tokens[idx] if idx < len(tokens) else "" for tokens in symptom_lists]
            for idx, column in enumerate(self.symptom_columns)
        }
        return build_symptom_documents(
            symptom_frame=pd.DataFrame(symptom_frame),
            symptom_columns=self.symptom_columns,
            severity_map=self.severity_map,
            frequency_meta=frequency_meta,
            fit_frequency=fit_frequency,
        )

    def _apply_training_noise(self, symptom_lists: list[list[str]]) -> list[list[str]]:
        rng = np.random.default_rng(self.random_state)
        noised_symptom_lists: list[list[str]] = []

        for symptoms in symptom_lists:
            if not symptoms:
                noised_symptom_lists.append(symptoms)
                continue

            unique_symptoms = list(dict.fromkeys(symptoms))

            if self.symptom_dropout_rate > 0:
                keep_mask = rng.random(len(unique_symptoms)) > self.symptom_dropout_rate
                kept_symptoms = [token for token, keep in zip(unique_symptoms, keep_mask) if keep]
            else:
                kept_symptoms = unique_symptoms.copy()

            # Guarantee at least one signal remains after dropout.
            if not kept_symptoms:
                kept_symptoms = [unique_symptoms[int(rng.integers(0, len(unique_symptoms)))]]

            available_noise = [symptom for symptom in self.all_possible_symptoms if symptom not in kept_symptoms]
            if self.noise_injection_rate > 0 and available_noise:
                max_noise = max(1, len(kept_symptoms))
                n_noise = int(rng.binomial(max_noise, self.noise_injection_rate))
                n_noise = min(n_noise, len(available_noise))
                if n_noise > 0:
                    injected = rng.choice(available_noise, size=n_noise, replace=False).tolist()
                    kept_symptoms.extend(injected)

            noised_symptom_lists.append(list(dict.fromkeys(kept_symptoms)))

        return noised_symptom_lists


def _deduplicate_symptom_rows(df: pd.DataFrame, symptom_cols: list[str]) -> pd.DataFrame:
    canonical_groups = df[symptom_cols].apply(
        lambda row: "|".join(sorted(token for token in row.tolist() if token and token != "nan")),
        axis=1,
    )
    return df.assign(symptom_group=canonical_groups).drop_duplicates(subset=["symptom_group"]).reset_index(drop=True)


def _simulate_multilabel_targets(
    labels: pd.Series,
    extra_label_probability: float,
    random_state: int,
) -> list[list[str]]:
    """
    Convert single disease labels to multi-label sets by adding one additional
    disease label with configured probability, simulating comorbidity/noisy
    labeling while preventing duplicate labels.
    """
    rng = np.random.default_rng(random_state)
    base_classes = sorted(labels.astype(str).unique().tolist())
    multilabel_targets: list[list[str]] = []

    for label in labels.astype(str).tolist():
        current = [label]
        if rng.random() < extra_label_probability and len(base_classes) > 1:
            candidates = [cls for cls in base_classes if cls != label]
            if candidates:
                current.append(candidates[int(rng.integers(0, len(candidates)))])
        multilabel_targets.append(list(dict.fromkeys(current)))

    return multilabel_targets


def _build_pipeline(
    symptom_cols: list[str],
    severity_map: dict[str, int],
    all_possible_symptoms: list[str],
    random_state: int,
    realistic_mode: bool,
    simulate_noise: bool,
    symptom_dropout_rate: float,
    noise_injection_rate: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "document_builder",
                SymptomDocumentTransformer(
                    symptom_columns=symptom_cols,
                    severity_map=severity_map,
                    all_possible_symptoms=all_possible_symptoms,
                    realistic_mode=realistic_mode,
                    simulate_noise=simulate_noise,
                    symptom_dropout_rate=symptom_dropout_rate,
                    noise_injection_rate=noise_injection_rate,
                    random_state=random_state,
                ),
            ),
            ("tfidf", build_symptom_vectorizer()),
            ("model", OneVsRestClassifier(MultinomialNB(alpha=0.15))),
        ]
    )


def _threshold_predictions(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    y_pred = (y_proba >= threshold).astype(int)

    # Ensure each sample has at least one predicted label.
    for i in range(y_pred.shape[0]):
        if y_pred[i].sum() == 0:
            max_idx = int(np.argmax(y_proba[i]))
            y_pred[i, max_idx] = 1
    return y_pred


def _multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc_micro"] = float(roc_auc_score(y_true, y_proba, average="micro"))
    except Exception:
        metrics["roc_auc_micro"] = float("nan")
    metrics["roc_auc"] = metrics["roc_auc_micro"]
    return metrics


def _label_signatures(y_label_sets: list[list[str]]) -> np.ndarray:
    return np.asarray(["|".join(sorted(labels)) for labels in y_label_sets], dtype=object)


def _primary_labels(y_label_sets: list[list[str]]) -> np.ndarray:
    return np.asarray([labels[0] if labels else "__unknown__" for labels in y_label_sets], dtype=object)


def _build_cv_splitter(
    y_label_sets: list[list[str]],
    random_state: int,
    max_splits: int = 5,
) -> tuple[object, np.ndarray | None, str]:
    signatures = _label_signatures(y_label_sets)
    unique, counts = np.unique(signatures, return_counts=True)

    if len(unique) > 1 and counts.min() >= 2:
        n_splits = max(2, min(max_splits, int(counts.min())))
        return (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            signatures,
            "stratified_signature_kfold",
        )

    primary = _primary_labels(y_label_sets)
    primary_unique, primary_counts = np.unique(primary, return_counts=True)
    if len(primary_unique) > 1 and primary_counts.min() >= 2:
        n_splits = max(2, min(max_splits, int(primary_counts.min())))
        return (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            primary,
            "stratified_primary_label_kfold",
        )

    n_splits = max(2, min(max_splits, len(y_label_sets)))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None, "kfold_fallback"


def train_symptom_model(
    data_path: str | Path = "dataset.csv",
    severity_path: str | Path = "Symptom-severity.csv",
    output_path: str | Path = "artifacts/models/symptom_model.joblib",
    random_state: int = 42,
    realistic_mode: bool = True,
    simulate_noise: bool = True,
    symptom_dropout_rate: float = 0.10,
    noise_injection_rate: float = 0.08,
    extra_label_probability: float = 0.25,
    prediction_threshold: float = 0.35,
) -> dict[str, float]:
    df, symptom_cols = load_symptom_dataset(data_path)
    severity_map = load_symptom_severity(severity_path)
    df = _deduplicate_symptom_rows(df, symptom_cols)

    all_possible_symptoms = sorted(
        {
            symptom
            for column in symptom_cols
            for symptom in df[column].tolist()
            if symptom and symptom != "nan"
        }
    )

    X = df[symptom_cols]
    y_single = df["Disease"].astype(str)
    y_label_sets = _simulate_multilabel_targets(
        labels=y_single,
        extra_label_probability=extra_label_probability,
        random_state=random_state,
    )
    signatures = _label_signatures(y_label_sets)
    primary_labels = _primary_labels(y_label_sets)
    signature_min_count = pd.Series(signatures).value_counts().min()
    primary_min_count = pd.Series(primary_labels).value_counts().min()
    if signature_min_count >= 2:
        stratify_for_split = signatures
    elif primary_min_count >= 2:
        stratify_for_split = primary_labels
    else:
        stratify_for_split = None

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_for_split,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train_sets = [y_label_sets[i] for i in train_idx]

    mlb = MultiLabelBinarizer()
    y_all_bin = mlb.fit_transform(y_label_sets)
    y_train = y_all_bin[train_idx]
    y_test = y_all_bin[test_idx]

    label_counter = Counter(label for labels in y_label_sets for label in labels)
    class_distribution = dict(sorted(label_counter.items(), key=lambda x: x[0]))
    imbalance_ratio = float(max(label_counter.values()) / max(1, min(label_counter.values())))
    label_density = float(np.mean(np.sum(y_all_bin, axis=1)))

    cv_splitter, cv_signatures, cv_strategy = _build_cv_splitter(
        y_label_sets=y_train_sets,
        random_state=random_state,
        max_splits=5,
    )
    fold_metrics: list[dict[str, float]] = []

    if cv_signatures is None:
        split_iter = cv_splitter.split(X_train)
    else:
        split_iter = cv_splitter.split(X_train, cv_signatures)

    for fold_train_idx, fold_val_idx in split_iter:
        X_fold_train = X_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_train = y_train[fold_train_idx]
        y_fold_val = y_train[fold_val_idx]

        pipeline = _build_pipeline(
            symptom_cols=symptom_cols,
            severity_map=severity_map,
            all_possible_symptoms=all_possible_symptoms,
            random_state=random_state,
            realistic_mode=realistic_mode,
            simulate_noise=simulate_noise,
            symptom_dropout_rate=symptom_dropout_rate,
            noise_injection_rate=noise_injection_rate,
        )
        pipeline.fit(X_fold_train, y_fold_train)

        y_fold_proba = np.asarray(pipeline.predict_proba(X_fold_val))
        y_fold_pred = _threshold_predictions(y_fold_proba, threshold=prediction_threshold)
        fold_metrics.append(_multilabel_metrics(y_fold_val, y_fold_pred, y_fold_proba))

    cv_summary = summarize_metric_folds(fold_metrics)

    final_pipeline = _build_pipeline(
        symptom_cols=symptom_cols,
        severity_map=severity_map,
        all_possible_symptoms=all_possible_symptoms,
        random_state=random_state,
        realistic_mode=realistic_mode,
        simulate_noise=simulate_noise,
        symptom_dropout_rate=symptom_dropout_rate,
        noise_injection_rate=noise_injection_rate,
    )
    final_pipeline.fit(X_train, y_train)

    y_proba = np.asarray(final_pipeline.predict_proba(X_test))
    y_pred = _threshold_predictions(y_proba, threshold=prediction_threshold)
    test_metrics = _multilabel_metrics(y_test, y_pred, y_proba)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = classification_report(
            y_test,
            y_pred,
            target_names=mlb.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )

    metrics = {
        **test_metrics,
        "cv_subset_accuracy": cv_summary.get("subset_accuracy", float("nan")),
        "cv_subset_accuracy_std": cv_summary.get("subset_accuracy_std", float("nan")),
        "cv_f1_micro": cv_summary.get("f1_micro", float("nan")),
        "cv_f1_micro_std": cv_summary.get("f1_micro_std", float("nan")),
        "cv_f1_macro": cv_summary.get("f1_macro", float("nan")),
        "cv_f1_macro_std": cv_summary.get("f1_macro_std", float("nan")),
        "cv_jaccard_micro": cv_summary.get("jaccard_micro", float("nan")),
        "cv_jaccard_micro_std": cv_summary.get("jaccard_micro_std", float("nan")),
        "cv_hamming_loss": cv_summary.get("hamming_loss", float("nan")),
        "cv_hamming_loss_std": cv_summary.get("hamming_loss_std", float("nan")),
        "cv_roc_auc_micro": cv_summary.get("roc_auc_micro", float("nan")),
        "cv_roc_auc_micro_std": cv_summary.get("roc_auc_micro_std", float("nan")),
        "cv_accuracy": cv_summary.get("accuracy", float("nan")),
        "cv_accuracy_std": cv_summary.get("accuracy_std", float("nan")),
        "cv_precision_weighted": cv_summary.get("precision_weighted", float("nan")),
        "cv_precision_weighted_std": cv_summary.get("precision_weighted_std", float("nan")),
        "cv_recall_weighted": cv_summary.get("recall_weighted", float("nan")),
        "cv_recall_weighted_std": cv_summary.get("recall_weighted_std", float("nan")),
        "cv_f1_weighted": cv_summary.get("f1_weighted", float("nan")),
        "cv_f1_weighted_std": cv_summary.get("f1_weighted_std", float("nan")),
        "cv_n_folds": cv_summary.get("n_folds", float("nan")),
        "cv_strategy": cv_strategy,
        "train_size": float(len(X_train)),
        "test_size": float(len(X_test)),
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "label_density": label_density,
        "multilabel_enabled": True,
        "extra_label_probability": extra_label_probability,
        "prediction_threshold": prediction_threshold,
        "smote_applied": False,
        "smote_reason": "Not applied: TF-IDF + MultinomialNB pipeline with sparse text features.",
        "realistic_mode": realistic_mode,
        "noise_simulation_enabled": simulate_noise,
        "symptom_dropout_rate": symptom_dropout_rate,
        "noise_injection_rate": noise_injection_rate,
        "evaluation_protocol": "multilabel_train_test_split_plus_cv_pipeline",
        "classification_report": report,
    }

    artifact = {
        "model_name": "one_vs_rest_multinomial_naive_bayes",
        "modality": "symptom_text",
        "multilabel": True,
        "pipeline": final_pipeline,
        "model": final_pipeline.named_steps["model"],
        "vectorizer": final_pipeline.named_steps["tfidf"],
        "document_builder": final_pipeline.named_steps["document_builder"],
        "symptom_columns": symptom_cols,
        "severity_map": severity_map,
        "all_possible_symptoms": all_possible_symptoms,
        "frequency_meta": final_pipeline.named_steps["document_builder"].frequency_meta_,
        "mlb": mlb,
        "classes": mlb.classes_.tolist(),
        "prediction_threshold": prediction_threshold,
        "extra_label_probability": extra_label_probability,
    }
    save_artifact(artifact, output_path)
    return metrics
