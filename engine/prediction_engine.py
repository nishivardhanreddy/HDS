from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from data.feature_engineering import build_symptom_documents
from engine.context_builder import build_context
from engine.explainability import explain_symptom_prediction, explain_tree_model_prediction
from engine.model_selector import select_models


class PredictionEngine:
    """Context-aware model execution with multi-modal prediction fusion."""

    def __init__(self, model_dir: str | Path = "artifacts/models"):
        self.model_dir = Path(model_dir)
        self.artifacts = self._load_artifacts()
        self.extension_models: dict[str, Callable] = {}

    def register_extension_model(self, name: str, predictor_callable) -> None:
        """
        Register a future model (e.g., neural physiological model) without
        changing the core context-aware workflow.
        """
        self.extension_models[name] = predictor_callable

    def _load_artifacts(self) -> dict[str, dict]:
        artifact_files = {
            "symptom": "symptom_model.joblib",
            "diabetes": "diabetes_model.joblib",
            "heart": "heart_model.joblib",
            "lifestyle": "lifestyle_risk_model.joblib",
        }
        loaded: dict[str, dict] = {}
        for key, filename in artifact_files.items():
            path = self.model_dir / filename
            if path.exists():
                loaded[key] = joblib.load(path)
        return loaded

    def available_models(self) -> list[str]:
        return sorted(self.artifacts.keys())

    def predict(self, payload: dict) -> dict:
        context = build_context(payload)
        requested_models = select_models(context)
        active_models = [m for m in requested_models if m in self.artifacts]

        predictions: list[dict] = []
        probs_for_fusion: list[float] = []

        for model_name in active_models:
            if model_name == "symptom":
                pred = self._predict_symptom(payload)
            elif model_name == "diabetes":
                pred = self._predict_structured_binary(payload, artifact_key="diabetes", source_key="diabetes_clinical")
            elif model_name == "heart":
                pred = self._predict_structured_binary(payload, artifact_key="heart", source_key="heart_clinical")
            else:
                pred = self._predict_structured_binary(payload, artifact_key="lifestyle", source_key="lifestyle")

            predictions.append(pred)
            if pred.get("probability") is not None:
                probs_for_fusion.append(float(pred["probability"]))

        combined_probability = float(np.mean(probs_for_fusion)) if probs_for_fusion else 0.0
        combined_level = self._probability_level(combined_probability)

        return {
            "context": context,
            "selected_models": active_models,
            "predictions": predictions,
            "combined_probability": combined_probability,
            "combined_level": combined_level,
        }

    @staticmethod
    def _probability_level(prob: float) -> str:
        if prob < 0.35:
            return "LOW"
        if prob < 0.65:
            return "MODERATE"
        return "HIGH"

    def _predict_symptom(self, payload: dict) -> dict:
        artifact = self.artifacts["symptom"]
        model = artifact["model"]
        vectorizer = artifact["vectorizer"]
        symptom_cols = artifact["symptom_columns"]
        severity_map = artifact["severity_map"]
        frequency_meta = artifact["frequency_meta"]

        symptoms = payload.get("symptoms", []) or []
        row_data = {col: symptoms[idx] if idx < len(symptoms) else "" for idx, col in enumerate(symptom_cols)}
        symptom_frame = pd.DataFrame([row_data])
        docs, _ = build_symptom_documents(
            symptom_frame=symptom_frame,
            symptom_columns=symptom_cols,
            severity_map=severity_map,
            frequency_meta=frequency_meta,
            fit_frequency=False,
        )

        x = vectorizer.transform(docs)
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        class_index = int(np.where(model.classes_ == pred)[0][0])
        confidence = float(proba[class_index])

        explanation = explain_symptom_prediction(artifact, symptoms, pred, top_n=6)
        return {
            "model": "symptom",
            "prediction": str(pred),
            "probability": confidence,
            "explanation": explanation,
        }

    def _predict_structured_binary(self, payload: dict, artifact_key: str, source_key: str) -> dict:
        artifact = self.artifacts[artifact_key]
        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        selector = artifact["selector"]
        input_features = artifact["input_features"]

        source = payload.get(source_key, {}) or {}
        history = payload.get("history", {}) or {}
        lifestyle = payload.get("lifestyle", {}) or {}

        merged = {}
        merged.update(history)
        merged.update(lifestyle)
        merged.update(source)

        row = {feature: merged.get(feature, np.nan) for feature in input_features}
        input_df = pd.DataFrame([row])

        x = selector.transform(preprocessor.transform(input_df))
        pred = int(model.predict(x)[0])
        proba = model.predict_proba(x)[0]
        confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
        explanation = explain_tree_model_prediction(artifact, input_df, top_n=5)

        positive_label = {
            "diabetes": "Elevated diabetes risk",
            "heart": "Elevated heart disease risk",
            "lifestyle": "Elevated long-term health risk",
        }[artifact_key]
        negative_label = {
            "diabetes": "Lower diabetes risk",
            "heart": "Lower heart disease risk",
            "lifestyle": "Lower long-term health risk",
        }[artifact_key]

        return {
            "model": artifact_key,
            "prediction": positive_label if pred == 1 else negative_label,
            "probability": confidence,
            "explanation": explanation,
        }
