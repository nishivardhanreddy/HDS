from __future__ import annotations


def select_models(context: dict) -> list[str]:
    selected: list[str] = []

    if context.get("has_symptoms"):
        selected.append("symptom")

    if context.get("has_lifestyle_or_history"):
        selected.append("lifestyle")

    if context.get("has_diabetes_clinical"):
        selected.append("diabetes")

    if context.get("has_heart_clinical"):
        selected.append("heart")

    return selected

