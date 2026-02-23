from __future__ import annotations


def _has_any_value(data: dict) -> bool:
    for value in data.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _count_present_values(data: dict) -> int:
    count = 0
    for value in data.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        count += 1
    return count


def build_context(payload: dict) -> dict:
    symptoms = payload.get("symptoms", []) or []
    lifestyle = payload.get("lifestyle", {}) or {}
    history = payload.get("history", {}) or {}
    diabetes_clinical = payload.get("diabetes_clinical", {}) or {}
    heart_clinical = payload.get("heart_clinical", {}) or {}

    has_symptoms = len(symptoms) > 0
    has_lifestyle = _has_any_value(lifestyle)
    has_history = _has_any_value(history)
    has_lifestyle_or_history = has_lifestyle or has_history
    has_diabetes_clinical = _count_present_values(diabetes_clinical) >= 2
    has_heart_clinical = _count_present_values(heart_clinical) >= 2

    active_modalities = []
    if has_symptoms:
        active_modalities.append("symptom")
    if has_lifestyle_or_history:
        active_modalities.append("lifestyle")
    if has_diabetes_clinical:
        active_modalities.append("diabetes")
    if has_heart_clinical:
        active_modalities.append("heart")

    return {
        "has_symptoms": has_symptoms,
        "has_lifestyle_or_history": has_lifestyle_or_history,
        "has_diabetes_clinical": has_diabetes_clinical,
        "has_heart_clinical": has_heart_clinical,
        "active_modalities": active_modalities,
        "is_multimodal": len(active_modalities) > 1,
    }

