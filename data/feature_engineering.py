from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


INVALID_BRFFS_CODES = {
    7,
    8,
    9,
    77,
    88,
    99,
    777,
    888,
    999,
    9999,
}


def bmi_risk_category(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 18.5:
        return "underweight"
    if value < 25:
        return "normal"
    if value < 30:
        return "overweight"
    return "obese"


def add_bmi_risk_category(df: pd.DataFrame, bmi_col: str = "bmi", out_col: str = "bmi_risk_category") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out[bmi_col].apply(bmi_risk_category)
    return out


def add_chronic_disease_indicators(
    df: pd.DataFrame,
    diabetes_col: str = "diabetes_history",
    heart_col: str = "heart_disease_history",
    hypertension_col: str = "hypertension_history",
) -> pd.DataFrame:
    out = df.copy()
    out["chronic_disease_count"] = (
        out[[diabetes_col, heart_col, hypertension_col]]
        .fillna(0)
        .astype(float)
        .clip(lower=0, upper=1)
        .sum(axis=1)
    )
    return out


def _normalize_symptom_token(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def extract_symptom_list(row: pd.Series, symptom_columns: list[str]) -> list[str]:
    values = []
    for col in symptom_columns:
        token = _normalize_symptom_token(row.get(col, ""))
        if token and token != "nan":
            values.append(token)
    return values


def compute_symptom_frequency(df: pd.DataFrame, symptom_columns: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, row in df[symptom_columns].iterrows():
        counts.update(extract_symptom_list(row, symptom_columns))
    return dict(counts)


def _frequency_bucket(frequency: int, q1: float, q2: float) -> str:
    if frequency >= q2:
        return "high"
    if frequency >= q1:
        return "medium"
    return "low"


def build_symptom_documents(
    symptom_frame: pd.DataFrame,
    symptom_columns: list[str],
    severity_map: dict[str, int] | None = None,
    frequency_meta: dict | None = None,
    fit_frequency: bool = False,
) -> tuple[list[str], dict]:
    if fit_frequency:
        frequency_map = compute_symptom_frequency(symptom_frame, symptom_columns)
        values = np.array(list(frequency_map.values()) or [1], dtype=float)
        q1, q2 = np.quantile(values, [0.4, 0.8]).tolist()
        frequency_meta = {"frequency_map": frequency_map, "q1": q1, "q2": q2}

    if frequency_meta is None:
        frequency_meta = {"frequency_map": {}, "q1": 1.0, "q2": 2.0}

    frequency_map = frequency_meta["frequency_map"]
    q1 = frequency_meta["q1"]
    q2 = frequency_meta["q2"]

    docs: list[str] = []
    for _, row in symptom_frame[symptom_columns].iterrows():
        symptoms = extract_symptom_list(row, symptom_columns)
        tokens: list[str] = []
        severity_acc = 0
        for symptom in symptoms:
            tokens.append(symptom)
            bucket = _frequency_bucket(frequency_map.get(symptom, 0), q1, q2)
            tokens.append(f"freqbin_{bucket}")
            tokens.append(f"{symptom}_freq_{bucket}")
            if severity_map:
                severity_acc += int(severity_map.get(symptom, 1))

        if severity_map:
            if severity_acc >= 12:
                tokens.append("severity_high")
            elif severity_acc >= 6:
                tokens.append("severity_medium")
            else:
                tokens.append("severity_low")

        docs.append(" ".join(tokens))

    return docs, frequency_meta


def _clean_code_series(series: pd.Series, invalid_codes: Iterable[int] = INVALID_BRFFS_CODES) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(numeric.isin(list(invalid_codes)))


def _decode_alcohol_days(alcohol_series: pd.Series) -> pd.Series:
    s = _clean_code_series(alcohol_series, invalid_codes={777, 999})
    out = pd.Series(np.nan, index=s.index, dtype=float)

    week_mask = s.between(101, 199, inclusive="both")
    month_mask = s.between(201, 299, inclusive="both")
    none_mask = s == 888

    out.loc[week_mask] = s.loc[week_mask] - 100
    out.loc[month_mask] = (s.loc[month_mask] - 200) / 4.345
    out.loc[none_mask] = 0.0
    return out


def prepare_lifestyle_features(brfss_df: pd.DataFrame) -> pd.DataFrame:
    df = brfss_df.copy()

    smoker = _clean_code_series(df.get("_SMOKER3", np.nan), invalid_codes={9})
    smoking_status = pd.Series(np.nan, index=df.index, dtype=float)
    smoking_status.loc[smoker == 4] = 0
    smoking_status.loc[smoker == 3] = 1
    smoking_status.loc[smoker.isin([1, 2])] = 2

    activity = _clean_code_series(df.get("_TOTINDA", np.nan), invalid_codes={9})
    physically_inactive = pd.Series(np.nan, index=df.index, dtype=float)
    physically_inactive.loc[activity == 1] = 0
    physically_inactive.loc[activity == 2] = 1

    bmi = _clean_code_series(df.get("_BMI5", np.nan), invalid_codes={9999}) / 100.0
    sleep = _clean_code_series(df.get("SLEPTIM1", np.nan), invalid_codes={77, 99})
    alcohol_per_week = _decode_alcohol_days(df.get("ALCDAY5", np.nan))

    rf_hype = _clean_code_series(df.get("_RFHYPE5", np.nan), invalid_codes={9})
    bphigh = _clean_code_series(df.get("BPHIGH4", np.nan), invalid_codes={7, 9})
    hypertension_history = pd.Series(np.nan, index=df.index, dtype=float)
    hypertension_history.loc[(rf_hype == 1) | (bphigh == 3)] = 0
    hypertension_history.loc[(rf_hype == 2) | (bphigh.isin([1, 2, 4]))] = 1

    diab = _clean_code_series(df.get("DIABETE3", np.nan), invalid_codes={7, 9})
    diabetes_history = pd.Series(np.nan, index=df.index, dtype=float)
    diabetes_history.loc[diab == 3] = 0
    diabetes_history.loc[diab.isin([1, 2, 4])] = 1

    heart_direct = _clean_code_series(df.get("CVDCRHD4", np.nan), invalid_codes={7, 9})
    heart_comp = _clean_code_series(df.get("_MICHD", np.nan), invalid_codes={9})
    heart_disease_history = pd.Series(np.nan, index=df.index, dtype=float)
    heart_disease_history.loc[(heart_direct == 2) | (heart_comp == 2)] = 0
    heart_disease_history.loc[(heart_direct == 1) | (heart_comp == 1)] = 1

    age_group = _clean_code_series(df.get("_AGEG5YR", np.nan), invalid_codes={14})
    sex = _clean_code_series(df.get("SEX", np.nan), invalid_codes={7, 9})
    gender = pd.Series("Unknown", index=df.index, dtype=object)
    gender.loc[sex == 1] = "Male"
    gender.loc[sex == 2] = "Female"

    health_target_raw = _clean_code_series(df.get("_RFHLTH", np.nan), invalid_codes={9})
    target_health_risk = pd.Series(np.nan, index=df.index, dtype=float)
    target_health_risk.loc[health_target_raw == 1] = 0
    target_health_risk.loc[health_target_raw == 2] = 1

    clean = pd.DataFrame(
        {
            "smoking_status": smoking_status,
            "alcohol_per_week": alcohol_per_week,
            "physically_inactive": physically_inactive,
            "sleep_hours": sleep,
            "bmi": bmi,
            "hypertension_history": hypertension_history,
            "diabetes_history": diabetes_history,
            "heart_disease_history": heart_disease_history,
            "age_group": age_group,
            "gender": gender,
            "target_health_risk": target_health_risk,
        }
    )

    clean["bmi_risk_category"] = clean["bmi"].apply(bmi_risk_category)
    clean = add_chronic_disease_indicators(clean)

    core = [
        "smoking_status",
        "alcohol_per_week",
        "physically_inactive",
        "sleep_hours",
        "bmi",
        "hypertension_history",
        "diabetes_history",
        "heart_disease_history",
        "age_group",
        "gender",
    ]
    clean = clean.dropna(subset=["target_health_risk"])
    clean = clean.dropna(subset=core, how="all")
    return clean

