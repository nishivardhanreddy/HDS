from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_symptom_dataset(path: str | Path = "dataset.csv") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(_resolve_path(path))
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom_")]
    df[symptom_cols] = (
        df[symptom_cols]
        .fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip().str.lower().str.replace(" ", "_", regex=False))
    )
    df["Disease"] = df["Disease"].astype(str).str.strip()
    return df, symptom_cols


def load_symptom_severity(path: str | Path = "Symptom-severity.csv") -> dict[str, int]:
    df = pd.read_csv(_resolve_path(path))
    df["Symptom"] = df["Symptom"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int)
    return dict(zip(df["Symptom"], df["weight"]))


def load_diabetes_dataset(path: str | Path = "diabetes_prediction_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(_resolve_path(path))
    df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce").fillna(0).astype(int)
    return df


def load_heart_dataset(path: str | Path = "heart.csv") -> pd.DataFrame:
    df = pd.read_csv(_resolve_path(path))
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    return df


def _existing_paths(paths: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for p in paths:
        rp = _resolve_path(p)
        if rp.exists():
            resolved.append(rp)
    return resolved


def load_brfss_lifestyle_dataset(
    paths: Sequence[str | Path] | None = None,
    max_rows_per_file: int = 200_000,
) -> pd.DataFrame:
    default_candidates = ["2013.csv", "2015.csv", "2014.csv", "2012.csv", "2011.csv"]
    source_paths = _existing_paths(paths or default_candidates)
    if not source_paths:
        raise FileNotFoundError("No BRFSS CSV file found in the project root.")

    brfss_columns = {
        "_SMOKER3",
        "ALCDAY5",
        "_TOTINDA",
        "SLEPTIM1",
        "_BMI5",
        "_RFHYPE5",
        "BPHIGH4",
        "DIABETE3",
        "CVDCRHD4",
        "_MICHD",
        "_AGEG5YR",
        "SEX",
        "_RFHLTH",
    }

    frames: list[pd.DataFrame] = []
    for source in source_paths:
        frame = pd.read_csv(
            source,
            usecols=lambda c: c in brfss_columns,
            nrows=max_rows_per_file,
            low_memory=False,
        )
        frame["source_file"] = source.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_symptom_reference_tables(
    description_path: str | Path = "symptom_Description.csv",
    precaution_path: str | Path = "symptom_precaution.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptions = pd.read_csv(_resolve_path(description_path))
    precautions = pd.read_csv(_resolve_path(precaution_path))
    descriptions["Disease"] = descriptions["Disease"].astype(str).str.strip()
    precautions["Disease"] = precautions["Disease"].astype(str).str.strip()
    return descriptions, precautions

