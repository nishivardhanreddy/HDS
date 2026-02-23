from __future__ import annotations

from typing import Iterable

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def infer_feature_types(columns: Iterable[str], data_frame) -> tuple[list[str], list[str]]:
    numeric = data_frame.select_dtypes(include=[np.number]).columns.intersection(columns).tolist()
    categorical = [col for col in columns if col not in numeric]
    return numeric, categorical


def build_structured_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler: str = "standard",
) -> ColumnTransformer:
    scaler_step = StandardScaler() if scaler == "standard" else MinMaxScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_symptom_vectorizer(max_features: int = 7_500) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )


def fit_transform_with_selection_and_smote(
    preprocessor: ColumnTransformer,
    X_train,
    y_train,
    variance_threshold: float = 0.0,
    random_state: int = 42,
):
    X_train_prepared = preprocessor.fit_transform(X_train)
    selector = VarianceThreshold(threshold=variance_threshold)
    X_train_selected = selector.fit_transform(X_train_prepared)

    smote = SMOTE(random_state=random_state)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)
    except ValueError:
        X_resampled, y_resampled = X_train_selected, y_train
    return X_resampled, y_resampled, selector


def transform_with_preprocessor_and_selector(
    preprocessor: ColumnTransformer,
    selector: VarianceThreshold,
    X_input,
):
    X_prepared = preprocessor.transform(X_input)
    return selector.transform(X_prepared)


def get_selected_feature_names(
    preprocessor: ColumnTransformer,
    selector: VarianceThreshold,
) -> list[str]:
    all_names = preprocessor.get_feature_names_out()
    mask = selector.get_support()
    return [name for name, keep in zip(all_names, mask) if keep]


def to_dense_if_sparse(matrix):
    if issparse(matrix):
        return matrix.toarray()
    return matrix

