from dataclasses import dataclass

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "product",
    "sales_agent",
    "regional_office",
    "engage_month"
]

NUMERIC_FEATURES = ["sales_price"]

MODEL_FEATURES = [
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES
]

TARGET = "is_won"


@dataclass
class OpportunityForecastModel:
    preprocessor: ColumnTransformer
    classifier: LogisticRegression
    calibrator: LogisticRegression


def _validate_modeling_data(
    data: pd.DataFrame,
    dataset_name: str
) -> None:
    required_columns = [
        "opportunity_id",
        *MODEL_FEATURES,
        TARGET
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError(
            f"{dataset_name} cannot be empty"
        )

    if data["opportunity_id"].duplicated().any():
        raise ValueError(
            f"{dataset_name} opportunity IDs must be unique"
        )

    if data[MODEL_FEATURES].isna().any().any():
        raise ValueError(
            f"{dataset_name} model features cannot be missing"
        )

    if data[TARGET].isna().any():
        raise ValueError(
            f"{dataset_name} target cannot be missing"
        )

    target_values = set(
        data[TARGET].unique()
    )

    if not target_values.issubset(
        {
            False,
            True,
            0,
            1
        }
    ):
        raise ValueError(
            f"{dataset_name} target must be binary"
        )

    if len(target_values) != 2:
        raise ValueError(
            f"{dataset_name} must contain both target classes"
        )


def _validate_scoring_data(
    data: pd.DataFrame
) -> None:
    required_columns = [
        "opportunity_id",
        *MODEL_FEATURES
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Scoring data is missing columns: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError(
            "Scoring data cannot be empty"
        )

    if data["opportunity_id"].duplicated().any():
        raise ValueError(
            "Scoring opportunity IDs must be unique"
        )

    if data[MODEL_FEATURES].isna().any().any():
        raise ValueError(
            "Scoring model features cannot be missing"
        )


def _build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    preprocessor.set_output(
        transform="pandas"
    )

    return preprocessor


def fit_opportunity_forecast_model(
    training_data: pd.DataFrame,
    calibration_data: pd.DataFrame
) -> OpportunityForecastModel:
    _validate_modeling_data(
        training_data,
        "Training data"
    )

    _validate_modeling_data(
        calibration_data,
        "Calibration data"
    )

    preprocessor = _build_preprocessor()

    X_training = training_data[
        MODEL_FEATURES
    ].copy()

    y_training = training_data[
        TARGET
    ].astype("int64")

    X_calibration = calibration_data[
        MODEL_FEATURES
    ].copy()

    y_calibration = calibration_data[
        TARGET
    ].astype("int64")

    X_training_preprocessed = (
        preprocessor.fit_transform(
            X_training
        )
    )

    X_calibration_preprocessed = (
        preprocessor.transform(
            X_calibration
        )
    )

    classifier = LogisticRegression(
        C=0.001,
        max_iter=1000,
        random_state=42
    )

    classifier.fit(
        X_training_preprocessed,
        y_training
    )

    calibration_scores = (
        classifier.decision_function(
            X_calibration_preprocessed
        )
        .reshape(-1, 1)
    )

    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    calibrator.fit(
        calibration_scores,
        y_calibration
    )

    return OpportunityForecastModel(
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator
    )


def score_opportunities(
    data: pd.DataFrame,
    model: OpportunityForecastModel
) -> pd.DataFrame:
    _validate_scoring_data(data)

    model_features = data[
        MODEL_FEATURES
    ].copy()

    preprocessed_features = (
        model.preprocessor.transform(
            model_features
        )
    )

    classification_scores = (
        model.classifier.decision_function(
            preprocessed_features
        )
        .reshape(-1, 1)
    )

    win_probabilities = (
        model.calibrator.predict_proba(
            classification_scores
        )[:, 1]
    )

    result = data.copy()

    result["win_probability"] = (
        win_probabilities
    )

    result["expected_value"] = (
        result["win_probability"]
        * result["sales_price"]
    )

    result["forecast_rank"] = (
        result["expected_value"]
        .rank(
            method="first",
            ascending=False
        )
        .astype("int64")
    )

    return result