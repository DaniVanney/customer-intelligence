import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


MODEL_FEATURES = [
    "recency_days",
    "tenure_days",
    "order_count",
    "total_spend",
    "unique_products",
    "average_order_value",
    "orders_last_30d",
    "orders_last_90d",
    "spend_last_90d",
    "unique_products_last_90d"
]

MODEL_LOG_FEATURES = [
    "order_count",
    "total_spend",
    "unique_products",
    "average_order_value",
    "orders_last_30d",
    "orders_last_90d",
    "spend_last_90d",
    "unique_products_last_90d"
]

MODEL_SCALE_ONLY_FEATURES = ["recency_days", "tenure_days"]

TARGET_COLUMN = "is_inactive"

ALERT_THRESHOLD = 0.45

RISK_BINS = [-np.inf, 0.30, 0.45, 0.65, np.inf]

RISK_LABELS = ["Bajo", "Moderado", "Alto", "Crítico"]


def validate_inactivity_input(data, require_target=False):
    required_columns = set(MODEL_FEATURES)

    if require_target:
        required_columns.add(TARGET_COLUMN)

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Inactivity modeling values cannot be missing")

    if require_target and not data[TARGET_COLUMN].isin([0, 1, False, True]).all():
        raise ValueError("The inactivity target must be binary")

    if data[MODEL_FEATURES].lt(0).any().any():
        raise ValueError("Inactivity modeling features cannot be negative")



def build_inactivity_pipeline():
    log_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_features", log_pipeline, MODEL_LOG_FEATURES),
            ("scale_only", StandardScaler(), MODEL_SCALE_ONLY_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ]
    )

    return model


def fit_inactivity_model(data):
    validate_inactivity_input(data, require_target=True)

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The target must contain both activity classes")

    model = build_inactivity_pipeline()

    model.fit(
        data[MODEL_FEATURES],
        data[TARGET_COLUMN].astype("int64")
    )

    return model


def score_inactivity_risk(data, model):
    validate_inactivity_input(data)

    result = data.copy()

    result["inactivity_risk_score"] = model.predict_proba(result[MODEL_FEATURES])[:, 1]

    result["risk_level"] = pd.cut(
        result["inactivity_risk_score"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False
    ).astype("string")

    result["is_alert"] = result["inactivity_risk_score"] >= ALERT_THRESHOLD

    return result