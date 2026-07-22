import pandas as pd


REQUIRED_RFM_COLUMNS = frozenset(
    {
        "customer_id",
        "recency_days",
        "order_count",
        "total_spend",
    }
)

PERCENTILE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def validate_rfm_input(data: pd.DataFrame) -> None:
    """Validate the columns and values required for RFM scoring."""
    missing_columns = REQUIRED_RFM_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    rfm_values = data[
        [
            "recency_days",
            "order_count",
            "total_spend",
        ]
    ]

    if rfm_values.isna().any().any():
        raise ValueError("RFM values cannot be missing")


def score_percentile(values: pd.Series) -> pd.Series:
    """Assign scores from 1 to 5 using percentile ranks."""
    percentile_rank = values.rank(
        method="average",
        pct=True,
    )

    scores = pd.cut(
        percentile_rank,
        bins=PERCENTILE_BINS,
        labels=False,
        include_lowest=True,
    )

    return (scores + 1).astype("int64")


def assign_rfm_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Assign recency, frequency and monetary scores."""
    validate_rfm_input(data)

    result = data.copy()

    result["recency_score"] = (
        6 - score_percentile(result["recency_days"])
    )
    result["frequency_score"] = score_percentile(
        result["order_count"]
    )
    result["monetary_score"] = score_percentile(
        result["total_spend"]
    )

    result["rfm_score"] = result[
        [
            "recency_score",
            "frequency_score",
            "monetary_score",
        ]
    ].sum(axis=1)

    return result