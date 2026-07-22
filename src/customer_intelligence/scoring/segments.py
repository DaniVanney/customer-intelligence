import pandas as pd


REQUIRED_SEGMENT_COLUMNS = frozenset(
    {
        "customer_id",
        "recency_score",
        "frequency_score",
        "monetary_score",
    }
)


def validate_segment_input(data: pd.DataFrame) -> None:
    """Validate the columns and values required for segmentation."""
    missing_columns = REQUIRED_SEGMENT_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    score_columns = [
        "recency_score",
        "frequency_score",
        "monetary_score",
    ]
    scores = data[score_columns]

    if scores.isna().any().any():
        raise ValueError("Segment scores cannot be missing")

    invalid_scores = scores.lt(1) | scores.gt(5)

    if invalid_scores.any().any():
        raise ValueError("Scores must be between 1 and 5")


def assign_customer_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Assign an actionable business segment to each customer."""
    validate_segment_input(data)

    result = data.copy()

    recency = result["recency_score"]
    frequency = result["frequency_score"]
    monetary = result["monetary_score"]

    segments = pd.Series(
        "Base general",
        index=result.index,
        dtype="string",
    )
    unassigned = pd.Series(
        True,
        index=result.index,
        dtype="boolean",
    )

    rules = [
        (
            "Valor estratégico",
            (recency >= 4) & (frequency >= 4) & (monetary >= 4),
        ),
        (
            "Incorporación reciente",
            (recency >= 4) & (frequency <= 2),
        ),
        (
            "Lealtad consolidada",
            (recency >= 3) & (frequency >= 4),
        ),
        (
            "Alto valor",
            (recency >= 3) & (monetary >= 4),
        ),
        (
            "Potencial de desarrollo",
            recency >= 4,
        ),
        (
            "Riesgo de inactividad",
            (recency <= 2) & ((frequency >= 3) | (monetary >= 3)),
        ),
        (
            "Inactividad prolongada",
            recency <= 2,
        ),
    ]

    for segment_name, condition in rules:
        selected = unassigned & condition
        segments.loc[selected] = segment_name
        unassigned.loc[selected] = False

    result["segment"] = segments

    return result