import pandas as pd


REQUIRED_COLUMNS = frozenset(
    {
        "opportunity_id",
        "deal_stage",
        "engage_date",
        "close_date",
        "close_value"
    }
)


def validate_opportunity_data(data: pd.DataFrame) -> None:
    """Validate the minimum data required for opportunity modeling."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data["opportunity_id"].isna().any():
        raise ValueError("Opportunity IDs cannot be missing")

    if data["opportunity_id"].duplicated().any():
        raise ValueError("Opportunity IDs must be unique")


def add_engagement_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add time features available when an opportunity is engaged."""
    result = data.copy()

    result["engage_date"] = pd.to_datetime(
        result["engage_date"],
        errors="raise"
    )
    result["engage_year"] = result["engage_date"].dt.year
    result["engage_month"] = result["engage_date"].dt.month
    result["engage_quarter"] = result["engage_date"].dt.quarter

    return result


def build_opportunity_datasets(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build modeling and operational scoring opportunity datasets."""
    validate_opportunity_data(data)

    modeling_data = data.loc[
        data["deal_stage"].isin(["Won", "Lost"])
    ].copy()

    scoring_data = data.loc[
        data["deal_stage"].eq("Engaging")
    ].copy()

    closed_outcome_columns = ["close_date", "close_value"]

    if modeling_data[closed_outcome_columns].isna().any().any():
        raise ValueError(
            "Closed opportunities must have close dates and values"
        )

    if modeling_data["engage_date"].isna().any():
        raise ValueError("Closed opportunities must have an engage date")

    if scoring_data["engage_date"].isna().any():
        raise ValueError("Engaging opportunities must have an engage date")

    modeling_data["is_won"] = (
        modeling_data["deal_stage"]
        .eq("Won")
        .astype("boolean")
    )

    modeling_data = add_engagement_time_features(modeling_data)
    scoring_data = add_engagement_time_features(scoring_data)

    return (
        modeling_data.reset_index(drop=True),
        scoring_data.reset_index(drop=True)
    )