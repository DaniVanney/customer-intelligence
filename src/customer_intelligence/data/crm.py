import pandas as pd


EXPECTED_PIPELINE_COLUMNS = frozenset(
    {
        "opportunity_id",
        "sales_agent",
        "product",
        "account",
        "deal_stage",
        "engage_date",
        "close_date",
        "close_value",
    }
)

VALID_DEAL_STAGES = frozenset(
    {
        "Prospecting",
        "Engaging",
        "Won",
        "Lost",
    }
)

PRODUCT_NAME_CORRECTIONS = {
    "GTXPro": "GTX Pro",
}


def validate_pipeline_schema(data: pd.DataFrame) -> None:
    """Validate columns and deal-stage values in the CRM pipeline."""
    actual_columns = set(data.columns)
    missing_columns = EXPECTED_PIPELINE_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_PIPELINE_COLUMNS

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Invalid CRM pipeline schema. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )

    if data["deal_stage"].isna().any():
        raise ValueError("CRM pipeline contains missing deal stages.")

    unexpected_stages = (
        set(data["deal_stage"].unique()) - VALID_DEAL_STAGES
    )
    if unexpected_stages:
        raise ValueError(
            f"Unexpected deal stages: {sorted(unexpected_stages)}."
        )


def prepare_sales_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare the CRM sales pipeline for analysis and modelling."""
    validate_pipeline_schema(data)
    result = data.copy()

    string_columns = [
        "opportunity_id",
        "sales_agent",
        "product",
        "account",
        "deal_stage",
    ]
    for column in string_columns:
        result[column] = result[column].astype("string")

    result["product"] = result["product"].replace(
        PRODUCT_NAME_CORRECTIONS
    )
    result["engage_date"] = pd.to_datetime(
        result["engage_date"], errors="raise"
    )
    result["close_date"] = pd.to_datetime(
        result["close_date"], errors="raise"
    )
    result["close_value"] = pd.to_numeric(
        result["close_value"], errors="raise"
    ).astype("float64")

    result["is_closed"] = result["deal_stage"].isin(["Won", "Lost"])
    result["is_won"] = (
        result["deal_stage"]
        .map({"Won": True, "Lost": False})
        .astype("boolean")
    )

    return result