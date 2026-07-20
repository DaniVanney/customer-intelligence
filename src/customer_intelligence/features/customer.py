import pandas as pd

REQUIRED_INPUT_COLUMNS = frozenset(
    {
        "customer_id",
        "invoice_id",
        "invoice_date",
        "stock_code",
        "quantity",
        "line_total",
        "is_valid_purchase",
    }
)


def validate_customer_feature_input(data: pd.DataFrame) -> None:
    """Validate the columns required to build customer features."""
    missing_columns = REQUIRED_INPUT_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def build_customer_features(
    data: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build one feature row per customer at a snapshot date."""
    validate_customer_feature_input(data)

    snapshot = pd.Timestamp(snapshot_date).normalize()

    valid_mask = (
        data["is_valid_purchase"].fillna(False)
        & (data["invoice_date"] < snapshot)
    )
    purchases = data.loc[valid_mask].copy()

    features = (
        purchases.groupby("customer_id", as_index=False)
        .agg(
            first_purchase_date=("invoice_date", "min"),
            last_purchase_date=("invoice_date", "max"),
            order_count=("invoice_id", "nunique"),
            total_spend=("line_total", "sum"),
            total_items=("quantity", "sum"),
            unique_products=("stock_code", "nunique"),
        )
    )

    features.insert(1, "snapshot_date", snapshot)

    features["recency_days"] = (
        snapshot - features["last_purchase_date"].dt.normalize()
    ).dt.days

    features["tenure_days"] = (
        snapshot - features["first_purchase_date"].dt.normalize()
    ).dt.days

    features["average_order_value"] = (
        features["total_spend"] / features["order_count"]
    )

    return features.sort_values("customer_id").reset_index(drop=True)