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


def reconcile_cancellations(data: pd.DataFrame) -> pd.DataFrame:
    """Adjust positive purchases using compatible later cancellations."""
    result = data.reset_index(drop=True).copy()

    valid_purchase = (
        result["is_valid_purchase"].fillna(False).astype(bool)
    )

    result["effective_quantity"] = 0
    result.loc[valid_purchase, "effective_quantity"] = result.loc[
        valid_purchase,
        "quantity",
    ]
    result["effective_quantity"] = result[
        "effective_quantity"
    ].astype("int64")

    nonzero_quantity = result["quantity"].ne(0)
    result["_unit_price_key"] = 0.0
    result.loc[nonzero_quantity, "_unit_price_key"] = (
        result.loc[nonzero_quantity, "line_total"]
        / result.loc[nonzero_quantity, "quantity"]
    ).round(6)

    cancellation_mask = (
        result["customer_id"].notna()
        & result["quantity"].lt(0)
    )

    cancellation_keys = set(
        result.loc[
            cancellation_mask,
            ["customer_id", "stock_code", "_unit_price_key"],
        ].itertuples(index=False, name=None)
    )

    result["_is_valid_purchase"] = valid_purchase

    relevant_mask = (
        result["customer_id"].notna()
        & (valid_purchase | cancellation_mask)
    )

    relevant = result.loc[relevant_mask].sort_values(
        "invoice_date",
        kind="stable",
    )

    purchase_stacks: dict[
        tuple[object, object, float],
        list[list[int]],
    ] = {}

    rows = relevant[
        [
            "customer_id",
            "stock_code",
            "quantity",
            "_unit_price_key",
            "_is_valid_purchase",
        ]
    ].itertuples(index=True, name=None)

    for (
        index,
        customer_id,
        stock_code,
        quantity,
        unit_price_key,
        is_valid_purchase,
    ) in rows:
        key = (customer_id, stock_code, unit_price_key)

        if is_valid_purchase:
            if key in cancellation_keys:
                purchase_stacks.setdefault(key, []).append(
                    [index, int(quantity)]
                )
            continue

        quantity_to_cancel = abs(int(quantity))
        stack = purchase_stacks.get(key, [])

        while quantity_to_cancel > 0 and stack:
            purchase = stack[-1]
            matched_quantity = min(
                quantity_to_cancel,
                purchase[1],
            )

            result.loc[
                purchase[0],
                "effective_quantity",
            ] -= matched_quantity

            purchase[1] -= matched_quantity
            quantity_to_cancel -= matched_quantity

            if purchase[1] == 0:
                stack.pop()

    effective_mask = result["effective_quantity"].gt(0)
    result["effective_line_total"] = 0.0

    result.loc[effective_mask, "effective_line_total"] = (
        result.loc[effective_mask, "line_total"]
        * result.loc[effective_mask, "effective_quantity"]
        / result.loc[effective_mask, "quantity"]
    )

    return result.drop(
        columns=["_unit_price_key", "_is_valid_purchase"]
    )


def build_customer_features(
    data: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build one feature row per customer at a snapshot date."""
    validate_customer_feature_input(data)

    snapshot = pd.Timestamp(snapshot_date).normalize()

    history = data.loc[data["invoice_date"] < snapshot].copy()
    reconciled = reconcile_cancellations(history)

    purchases = reconciled.loc[
        reconciled["effective_quantity"].gt(0)
    ].copy()

    features = (
        purchases.groupby("customer_id", as_index=False)
        .agg(
            first_purchase_date=("invoice_date", "min"),
            last_purchase_date=("invoice_date", "max"),
            order_count=("invoice_id", "nunique"),
            total_spend=("effective_line_total", "sum"),
            total_items=("effective_quantity", "sum"),
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