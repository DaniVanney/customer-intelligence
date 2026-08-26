import pandas as pd

from customer_intelligence.features.customer import build_customer_features, reconcile_cancellations


RECENT_ACTIVITY_COLUMNS = [
    "orders_last_30d",
    "orders_last_90d",
    "spend_last_90d",
    "items_last_90d",
    "unique_products_last_90d"
]


def build_recent_activity_features(data, cutoff_date):
    cutoff = pd.Timestamp(cutoff_date).normalize()

    history = data.loc[data["invoice_date"] < cutoff].copy()

    reconciled_history = reconcile_cancellations(history)

    purchases = reconciled_history.loc[reconciled_history["effective_quantity"] > 0].copy()

    start_30d = cutoff - pd.Timedelta(days=30)

    start_90d = cutoff - pd.Timedelta(days=90)

    purchases_30d = purchases.loc[purchases["invoice_date"] >= start_30d]

    purchases_90d = purchases.loc[purchases["invoice_date"] >= start_90d]

    orders_30d = purchases_30d.groupby("customer_id")["invoice_id"].nunique().rename("orders_last_30d")

    activity_90d = purchases_90d.groupby("customer_id").agg(
        orders_last_90d=("invoice_id", "nunique"),
        spend_last_90d=("effective_line_total", "sum"),
        items_last_90d=("effective_quantity", "sum"),
        unique_products_last_90d=("stock_code", "nunique")
    )

    recent_activity = activity_90d.join(orders_30d, how="outer").fillna(0).reset_index()

    recent_activity = recent_activity[["customer_id", *RECENT_ACTIVITY_COLUMNS]]

    integer_columns = ["orders_last_30d", "orders_last_90d", "items_last_90d", "unique_products_last_90d"]

    recent_activity[integer_columns] = recent_activity[integer_columns].astype("int64")

    return recent_activity


def build_inactivity_snapshot(data, cutoff_date, observation_days=180, prediction_days=90):
    cutoff = pd.Timestamp(cutoff_date).normalize()

    prediction_end = cutoff + pd.Timedelta(days=prediction_days)

    customer_features = build_customer_features(data, cutoff)

    eligible_customers = customer_features.loc[customer_features["recency_days"] <= observation_days].copy()

    recent_activity = build_recent_activity_features(data, cutoff)

    eligible_customers = eligible_customers.merge(recent_activity, on="customer_id", how="left")

    eligible_customers[RECENT_ACTIVITY_COLUMNS] = eligible_customers[RECENT_ACTIVITY_COLUMNS].fillna(0)

    integer_columns = ["orders_last_30d", "orders_last_90d", "items_last_90d", "unique_products_last_90d"]

    eligible_customers[integer_columns] = eligible_customers[integer_columns].astype("int64")

    future_mask = (data["invoice_date"] >= cutoff) & (data["invoice_date"] < prediction_end)

    future_data = data.loc[future_mask].copy()

    reconciled_future = reconcile_cancellations(future_data)

    future_customers = reconciled_future.loc[reconciled_future["effective_quantity"] > 0, "customer_id"].dropna().unique()

    eligible_customers.insert(2, "prediction_end_date", prediction_end)

    eligible_customers["is_inactive"] = ~eligible_customers["customer_id"].isin(future_customers)

    return eligible_customers.sort_values("customer_id").reset_index(drop=True)


def build_inactivity_dataset(data, cutoff_dates, observation_days=180, prediction_days=90):
    snapshots = []

    for cutoff_date in cutoff_dates:
        snapshot = build_inactivity_snapshot(data, cutoff_date, observation_days, prediction_days)

        snapshots.append(snapshot)

    if not snapshots:
        raise ValueError("At least one cutoff date is required")

    result = pd.concat(snapshots, ignore_index=True)

    if result.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Customer and snapshot combinations must be unique")

    return result.sort_values(["snapshot_date", "customer_id"]).reset_index(drop=True)