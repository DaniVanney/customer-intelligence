import pandas as pd

from customer_intelligence.features.customer import build_customer_features
from customer_intelligence.features.customer import reconcile_cancellations


def build_inactivity_snapshot(data, cutoff_date, observation_days=180, prediction_days=90):
    cutoff = pd.Timestamp(cutoff_date).normalize()

    prediction_end = cutoff + pd.Timedelta(days=prediction_days)

    customer_features = build_customer_features(data, cutoff)

    eligible_customers = customer_features.loc[customer_features["recency_days"] <= observation_days].copy()

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