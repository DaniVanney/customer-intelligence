import pandas as pd

from customer_intelligence.churn.dataset import build_inactivity_dataset, build_inactivity_features, build_inactivity_snapshot

def test_build_inactivity_snapshot_creates_temporal_target():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "A"],
            "invoice_id": ["INV-1", "INV-2", "INV-3", "INV-4"],
            "invoice_date": pd.to_datetime(["2021-01-10", "2021-02-01", "2020-08-01", "2021-03-15"]),
            "stock_code": ["P1", "P2", "P3", "P4"],
            "quantity": [2, 1, 3, 1],
            "line_total": [20.0, 15.0, 30.0, 12.0],
            "is_valid_purchase": [True, True, True, True]
        }
    )

    result = build_inactivity_snapshot(data, cutoff_date="2021-03-01", observation_days=180, prediction_days=90)

    customers = result.set_index("customer_id")

    assert set(customers.index) == {"A", "B"}
    assert not customers.loc["A", "is_inactive"]
    assert customers.loc["B", "is_inactive"]

    assert customers.loc["A", "order_count"] == 1
    assert customers.loc["A", "total_spend"] == 20.0
    assert customers.loc["A", "snapshot_date"] == pd.Timestamp("2021-03-01")
    assert customers.loc["A", "prediction_end_date"] == pd.Timestamp("2021-05-30")

    assert customers.loc["A", "orders_last_30d"] == 0
    assert customers.loc["A", "orders_last_90d"] == 1
    assert customers.loc["A", "spend_last_90d"] == 20.0
    assert customers.loc["A", "items_last_90d"] == 2
    assert customers.loc["A", "unique_products_last_90d"] == 1

    assert customers.loc["B", "orders_last_30d"] == 1
    assert customers.loc["B", "orders_last_90d"] == 1
    assert customers.loc["B", "spend_last_90d"] == 15.0


def test_build_inactivity_snapshot_ignores_fully_cancelled_future_purchase():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A"],
            "invoice_id": ["INV-1", "INV-2", "C-INV-2"],
            "invoice_date": pd.to_datetime(["2021-02-01", "2021-03-15", "2021-03-16"]),
            "stock_code": ["P1", "P2", "P2"],
            "quantity": [1, 10, -10],
            "line_total": [20.0, 100.0, -100.0],
            "is_valid_purchase": [True, True, False]
        }
    )

    result = build_inactivity_snapshot(data, cutoff_date="2021-03-01", observation_days=180, prediction_days=90)

    customer = result.iloc[0]

    assert customer["is_inactive"]
    assert customer["order_count"] == 1
    assert customer["total_spend"] == 20.0


def test_build_inactivity_dataset_combines_temporal_snapshots():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "A", "A", "A"],
            "invoice_id": ["INV-1", "INV-2", "INV-3", "INV-4", "INV-5"],
            "invoice_date": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-03-10", "2021-05-20", "2021-06-15"]),
            "stock_code": ["P1", "P2", "P3", "P4", "P5"],
            "quantity": [1, 1, 1, 1, 1],
            "line_total": [20.0, 15.0, 25.0, 30.0, 35.0],
            "is_valid_purchase": [True, True, True, True, True]
        }
    )

    result = build_inactivity_dataset(data, cutoff_dates=["2021-03-01", "2021-06-01"], observation_days=180, prediction_days=30)

    targets = result.pivot(index="customer_id", columns="snapshot_date", values="is_inactive")

    assert len(result) == 4
    assert not result.duplicated(["customer_id", "snapshot_date"]).any()
    assert not targets.loc["A", pd.Timestamp("2021-03-01")]
    assert targets.loc["B", pd.Timestamp("2021-03-01")]
    assert not targets.loc["A", pd.Timestamp("2021-06-01")]
    assert targets.loc["B", pd.Timestamp("2021-06-01")]

def test_build_inactivity_features_creates_scoring_snapshot_without_target():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "A"],
            "invoice_id": ["INV-1", "INV-2", "INV-3"],
            "invoice_date": pd.to_datetime(["2021-02-01", "2020-01-01", "2021-03-15"]),
            "stock_code": ["P1", "P2", "P3"],
            "quantity": [2, 1, 3],
            "line_total": [20.0, 15.0, 30.0],
            "is_valid_purchase": [True, True, True]
        }
    )

    result = build_inactivity_features(
        data,
        cutoff_date="2021-03-01",
        observation_days=180
    )

    customer = result.iloc[0]

    assert result["customer_id"].tolist() == ["A"]
    assert customer["snapshot_date"] == pd.Timestamp("2021-03-01")
    assert customer["order_count"] == 1
    assert customer["orders_last_30d"] == 1
    assert customer["orders_last_90d"] == 1
    assert "prediction_end_date" not in result.columns
    assert "is_inactive" not in result.columns