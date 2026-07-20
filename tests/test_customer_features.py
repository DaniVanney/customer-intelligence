import pandas as pd

import pytest

from customer_intelligence.features.customer import (
    build_customer_features,
)


def test_build_customer_features_calculates_customer_metrics():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A"],
            "invoice_id": ["INV-1", "INV-1", "INV-2"],
            "invoice_date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-02-15",
                ]
            ),
            "stock_code": ["P1", "P2", "P1"],
            "quantity": [2, 1, 3],
            "line_total": [20.0, 15.0, 30.0],
            "is_valid_purchase": [True, True, True],
        }
    )

    result = build_customer_features(
        data,
        snapshot_date="2021-03-01",
    )

    customer = result.iloc[0]

    assert len(result) == 1
    assert customer["customer_id"] == "A"
    assert customer["order_count"] == 2
    assert customer["total_spend"] == 65.0
    assert customer["total_items"] == 6
    assert customer["unique_products"] == 2
    assert customer["average_order_value"] == 32.5
    assert customer["recency_days"] == 14
    assert customer["tenure_days"] == 59


def test_build_customer_features_ignores_invalid_and_future_rows():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B", "C"],
            "invoice_id": [
                "A-1",
                "A-INVALID",
                "A-FUTURE",
                "B-1",
                "C-FUTURE",
            ],
            "invoice_date": pd.to_datetime(
                [
                    "2021-01-10",
                    "2021-01-20",
                    "2021-03-01",
                    "2021-02-01",
                    "2021-04-01",
                ]
            ),
            "stock_code": ["P1", "P2", "P3", "P1", "P1"],
            "quantity": [1, 100, 200, 2, 300],
            "line_total": [10.0, 1000.0, 2000.0, 20.0, 3000.0],
            "is_valid_purchase": [True, False, True, True, True],
        }
    )

    result = build_customer_features(
        data,
        snapshot_date="2021-03-01",
    )
    customers = result.set_index("customer_id")

    assert set(result["customer_id"]) == {"A", "B"}
    assert len(result) == 2

    assert customers.loc["A", "order_count"] == 1
    assert customers.loc["A", "total_spend"] == 10.0

    assert customers.loc["B", "order_count"] == 1
    assert customers.loc["B", "total_spend"] == 20.0

def test_build_customer_features_rejects_missing_columns():
    incomplete_data = pd.DataFrame(
        {
            "customer_id": ["A"],
            "invoice_id": ["INV-1"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        build_customer_features(
            incomplete_data,
            snapshot_date="2021-03-01",
        )