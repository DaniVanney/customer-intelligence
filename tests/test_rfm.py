import pandas as pd

import pytest

from customer_intelligence.scoring.rfm import assign_rfm_scores


def test_assign_rfm_scores_orders_customers_from_best_to_worst():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E"],
            "recency_days": [1, 10, 20, 30, 40],
            "order_count": [5, 4, 3, 2, 1],
            "total_spend": [500.0, 400.0, 300.0, 200.0, 100.0],
        }
    )

    result = assign_rfm_scores(data)

    assert result["recency_score"].tolist() == [5, 4, 3, 2, 1]
    assert result["frequency_score"].tolist() == [5, 4, 3, 2, 1]
    assert result["monetary_score"].tolist() == [5, 4, 3, 2, 1]
    assert result["rfm_score"].tolist() == [15, 12, 9, 6, 3]


def test_assign_rfm_scores_gives_tied_values_same_score():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E", "F"],
            "recency_days": [1, 2, 3, 4, 5, 6],
            "order_count": [1, 1, 2, 3, 4, 5],
            "total_spend": [
                100.0,
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
            ],
        }
    )

    result = assign_rfm_scores(data)

    assert (
        result.loc[0, "frequency_score"]
        == result.loc[1, "frequency_score"]
    )
    assert (
        result.loc[0, "monetary_score"]
        == result.loc[1, "monetary_score"]
    )

    score_columns = [
        "recency_score",
        "frequency_score",
        "monetary_score",
    ]

    for column in score_columns:
        assert result[column].between(1, 5).all()


def test_assign_rfm_scores_rejects_missing_columns():
    incomplete_data = pd.DataFrame(
        {
            "customer_id": ["A"],
            "recency_days": [10],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        assign_rfm_scores(incomplete_data)


def test_assign_rfm_scores_rejects_missing_values():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "recency_days": [10, None],
            "order_count": [2, 3],
            "total_spend": [100.0, 200.0],
        }
    )

    with pytest.raises(ValueError, match="RFM values cannot be missing"):
        assign_rfm_scores(data)