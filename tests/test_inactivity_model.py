import pandas as pd

import pytest

from customer_intelligence.churn.model import fit_inactivity_model, score_inactivity_risk


def create_modeling_data():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "recency_days": [10, 20, 30, 40, 120, 140, 160, 180],
            "tenure_days": [300, 280, 260, 240, 220, 200, 180, 160],
            "order_count": [12, 10, 8, 6, 2, 1, 1, 1],
            "total_spend": [2400.0, 2000.0, 1600.0, 1200.0, 400.0, 250.0, 180.0, 120.0],
            "unique_products": [80, 70, 60, 50, 20, 15, 10, 8],
            "average_order_value": [200.0, 200.0, 200.0, 200.0, 200.0, 250.0, 180.0, 120.0],
            "orders_last_30d": [2, 2, 1, 1, 0, 0, 0, 0],
            "orders_last_90d": [4, 3, 3, 2, 1, 0, 0, 0],
            "spend_last_90d": [800.0, 650.0, 600.0, 400.0, 100.0, 0.0, 0.0, 0.0],
            "unique_products_last_90d": [30, 25, 20, 15, 5, 0, 0, 0],
            "is_inactive": [False, False, False, False, True, True, True, True]
        }
    )

    return data


def test_fit_and_score_inactivity_model_assigns_risk():
    data = create_modeling_data()

    model = fit_inactivity_model(data)

    scored = score_inactivity_risk(data.drop(columns=["is_inactive"]), model)

    assert len(scored) == len(data)
    assert scored["inactivity_risk_score"].between(0, 1).all()
    assert scored["risk_level"].notna().all()
    assert set(scored["risk_level"].astype("string")) <= {"Bajo", "Moderado", "Alto", "Crítico"}
    assert scored["is_alert"].equals(scored["inactivity_risk_score"].ge(0.45))
    assert set(model.named_steps) == {"preprocessor", "classifier"}


def test_fit_inactivity_model_rejects_missing_columns():
    data = create_modeling_data().drop(columns=["total_spend"])

    with pytest.raises(ValueError, match="Missing required columns"):
        fit_inactivity_model(data)


def test_fit_inactivity_model_rejects_missing_values():
    data = create_modeling_data()

    data.loc[0, "recency_days"] = None

    with pytest.raises(ValueError, match="cannot be missing"):
        fit_inactivity_model(data)


def test_fit_inactivity_model_rejects_negative_values():
    data = create_modeling_data()

    data.loc[0, "total_spend"] = -100.0

    with pytest.raises(ValueError, match="cannot be negative"):
        fit_inactivity_model(data)


def test_fit_inactivity_model_requires_both_classes():
    data = create_modeling_data()

    data["is_inactive"] = False

    with pytest.raises(ValueError, match="both activity classes"):
        fit_inactivity_model(data)


def test_fit_inactivity_model_rejects_non_binary_target():
    data = create_modeling_data()

    data["is_inactive"] = data["is_inactive"].astype("int64")

    data.loc[0, "is_inactive"] = 2

    with pytest.raises(ValueError, match="binary"):
        fit_inactivity_model(data)