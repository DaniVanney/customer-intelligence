import numpy as np
import pandas as pd

from customer_intelligence.forecasting.model import (
    fit_opportunity_forecast_model,
    score_opportunities
)


def create_modeling_data():
    return pd.DataFrame(
        {
            "opportunity_id": [
                f"OPP-{number}"
                for number in range(16)
            ],
            "product": [
                "GTX Basic",
                "GTX Pro"
            ] * 8,
            "sales_agent": [
                "Agent A",
                "Agent B",
                "Agent C",
                "Agent D"
            ] * 4,
            "regional_office": [
                "Central",
                "West"
            ] * 8,
            "engage_month": [
                1,
                2,
                3,
                4
            ] * 4,
            "sales_price": [
                550.0,
                4821.0
            ] * 8,
            "is_won": [
                True,
                False,
                True,
                False
            ] * 4
        }
    )


def create_scoring_data():
    return pd.DataFrame(
        {
            "opportunity_id": [
                "SCORE-1",
                "SCORE-2",
                "SCORE-3"
            ],
            "product": [
                "GTX Basic",
                "GTX Pro",
                "GTX Basic"
            ],
            "sales_agent": [
                "Agent A",
                "Agent B",
                "Agent C"
            ],
            "regional_office": [
                "Central",
                "West",
                "Central"
            ],
            "engage_month": [
                5,
                6,
                7
            ],
            "sales_price": [
                550.0,
                4821.0,
                550.0
            ]
        }
    )


def test_score_opportunities_builds_expected_value_forecast():
    modeling_data = create_modeling_data()

    training_data = modeling_data.iloc[:8].copy()
    calibration_data = modeling_data.iloc[8:].copy()

    scoring_data = create_scoring_data()

    model = fit_opportunity_forecast_model(
        training_data,
        calibration_data
    )

    result = score_opportunities(
        scoring_data,
        model
    )

    assert len(result) == len(scoring_data)
    assert result["opportunity_id"].is_unique
    assert result["win_probability"].between(0, 1).all()
    assert result["expected_value"].ge(0).all()

    np.testing.assert_allclose(
        result["expected_value"],
        result["win_probability"]
        * result["sales_price"]
    )

    assert sorted(result["forecast_rank"].tolist()) == [
        1,
        2,
        3
    ]