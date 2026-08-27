import pandas as pd
import pytest

from customer_intelligence.forecasting.dataset import build_opportunity_datasets


def create_opportunity_data():
    return pd.DataFrame(
        {
            "opportunity_id": ["OPP-1", "OPP-2", "OPP-3", "OPP-4"],
            "sales_agent": ["Agent A", "Agent B", "Agent A", "Agent B"],
            "product": ["Product A", "Product B", "Product A", "Product B"],
            "deal_stage": ["Won", "Lost", "Engaging", "Prospecting"],
            "engage_date": pd.to_datetime(
                [
                    "2017-01-10",
                    "2017-01-15",
                    "2017-02-01",
                    None
                ]
            ),
            "close_date": pd.to_datetime(
                [
                    "2017-02-10",
                    "2017-02-15",
                    None,
                    None
                ]
            ),
            "close_value": [1000.0, 0.0, None, None],
            "sales_price": [500.0, 750.0, 500.0, 750.0],
            "manager": ["Manager A", "Manager B", "Manager A", "Manager B"],
            "regional_office": ["East", "West", "East", "West"]
        }
    )


def test_build_opportunity_datasets_separates_business_populations():
    data = create_opportunity_data()

    modeling_data, scoring_data = build_opportunity_datasets(data)

    assert modeling_data["opportunity_id"].tolist() == ["OPP-1", "OPP-2"]
    assert modeling_data["is_won"].tolist() == [True, False]
    assert scoring_data["opportunity_id"].tolist() == ["OPP-3"]

def test_build_opportunity_datasets_rejects_incomplete_closed_outcomes():
    data = create_opportunity_data()

    data.loc[0, "close_value"] = None

    with pytest.raises(
        ValueError,
        match="Closed opportunities must have close dates and values"
    ):
        build_opportunity_datasets(data)

def test_build_opportunity_datasets_adds_engagement_time_features():
    data = create_opportunity_data()

    modeling_data, scoring_data = build_opportunity_datasets(data)

    assert modeling_data["engage_year"].tolist() == [2017, 2017]
    assert modeling_data["engage_month"].tolist() == [1, 1]
    assert modeling_data["engage_quarter"].tolist() == [1, 1]

    assert scoring_data.loc[0, "engage_year"] == 2017
    assert scoring_data.loc[0, "engage_month"] == 2
    assert scoring_data.loc[0, "engage_quarter"] == 1