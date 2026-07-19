import pandas as pd
import pytest

from customer_intelligence.data.crm_dataset import (
    build_opportunity_dataset,
)


def create_test_tables():
    pipeline = pd.DataFrame(
        {
            "opportunity_id": ["OP-1", "OP-2"],
            "sales_agent": ["Alex", "Alex"],
            "product": ["GTX Pro", "GTX Pro"],
            "account": [pd.NA, "Acme"],
            "deal_stage": ["Engaging", "Won"],
            "engage_date": pd.to_datetime(
                ["2017-06-01", "2017-05-01"]
            ),
            "close_date": pd.to_datetime([pd.NaT, "2017-06-15"]),
            "close_value": [float("nan"), 1000.0],
            "is_closed": [False, True],
            "is_won": pd.array([pd.NA, True], dtype="boolean"),
        }
    )

    accounts = pd.DataFrame(
        {
            "account": ["Acme"],
            "sector": ["Retail"],
            "year_established": [2000],
            "revenue": [10.0],
            "employees": [50],
            "office_location": ["Boston"],
            "subsidiary_of": pd.array([pd.NA], dtype="string"),
        }
    )

    products = pd.DataFrame(
        {
            "product": ["GTX Pro"],
            "series": ["GTX"],
            "sales_price": [4821.0],
        }
    )

    sales_teams = pd.DataFrame(
        {
            "sales_agent": ["Alex"],
            "manager": ["Morgan"],
            "regional_office": ["East"],
        }
    )

    return pipeline, accounts, products, sales_teams


def test_build_opportunity_dataset_preserves_opportunities():
    pipeline, accounts, products, sales_teams = create_test_tables()

    result = build_opportunity_dataset(
        pipeline,
        accounts,
        products,
        sales_teams,
    )

    assert len(result) == len(pipeline)
    assert result["opportunity_id"].is_unique
    assert result.loc[1, "sector"] == "Retail"
    assert pd.isna(result.loc[0, "account"])
    assert result.loc[0, "series"] == "GTX"
    assert result.loc[0, "manager"] == "Morgan"


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("product", "Unknown Product"),
        ("sales_agent", "Unknown Agent"),
        ("account", "Unknown Account"),
    ],
)
def test_build_opportunity_dataset_rejects_unmatched_values(
    column,
    invalid_value,
):
    pipeline, accounts, products, sales_teams = create_test_tables()
    pipeline.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=column):
        build_opportunity_dataset(
            pipeline,
            accounts,
            products,
            sales_teams,
        )