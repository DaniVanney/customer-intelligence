import pandas as pd
import pytest

from customer_intelligence.data.crm import prepare_sales_pipeline


def make_pipeline_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "opportunity_id": ["1", "2"],
            "sales_agent": ["Agent A", "Agent B"],
            "product": ["GTXPro", "MG Special"],
            "account": ["Account A", None],
            "deal_stage": ["Won", "Engaging"],
            "engage_date": ["2017-01-01", "2017-02-01"],
            "close_date": ["2017-03-01", None],
            "close_value": [1000, None],
        }
    )


def test_prepare_sales_pipeline() -> None:
    result = prepare_sales_pipeline(make_pipeline_data())

    assert result["product"].tolist() == ["GTX Pro", "MG Special"]
    assert result["is_closed"].tolist() == [True, False]
    assert result["is_won"].tolist() == [True, pd.NA]
    assert pd.api.types.is_datetime64_any_dtype(result["engage_date"])
    assert pd.api.types.is_float_dtype(result["close_value"])
    assert pd.isna(result.loc[1, "account"])


def test_prepare_sales_pipeline_rejects_missing_columns() -> None:
    data = make_pipeline_data().drop(columns="account")

    with pytest.raises(ValueError, match="Missing columns"):
        prepare_sales_pipeline(data)


def test_prepare_sales_pipeline_rejects_unknown_stage() -> None:
    data = make_pipeline_data()
    data.loc[0, "deal_stage"] = "Unknown"

    with pytest.raises(ValueError, match="Unexpected deal stages"):
        prepare_sales_pipeline(data)