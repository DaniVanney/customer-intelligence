import pandas as pd
import pytest

from customer_intelligence.data.crm_references import (
    prepare_accounts,
    prepare_products,
    prepare_sales_teams,
)


def test_prepare_accounts() -> None:
    data = pd.DataFrame(
        {
            "account": ["Acme"],
            "sector": ["Technology"],
            "year_established": [2000],
            "revenue": [10],
            "employees": [50],
            "office_location": ["USA"],
            "subsidiary_of": [None],
        }
    )

    result = prepare_accounts(data)

    assert pd.api.types.is_integer_dtype(result["year_established"])
    assert pd.api.types.is_float_dtype(result["revenue"])
    assert pd.isna(result.loc[0, "subsidiary_of"])


def test_prepare_products() -> None:
    data = pd.DataFrame(
        {
            "product": ["GTX Pro"],
            "series": ["GTX"],
            "sales_price": [4821],
        }
    )

    result = prepare_products(data)

    assert result.loc[0, "product"] == "GTX Pro"
    assert pd.api.types.is_float_dtype(result["sales_price"])


def test_prepare_sales_teams() -> None:
    data = pd.DataFrame(
        {
            "sales_agent": ["Agent A"],
            "manager": ["Manager A"],
            "regional_office": ["East"],
        }
    )

    result = prepare_sales_teams(data)

    assert result.loc[0, "regional_office"] == "East"
    assert pd.api.types.is_string_dtype(result["sales_agent"])


def test_reference_table_rejects_duplicate_keys() -> None:
    data = pd.DataFrame(
        {
            "product": ["GTX Pro", "GTX Pro"],
            "series": ["GTX", "GTX"],
            "sales_price": [4821, 4821],
        }
    )

    with pytest.raises(ValueError, match="duplicate values"):
        prepare_products(data)