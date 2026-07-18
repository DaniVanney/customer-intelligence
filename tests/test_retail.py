import pandas as pd
import pytest

from customer_intelligence.data.retail import (
    coerce_column_types,
    standardize_column_names,
)


RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

EXPECTED_COLUMNS = [
    "invoice_id",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
]


def test_standardize_column_names() -> None:
    raw_data = pd.DataFrame(columns=RAW_COLUMNS)

    result = standardize_column_names(raw_data)

    assert result.columns.tolist() == EXPECTED_COLUMNS
    assert raw_data.columns.tolist() == RAW_COLUMNS


def test_standardize_column_names_rejects_missing_columns() -> None:
    raw_data = pd.DataFrame(columns=RAW_COLUMNS[:-1])

    with pytest.raises(ValueError, match="Missing columns"):
        standardize_column_names(raw_data)


def test_standardize_column_names_rejects_unexpected_columns() -> None:
    raw_data = pd.DataFrame(columns=[*RAW_COLUMNS, "Unexpected"])

    with pytest.raises(ValueError, match="Unexpected columns"):
        standardize_column_names(raw_data)



def test_coerce_column_types() -> None:
    standardized_data = pd.DataFrame(
        {
            "invoice_id": [123, "C124"],
            "stock_code": ["A1", "B2"],
            "description": ["Product", None],
            "quantity": [2, -1],
            "invoice_date": ["2011-01-01", "2011-01-02"],
            "unit_price": [3, 4.5],
            "customer_id": [456.0, None],
            "country": ["United Kingdom", "France"],
        }
    )

    result = coerce_column_types(standardized_data)

    assert result["invoice_id"].tolist() == ["123", "C124"]
    assert result.loc[0, "customer_id"] == "456"
    assert pd.isna(result.loc[1, "customer_id"])
    assert pd.api.types.is_string_dtype(result["stock_code"])
    assert pd.api.types.is_integer_dtype(result["quantity"])
    assert pd.api.types.is_float_dtype(result["unit_price"])
    assert pd.api.types.is_datetime64_any_dtype(result["invoice_date"])