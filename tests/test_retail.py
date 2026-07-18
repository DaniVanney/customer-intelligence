import pandas as pd
import pytest

from customer_intelligence.data.retail import (
    add_derived_columns,
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



def test_add_derived_columns() -> None:
    data = pd.DataFrame(
        {
            "invoice_id": ["100", "C101", "102", "103"],
            "stock_code": ["A", "B", "C", "D"],
            "description": ["One", "Two", "Three", "Four"],
            "quantity": [2, -1, 3, 1],
            "invoice_date": [
                "2011-01-01",
                "2011-01-02",
                "2011-01-03",
                "2011-01-04",
            ],
            "unit_price": [10.0, 10.0, 2.0, 0.0],
            "customer_id": [1, 2, None, 4],
            "country": ["UK", "UK", "France", "Spain"],
        }
    )
    typed_data = coerce_column_types(data)

    result = add_derived_columns(typed_data)

    assert result["is_cancellation"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert result["line_total"].tolist() == [20.0, -10.0, 6.0, 0.0]
    assert result["is_valid_purchase"].tolist() == [
        True,
        False,
        False,
        False,
    ]