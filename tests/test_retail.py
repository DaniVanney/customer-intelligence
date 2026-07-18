import pandas as pd
import pytest

from customer_intelligence.data.retail import standardize_column_names


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