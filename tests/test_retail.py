import pandas as pd

from customer_intelligence.data.retail import standardize_column_names


def test_standardize_column_names() -> None:
    raw_columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    expected_columns = [
        "invoice_id",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]

    raw_data = pd.DataFrame(columns=raw_columns)

    result = standardize_column_names(raw_data)

    assert result.columns.tolist() == expected_columns
    assert raw_data.columns.tolist() == raw_columns