import pandas as pd


COLUMN_MAPPING = {
    "Invoice": "invoice_id",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

EXPECTED_RAW_COLUMNS = frozenset(COLUMN_MAPPING)


def validate_raw_schema(data: pd.DataFrame) -> None:
    """Raise an error when the raw retail schema is not the expected one."""
    actual_columns = set(data.columns)
    missing_columns = EXPECTED_RAW_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_RAW_COLUMNS

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Invalid retail schema. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw schema and return standardized column names."""
    validate_raw_schema(data)
    return data.rename(columns=COLUMN_MAPPING).copy()