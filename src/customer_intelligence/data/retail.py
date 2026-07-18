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



def coerce_column_types(data: pd.DataFrame) -> pd.DataFrame:
    """Return retail data with explicit and consistent column types."""
    result = data.copy()

    result["invoice_id"] = result["invoice_id"].astype("string")
    result["stock_code"] = result["stock_code"].astype("string")
    result["description"] = result["description"].astype("string")
    result["country"] = result["country"].astype("string")

    result["customer_id"] = (
        pd.to_numeric(result["customer_id"], errors="raise")
        .astype("Int64")
        .astype("string")
    )

    result["quantity"] = pd.to_numeric(
        result["quantity"], errors="raise"
    ).astype("int64")
    result["unit_price"] = pd.to_numeric(
        result["unit_price"], errors="raise"
    ).astype("float64")
    result["invoice_date"] = pd.to_datetime(
        result["invoice_date"], errors="raise"
    )

    return result



def add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add business flags and transaction-line value."""
    result = data.copy()

    result["is_cancellation"] = result["invoice_id"].str.startswith(
        "C", na=False
    )
    result["line_total"] = result["quantity"] * result["unit_price"]
    result["is_valid_purchase"] = (
        result["customer_id"].notna()
        & ~result["is_cancellation"]
        & (result["quantity"] > 0)
        & (result["unit_price"] > 0)
    )

    return result