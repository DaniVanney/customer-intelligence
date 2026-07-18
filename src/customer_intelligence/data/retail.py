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


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized retail column names."""
    return data.rename(columns=COLUMN_MAPPING).copy()