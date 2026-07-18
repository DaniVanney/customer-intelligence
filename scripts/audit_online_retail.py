from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/online_retail_ii/online_retail_II.xlsx")

sheets = pd.read_excel(DATA_PATH, sheet_name=None)
data = pd.concat(sheets.values(), ignore_index=True)

rows_by_sheet = {
    sheet_name: len(sheet_data)
    for sheet_name, sheet_data in sheets.items()
}

invoice_text = data["Invoice"].astype("string")
cancellation_mask = invoice_text.str.startswith("C", na=False)

print("ONLINE RETAIL II — RAW DATA AUDIT")
print(f"Rows by sheet: {rows_by_sheet}")
print(f"Combined shape: {data.shape}")
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nData types:")
print(data.dtypes.to_string())

print("\nDate range:")
print(f"Minimum: {data['InvoiceDate'].min()}")
print(f"Maximum: {data['InvoiceDate'].max()}")

print("\nMissing values:")
print(data.isna().sum().to_string())

print("\nUnique entities:")
print(f"Invoices: {data['Invoice'].nunique(dropna=True)}")
print(f"Customers: {data['Customer ID'].nunique(dropna=True)}")
print(f"Products: {data['StockCode'].nunique(dropna=True)}")
print(f"Countries: {data['Country'].nunique(dropna=True)}")

print("\nPotential data-quality issues:")
print(f"Exact duplicate rows: {data.duplicated().sum()}")
print(f"Cancellation rows: {cancellation_mask.sum()}")
print(
    "Cancelled invoices: "
    f"{data.loc[cancellation_mask, 'Invoice'].nunique(dropna=True)}"
)
print(f"Rows with quantity <= 0: {(data['Quantity'] <= 0).sum()}")
print(f"Rows with price <= 0: {(data['Price'] <= 0).sum()}")



valid_purchase_mask = (
    data["Customer ID"].notna()
    & ~cancellation_mask
    & (data["Quantity"] > 0)
    & (data["Price"] > 0)
)

purchases = data.loc[valid_purchase_mask].copy()

invoice_counts = purchases.groupby("Customer ID")["Invoice"].nunique()
repeat_customers = (invoice_counts >= 2).sum()
single_purchase_customers = (invoice_counts == 1).sum()

print("\nCUSTOMER RECURRENCE")
print(f"Customers with valid purchases: {len(invoice_counts)}")
print(f"Customers with one invoice: {single_purchase_customers}")
print(f"Customers with two or more invoices: {repeat_customers}")
print(f"Repeat-customer rate: {repeat_customers / len(invoice_counts):.2%}")



observation_days = 180
prediction_days = 90

cutoff_dates = pd.to_datetime(
    [
        "2010-09-01",
        "2010-12-01",
        "2011-03-01",
        "2011-06-01",
        "2011-09-10",
    ]
)

print("\nINACTIVITY FEASIBILITY")
print("Cutoff | Eligible customers | Inactive customers | Inactivity rate")

for cutoff_date in cutoff_dates:
    observation_start = cutoff_date - pd.Timedelta(days=observation_days)
    prediction_end = cutoff_date + pd.Timedelta(days=prediction_days)

    observation_mask = (
        (purchases["InvoiceDate"] >= observation_start)
        & (purchases["InvoiceDate"] < cutoff_date)
    )
    prediction_mask = (
        (purchases["InvoiceDate"] >= cutoff_date)
        & (purchases["InvoiceDate"] < prediction_end)
    )

    eligible_customers = pd.Index(
        purchases.loc[observation_mask, "Customer ID"].unique()
    )
    future_customers = purchases.loc[
        prediction_mask, "Customer ID"
    ].unique()

    inactive_mask = ~eligible_customers.isin(future_customers)
    inactive_count = inactive_mask.sum()
    inactivity_rate = inactive_count / len(eligible_customers)

    print(
        f"{cutoff_date.date()} | "
        f"{len(eligible_customers)} | "
        f"{inactive_count} | "
        f"{inactivity_rate:.2%}"
    )