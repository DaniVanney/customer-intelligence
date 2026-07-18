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