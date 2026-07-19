from pathlib import Path

from customer_intelligence.data.retail import prepare_retail_data
from customer_intelligence.data.retail_io import (
    load_retail_excel,
    save_parquet,
)


INPUT_PATH = Path(
    "data/raw/online_retail_ii/online_retail_II.xlsx"
)
OUTPUT_PATH = Path(
    "data/interim/online_retail.parquet"
)

print(f"Loading raw data from {INPUT_PATH}...")
raw_data = load_retail_excel(INPUT_PATH)

print(f"Raw rows: {len(raw_data):,}")
print("Applying retail transformations...")
prepared_data = prepare_retail_data(raw_data)

removed_duplicates = len(raw_data) - len(prepared_data)
valid_purchases = int(prepared_data["is_valid_purchase"].sum())
cancellations = int(prepared_data["is_cancellation"].sum())

print(f"Interim rows: {len(prepared_data):,}")
print(f"Removed duplicate copies: {removed_duplicates:,}")
print(f"Valid purchase rows: {valid_purchases:,}")
print(f"Cancellation rows: {cancellations:,}")

print(f"Saving interim data to {OUTPUT_PATH}...")
save_parquet(prepared_data, OUTPUT_PATH)

size_mb = OUTPUT_PATH.stat().st_size / 1024**2
print(f"Output size: {size_mb:.2f} MB")
print("Retail interim dataset created successfully.")