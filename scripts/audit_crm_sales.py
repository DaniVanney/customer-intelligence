from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/crm_sales")

tables = {
    csv_path.stem: pd.read_csv(csv_path)
    for csv_path in sorted(DATA_DIR.glob("*.csv"))
}

print("CRM SALES — RAW DATA AUDIT")

for table_name, table in tables.items():
    missing_values = table.isna().sum()
    missing_values = missing_values[missing_values > 0]

    print(f"\nTable: {table_name}")
    print(f"Shape: {table.shape}")
    print(f"Columns: {table.columns.tolist()}")
    print("Missing values:")
    print(missing_values.to_string() if not missing_values.empty else "None")

pipeline = tables["sales_pipeline"]

print("\nOPPORTUNITY AUDIT")
print(f"Duplicate opportunity IDs: {pipeline['opportunity_id'].duplicated().sum()}")

print("\nDeal-stage distribution:")
print(pipeline["deal_stage"].value_counts(dropna=False).to_string())

closed_pipeline = pipeline[
    pipeline["deal_stage"].isin(["Won", "Lost"])
].copy()

won_count = (closed_pipeline["deal_stage"] == "Won").sum()
lost_count = (closed_pipeline["deal_stage"] == "Lost").sum()

print("\nClosed opportunities:")
print(f"Total: {len(closed_pipeline)}")
print(f"Won: {won_count}")
print(f"Lost: {lost_count}")
print(f"Win rate: {won_count / len(closed_pipeline):.2%}")

engage_dates = pd.to_datetime(pipeline["engage_date"], errors="coerce")
close_dates = pd.to_datetime(pipeline["close_date"], errors="coerce")

print("\nDate coverage:")
print(f"First engagement: {engage_dates.min()}")
print(f"Last engagement: {engage_dates.max()}")
print(f"First close: {close_dates.min()}")
print(f"Last close: {close_dates.max()}")

print("\nClose value by stage:")
print(
    pipeline.groupby("deal_stage")["close_value"]
    .agg(["count", "min", "median", "max", "sum"])
    .to_string()
)



print("\nDATA AVAILABILITY BY STAGE")
missing_columns = [
    "account",
    "engage_date",
    "close_date",
    "close_value",
]
print(
    pipeline.groupby("deal_stage")[missing_columns]
    .agg(lambda column: column.isna().sum())
    .to_string()
)

product_coverage = pipeline["product"].isin(
    tables["products"]["product"]
).mean()
sales_agent_coverage = pipeline["sales_agent"].isin(
    tables["sales_teams"]["sales_agent"]
).mean()

known_account_mask = pipeline["account"].notna()
account_coverage = pipeline.loc[
    known_account_mask, "account"
].isin(tables["accounts"]["account"]).mean()

print("\nRELATIONAL COVERAGE")
print(f"Products matched: {product_coverage:.2%}")
print(f"Sales agents matched: {sales_agent_coverage:.2%}")
print(f"Non-missing accounts matched: {account_coverage:.2%}")

closed_pipeline["close_date"] = pd.to_datetime(
    closed_pipeline["close_date"]
)
closed_pipeline["close_quarter"] = (
    closed_pipeline["close_date"].dt.to_period("Q").astype("string")
)

print("\nCLOSED OPPORTUNITIES BY QUARTER")
print(
    pd.crosstab(
        closed_pipeline["close_quarter"],
        closed_pipeline["deal_stage"],
    ).to_string()
)