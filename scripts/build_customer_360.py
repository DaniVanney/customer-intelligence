from pathlib import Path

import pandas as pd

from customer_intelligence.features.customer import (
    build_customer_features,
)


INPUT_PATH = Path("data/interim/online_retail.parquet")
OUTPUT_PATH = Path("data/processed/customer_360.parquet")
SNAPSHOT_DATE = pd.Timestamp("2011-12-10")


def main() -> None:
    """Build the final transactional Customer 360 table."""
    print(f"Loading retail data from {INPUT_PATH}...")
    transactions = pd.read_parquet(INPUT_PATH)

    print(f"Building customer features at {SNAPSHOT_DATE.date()}...")
    customer_360 = build_customer_features(
        transactions,
        snapshot_date=SNAPSHOT_DATE,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer_360.to_parquet(OUTPUT_PATH, index=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)

    print()
    print("CUSTOMER 360 SUMMARY")
    print(f"Customers: {len(customer_360):,}")
    print(
        "Unique customer IDs: "
        f"{customer_360['customer_id'].nunique():,}"
    )
    print(
        "Total historical spend: "
        f"{customer_360['total_spend'].sum():,.2f}"
    )
    print(
        "Median orders per customer: "
        f"{customer_360['order_count'].median():.1f}"
    )
    print(
        "Median recency in days: "
        f"{customer_360['recency_days'].median():.1f}"
    )
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("Customer 360 dataset created successfully.")


if __name__ == "__main__":
    main()