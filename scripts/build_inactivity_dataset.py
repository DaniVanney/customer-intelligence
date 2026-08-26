from pathlib import Path

import pandas as pd

from customer_intelligence.churn.dataset import build_inactivity_dataset


INPUT_PATH = Path("data/interim/online_retail.parquet")
OUTPUT_PATH = Path("data/processed/inactivity_modeling.parquet")

CUTOFF_DATES = [
    "2010-09-01",
    "2010-12-01",
    "2011-03-01",
    "2011-06-01",
    "2011-09-10"
]

OBSERVATION_DAYS = 180
PREDICTION_DAYS = 90


def main():
    print(f"Loading retail data from {INPUT_PATH}...")

    retail_data = pd.read_parquet(INPUT_PATH)

    print("Building temporal inactivity snapshots...")

    inactivity_data = build_inactivity_dataset(
        retail_data,
        cutoff_dates=CUTOFF_DATES,
        observation_days=OBSERVATION_DAYS,
        prediction_days=PREDICTION_DAYS
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    inactivity_data.to_parquet(OUTPUT_PATH, index=False)

    summary = inactivity_data.groupby("snapshot_date").agg(
        customers=("customer_id", "size"),
        inactive_customers=("is_inactive", "sum")
    )

    summary["inactivity_rate_pct"] = summary["inactive_customers"] / summary["customers"] * 100

    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)

    print()
    print("INACTIVITY MODELING DATASET")
    print(f"Rows: {len(inactivity_data):,}")
    print(f"Unique customers: {inactivity_data['customer_id'].nunique():,}")
    print(f"Snapshots: {inactivity_data['snapshot_date'].nunique()}")
    print()

    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(summary.to_string())

    print()
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("Inactivity modeling dataset created successfully.")


if __name__ == "__main__":
    main()