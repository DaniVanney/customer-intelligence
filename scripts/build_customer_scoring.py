from pathlib import Path

import pandas as pd

from customer_intelligence.scoring.rfm import assign_rfm_scores
from customer_intelligence.scoring.segments import (
    assign_customer_segments,
)


INPUT_PATH = Path("data/processed/customer_360.parquet")
OUTPUT_PATH = Path("data/processed/customer_scoring.parquet")


def build_segment_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer behavior and value by segment."""
    summary = (
        data.groupby("segment", as_index=True)
        .agg(
            customers=("customer_id", "count"),
            total_spend=("total_spend", "sum"),
            median_spend=("total_spend", "median"),
            median_orders=("order_count", "median"),
            median_recency=("recency_days", "median"),
        )
    )

    summary["customer_share_pct"] = (
        summary["customers"] / len(data) * 100
    )
    summary["spend_share_pct"] = (
        summary["total_spend"] / data["total_spend"].sum() * 100
    )

    column_order = [
        "customers",
        "customer_share_pct",
        "total_spend",
        "spend_share_pct",
        "median_spend",
        "median_orders",
        "median_recency",
    ]

    return summary[column_order].sort_values(
        "total_spend",
        ascending=False,
    )


def main() -> None:
    """Build the final customer scoring dataset."""
    print(f"Loading Customer 360 from {INPUT_PATH}...")
    customer_360 = pd.read_parquet(INPUT_PATH)

    print("Assigning RFM scores...")
    scored_customers = assign_rfm_scores(customer_360)

    print("Assigning customer segments...")
    scored_customers = assign_customer_segments(scored_customers)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_customers.to_parquet(OUTPUT_PATH, index=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)
    segment_summary = build_segment_summary(scored_customers)

    print()
    print("CUSTOMER SCORING SUMMARY")
    print(f"Customers: {len(scored_customers):,}")
    print(
        "RFM score range: "
        f"{scored_customers['rfm_score'].min()}–"
        f"{scored_customers['rfm_score'].max()}"
    )

    print()
    print("SEGMENT SUMMARY")

    with pd.option_context(
        "display.float_format",
        "{:,.2f}".format,
    ):
        print(segment_summary.to_string())

    print()
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("Customer scoring dataset created successfully.")


if __name__ == "__main__":
    main()