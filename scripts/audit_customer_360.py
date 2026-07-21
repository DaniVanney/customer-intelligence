from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/customer_360.parquet")

METRIC_COLUMNS = [
    "recency_days",
    "tenure_days",
    "order_count",
    "total_spend",
    "total_items",
    "unique_products",
    "average_order_value",
]


def percentage(count: int, total: int) -> float:
    """Return a count as a percentage of a total."""
    return count / total * 100


def main() -> None:
    """Audit the Customer 360 behavioral distributions."""
    data = pd.read_parquet(DATA_PATH)
    customer_count = len(data)

    print("CUSTOMER 360 — BEHAVIORAL AUDIT")
    print(f"Customers: {customer_count:,}")
    print(f"Snapshot: {data['snapshot_date'].iloc[0].date()}")

    print()
    print("METRIC DISTRIBUTIONS")

    summary = data[METRIC_COLUMNS].describe(
        percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    )

    with pd.option_context(
        "display.float_format",
        "{:,.2f}".format,
    ):
        print(summary.T.to_string())

    single_order_count = int(data["order_count"].eq(1).sum())
    inactive_90_count = int(data["recency_days"].gt(90).sum())
    inactive_180_count = int(data["recency_days"].gt(180).sum())

    spend = data["total_spend"].sort_values(ascending=False)
    top_1_count = max(1, round(customer_count * 0.01))
    top_5_count = max(1, round(customer_count * 0.05))

    top_1_share = spend.head(top_1_count).sum() / spend.sum()
    top_5_share = spend.head(top_5_count).sum() / spend.sum()

    print()
    print("CUSTOMER BEHAVIOR")
    print(
        f"One-order customers: {single_order_count:,} "
        f"({percentage(single_order_count, customer_count):.2f}%)"
    )
    print(
        f"Recency over 90 days: {inactive_90_count:,} "
        f"({percentage(inactive_90_count, customer_count):.2f}%)"
    )
    print(
        f"Recency over 180 days: {inactive_180_count:,} "
        f"({percentage(inactive_180_count, customer_count):.2f}%)"
    )

    print()
    print("SPEND CONCENTRATION")
    print(f"Top 1% customer share: {top_1_share:.2%}")
    print(f"Top 5% customer share: {top_5_share:.2%}")


if __name__ == "__main__":
    main()