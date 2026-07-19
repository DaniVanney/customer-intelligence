from pathlib import Path

import pandas as pd

from customer_intelligence.data.crm import prepare_sales_pipeline
from customer_intelligence.data.crm_dataset import (
    build_opportunity_dataset,
)
from customer_intelligence.data.crm_references import (
    prepare_accounts,
    prepare_products,
    prepare_sales_teams,
)


RAW_DIR = Path("data/raw/crm_sales")
OUTPUT_DIR = Path("data/interim/crm")


def save_table(data: pd.DataFrame, filename: str) -> None:
    """Save an interim table as Parquet."""
    output_path = OUTPUT_DIR / filename
    data.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"Saved {output_path} ({size_mb:.2f} MB)")


def main() -> None:
    """Build the CRM interim datasets."""
    print("Loading raw CRM tables...")

    raw_pipeline = pd.read_csv(RAW_DIR / "sales_pipeline.csv")
    raw_accounts = pd.read_csv(RAW_DIR / "accounts.csv")
    raw_products = pd.read_csv(RAW_DIR / "products.csv")
    raw_sales_teams = pd.read_csv(RAW_DIR / "sales_teams.csv")

    print("Preparing CRM tables...")

    pipeline = prepare_sales_pipeline(raw_pipeline)
    accounts = prepare_accounts(raw_accounts)
    products = prepare_products(raw_products)
    sales_teams = prepare_sales_teams(raw_sales_teams)

    opportunities = build_opportunity_dataset(
        pipeline,
        accounts,
        products,
        sales_teams,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_table(pipeline, "sales_pipeline.parquet")
    save_table(accounts, "accounts.parquet")
    save_table(products, "products.parquet")
    save_table(sales_teams, "sales_teams.parquet")
    save_table(opportunities, "opportunities_enriched.parquet")

    closed_count = int(pipeline["is_closed"].sum())
    won_count = int(pipeline["is_won"].fillna(False).sum())
    engaging_count = int(
        pipeline["deal_stage"].eq("Engaging").sum()
    )

    print()
    print("CRM INTERIM SUMMARY")
    print(f"Pipeline rows: {len(pipeline):,}")
    print(f"Enriched opportunity rows: {len(opportunities):,}")
    print(f"Closed opportunities: {closed_count:,}")
    print(f"Won opportunities: {won_count:,}")
    print(f"Engaging opportunities: {engaging_count:,}")
    print("CRM interim datasets created successfully.")


if __name__ == "__main__":
    main()