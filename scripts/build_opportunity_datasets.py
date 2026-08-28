from pathlib import Path

import pandas as pd

from customer_intelligence.forecasting.dataset import build_opportunity_datasets


INPUT_PATH = Path("data/interim/crm/opportunities_enriched.parquet")

MODELING_OUTPUT_PATH = Path(
    "data/processed/opportunity_modeling.parquet"
)

SCORING_OUTPUT_PATH = Path(
    "data/processed/opportunity_scoring.parquet"
)


def save_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """Save a processed opportunity dataset as Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024**2)

    print(f"Saved to {output_path} ({size_mb:.2f} MB)")


def main() -> None:
    """Build the opportunity modeling and scoring datasets."""
    print(f"Loading enriched opportunities from {INPUT_PATH}...")

    opportunities = pd.read_parquet(INPUT_PATH)

    print("Building opportunity datasets...")

    modeling_data, scoring_data = build_opportunity_datasets(
        opportunities
    )

    save_dataset(modeling_data, MODELING_OUTPUT_PATH)
    save_dataset(scoring_data, SCORING_OUTPUT_PATH)

    closed_by_quarter = pd.crosstab(
        modeling_data["close_date"].dt.to_period("Q"),
        modeling_data["deal_stage"]
    )

    won_opportunities = int(modeling_data["is_won"].sum())
    lost_opportunities = len(modeling_data) - won_opportunities
    win_rate = modeling_data["is_won"].mean() * 100

    print()
    print("OPPORTUNITY MODELING DATASET")
    print(f"Closed opportunities: {len(modeling_data):,}")
    print(f"Won opportunities: {won_opportunities:,}")
    print(f"Lost opportunities: {lost_opportunities:,}")
    print(f"Historical win rate: {win_rate:.2f}%")
    print()

    print("CLOSED OPPORTUNITIES BY QUARTER")
    print(closed_by_quarter.to_string())
    print()

    print("OPPORTUNITY SCORING DATASET")
    print(f"Engaging opportunities: {len(scoring_data):,}")
    print(
        "Missing engagement dates: ",
        scoring_data["engage_date"].isna().sum()
    )
    print("Opportunity datasets created successfully.")


if __name__ == "__main__":
    main()