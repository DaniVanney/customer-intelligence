from pathlib import Path

import pandas as pd

from customer_intelligence.churn.dataset import build_inactivity_features
from customer_intelligence.churn.model import fit_inactivity_model, score_inactivity_risk


RETAIL_INPUT_PATH = Path("data/interim/online_retail.parquet")
MODEL_INPUT_PATH = Path("data/processed/inactivity_modeling.parquet")
OUTPUT_PATH = Path("data/processed/customer_inactivity_risk.parquet")

SCORING_DATE = "2011-12-10"
OBSERVATION_DAYS = 180

RETENTION_ACTIONS = {
    "Bajo": "Seguimiento regular",
    "Moderado": "Monitorear evolución de actividad",
    "Alto": "Contactar con acción de retención",
    "Crítico": "Intervención prioritaria de retención"
}


def build_risk_summary(data):
    summary = data.groupby("risk_level", observed=True).agg(
        customers=("customer_id", "size"),
        alerts=("is_alert", "sum"),
        average_risk_score=("inactivity_risk_score", "mean"),
        median_recency_days=("recency_days", "median"),
        median_order_count=("order_count", "median")
    )

    summary["customer_share_pct"] = summary["customers"] / len(data) * 100

    summary["average_risk_score_pct"] = summary["average_risk_score"] * 100

    summary = summary[
        [
            "customers",
            "customer_share_pct",
            "alerts",
            "average_risk_score_pct",
            "median_recency_days",
            "median_order_count"
        ]
    ]

    return summary.reindex(["Crítico", "Alto", "Moderado", "Bajo"])


def main():
    print(f"Loading modeling data from {MODEL_INPUT_PATH}...")

    modeling_data = pd.read_parquet(MODEL_INPUT_PATH)

    print(f"Loading retail data from {RETAIL_INPUT_PATH}...")

    retail_data = pd.read_parquet(RETAIL_INPUT_PATH)

    print("Training final inactivity model...")

    model = fit_inactivity_model(modeling_data)

    print(f"Building current customer features at {SCORING_DATE}...")

    scoring_features = build_inactivity_features(
        retail_data,
        cutoff_date=SCORING_DATE,
        observation_days=OBSERVATION_DAYS
    )

    print("Assigning inactivity risk scores...")

    risk_data = score_inactivity_risk(scoring_features, model)

    risk_data["retention_action"] = risk_data["risk_level"].map(
        RETENTION_ACTIONS
    )

    risk_data = risk_data.sort_values(
        ["inactivity_risk_score", "customer_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    risk_data.insert(
        1,
        "risk_rank",
        range(1, len(risk_data) + 1)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    risk_data.to_parquet(OUTPUT_PATH, index=False)

    risk_summary = build_risk_summary(risk_data)

    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)

    print()
    print("CUSTOMER INACTIVITY RISK SUMMARY")
    print(f"Scoring date: {SCORING_DATE}")
    print(f"Training rows: {len(modeling_data):,}")
    print(f"Training snapshots: {modeling_data['snapshot_date'].nunique()}")
    print(f"Eligible customers scored: {len(risk_data):,}")
    print(f"Alerts generated: {risk_data['is_alert'].sum():,}")
    print(f"Alert rate: {risk_data['is_alert'].mean():.2%}")
    print()

    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(risk_summary.to_string())

    print()
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("Customer inactivity risk dataset created successfully.")


if __name__ == "__main__":
    main()