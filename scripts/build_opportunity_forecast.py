from pathlib import Path

import pandas as pd

from customer_intelligence.forecasting.model import (
    fit_opportunity_forecast_model,
    score_opportunities
)


MODELING_PATH = Path(
    "data/processed/opportunity_modeling.parquet"
)

SCORING_PATH = Path(
    "data/processed/opportunity_scoring.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/opportunity_forecast.parquet"
)

CALIBRATION_START_DATE = pd.Timestamp(
    "2017-10-01"
)


print(
    f"Loading modeling data from "
    f"{MODELING_PATH}..."
)

modeling_data = pd.read_parquet(
    MODELING_PATH
)

print(
    f"Loading scoring data from "
    f"{SCORING_PATH}..."
)

scoring_data = pd.read_parquet(
    SCORING_PATH
)

training_data = modeling_data.loc[
    modeling_data["close_date"]
    < CALIBRATION_START_DATE
].copy()

calibration_data = modeling_data.loc[
    modeling_data["close_date"]
    >= CALIBRATION_START_DATE
].copy()

print("Training opportunity forecast model...")

model = fit_opportunity_forecast_model(
    training_data,
    calibration_data
)

print("Scoring engaging opportunities...")

forecast = score_opportunities(
    scoring_data,
    model
)

forecast_date = modeling_data[
    "close_date"
].max().normalize()

forecast.insert(
    1,
    "forecast_date",
    forecast_date
)

forecast_rank_share = (
    forecast["forecast_rank"]
    / len(forecast)
)

forecast["forecast_priority"] = pd.cut(
    forecast_rank_share,
    bins=[
        0,
        0.10,
        0.30,
        0.60,
        1.00
    ],
    labels=[
        "Prioridad 1 - Estratégica",
        "Prioridad 2 - Alta",
        "Prioridad 3 - Media",
        "Prioridad 4 - Estándar"
    ],
    include_lowest=True
).astype("string")

forecast = (
    forecast
    .sort_values("forecast_rank")
    .reset_index(drop=True)
)

if forecast["opportunity_id"].duplicated().any():
    raise ValueError(
        "Opportunity IDs must remain unique"
    )

if forecast["win_probability"].isna().any():
    raise ValueError(
        "Win probabilities cannot be missing"
    )

if not forecast["win_probability"].between(
    0,
    1
).all():
    raise ValueError(
        "Win probabilities must be between 0 and 1"
    )

if forecast["expected_value"].lt(0).any():
    raise ValueError(
        "Expected values cannot be negative"
    )

if forecast["forecast_rank"].duplicated().any():
    raise ValueError(
        "Forecast ranks must be unique"
    )

if forecast["forecast_priority"].isna().any():
    raise ValueError(
        "Forecast priorities cannot be missing"
    )

if set(forecast["opportunity_id"]) != set(
    scoring_data["opportunity_id"]
):
    raise ValueError(
        "Every scoring opportunity must be included"
    )

priority_summary = (
    forecast
    .groupby(
        "forecast_priority",
        observed=True
    )
    .agg(
        opportunities=(
            "opportunity_id",
            "size"
        ),
        expected_wins=(
            "win_probability",
            "sum"
        ),
        expected_pipeline_value=(
            "expected_value",
            "sum"
        ),
        average_win_probability=(
            "win_probability",
            "mean"
        )
    )
)

priority_summary["opportunity_share_pct"] = (
    priority_summary["opportunities"]
    / len(forecast)
    * 100
)

priority_summary["average_win_probability_pct"] = (
    priority_summary["average_win_probability"]
    * 100
)

priority_summary = priority_summary[
    [
        "opportunities",
        "opportunity_share_pct",
        "expected_wins",
        "expected_pipeline_value",
        "average_win_probability_pct"
    ]
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast.to_parquet(
    OUTPUT_PATH,
    index=False
)

output_size = (
    OUTPUT_PATH.stat().st_size
    / 1024**2
)

print()
print("OPPORTUNITY FORECAST SUMMARY")
print(
    "Forecast date: ",
    forecast_date.date()
)
print(
    "Training rows: ",
    f"{len(training_data):,}"
)
print(
    "Calibration rows: ",
    f"{len(calibration_data):,}"
)
print(
    "Opportunities scored: ",
    f"{len(forecast):,}"
)
print(
    "Expected wins: ",
    f"{forecast['win_probability'].sum():,.1f}"
)
print(
    "Predicted win rate: ",
    f"{forecast['win_probability'].mean():.2%}"
)
print(
    "Expected pipeline value: ",
    f"{forecast['expected_value'].sum():,.2f}"
)
print()
print(priority_summary.round(2))
print()
print(
    f"Saved to {OUTPUT_PATH} "
    f"({output_size:.2f} MB)"
)
print(
    "Opportunity forecast created successfully."
)