from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error
)

from customer_intelligence.forecasting.model import (
    fit_opportunity_forecast_model,
    score_opportunities
)


MODELING_PATH = Path(
    "data/processed/opportunity_modeling.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/opportunity_forecast_validation.parquet"
)

VALIDATION_START_DATE = pd.Timestamp(
    "2017-07-01"
)

TEST_START_DATE = pd.Timestamp(
    "2017-10-01"
)

FORECAST_DATE = (
    TEST_START_DATE
    - pd.Timedelta(days=1)
)


print(
    f"Loading modeling data from "
    f"{MODELING_PATH}..."
)

modeling_data = pd.read_parquet(
    MODELING_PATH
)

training_data = modeling_data.loc[
    modeling_data["close_date"]
    < VALIDATION_START_DATE
].copy()

calibration_data = modeling_data.loc[
    modeling_data["close_date"].between(
        VALIDATION_START_DATE,
        TEST_START_DATE,
        inclusive="left"
    )
].copy()

test_data = modeling_data.loc[
    modeling_data["close_date"]
    >= TEST_START_DATE
].copy()

print("Training temporal validation model...")

model = fit_opportunity_forecast_model(
    training_data,
    calibration_data
)

print("Generating test-period forecast...")

validation_forecast = score_opportunities(
    test_data,
    model
)

validation_forecast.insert(
    1,
    "forecast_date",
    FORECAST_DATE
)

validation_forecast["validation_period"] = (
    validation_forecast["close_date"]
    .dt.to_period("Q")
    .astype("string")
)

validation_forecast["forecast_error"] = (
    validation_forecast["expected_value"]
    - validation_forecast["close_value"]
)

validation_forecast["absolute_forecast_error"] = (
    validation_forecast["forecast_error"]
    .abs()
)

if validation_forecast["opportunity_id"].duplicated().any():
    raise ValueError(
        "Opportunity IDs must remain unique"
    )

if validation_forecast[
    [
        "win_probability",
        "expected_value",
        "forecast_error",
        "absolute_forecast_error"
    ]
].isna().any().any():
    raise ValueError(
        "Validation forecast values cannot be missing"
    )

if not validation_forecast[
    "win_probability"
].between(0, 1).all():
    raise ValueError(
        "Win probabilities must be between 0 and 1"
    )

if validation_forecast["expected_value"].lt(0).any():
    raise ValueError(
        "Expected values cannot be negative"
    )

if set(
    validation_forecast["opportunity_id"]
) != set(
    test_data["opportunity_id"]
):
    raise ValueError(
        "Every test opportunity must be included"
    )

actual_wins = validation_forecast[
    "is_won"
].astype("int64")

actual_values = validation_forecast[
    "close_value"
]

predicted_pipeline_value = validation_forecast[
    "expected_value"
].sum()

actual_pipeline_value = actual_values.sum()

aggregate_error_pct = (
    predicted_pipeline_value
    / actual_pipeline_value
    - 1
) * 100

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

validation_forecast.to_parquet(
    OUTPUT_PATH,
    index=False
)

output_size = (
    OUTPUT_PATH.stat().st_size
    / 1024**2
)

print()
print("OPPORTUNITY FORECAST VALIDATION")
print(
    "Forecast date: ",
    FORECAST_DATE.date()
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
    "Test rows: ",
    f"{len(test_data):,}"
)
print()
print(
    "ROC AUC: ",
    f"{roc_auc_score(actual_wins, validation_forecast['win_probability']):.3f}"
)
print(
    "Average precision: ",
    f"{average_precision_score(actual_wins, validation_forecast['win_probability']):.3f}"
)
print(
    "Brier score: ",
    f"{brier_score_loss(actual_wins, validation_forecast['win_probability']):.3f}"
)
print(
    "Predicted pipeline value: ",
    f"{predicted_pipeline_value:,.2f}"
)
print(
    "Actual pipeline value: ",
    f"{actual_pipeline_value:,.2f}"
)
print(
    "Aggregate error: ",
    f"{aggregate_error_pct:.2f}%"
)
print(
    "Opportunity MAE: ",
    f"{mean_absolute_error(actual_values, validation_forecast['expected_value']):,.2f}"
)
print(
    "Opportunity RMSE: ",
    f"{root_mean_squared_error(actual_values, validation_forecast['expected_value']):,.2f}"
)
print()
print(
    f"Saved to {OUTPUT_PATH} "
    f"({output_size:.2f} MB)"
)
print(
    "Opportunity forecast validation dataset "
    "created successfully."
)