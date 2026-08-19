import numpy as np

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


MODEL_FEATURES = ["recency_days", "order_count", "total_spend", "tenure_days", "unique_products"]

MODEL_LOG_FEATURES = ["recency_days", "order_count", "total_spend", "unique_products"]

NUMBER_OF_CLUSTERS = 4

SEGMENT_ACTIONS = {
    "Valor intermedio en seguimiento": "Seguimiento y reactivación selectiva",
    "Alto valor consolidado": "Retención prioritaria y desarrollo de valor",
    "Baja actividad prolongada": "Reactivación de bajo costo o despriorización",
    "Desarrollo reciente": "Impulsar una segunda compra y ampliar la relación"
}


def validate_segmentation_input(data):
    required_columns = {"customer_id", *MODEL_FEATURES}

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if data["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be unique")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Segmentation values cannot be missing")

    if data[MODEL_FEATURES].lt(0).any().any():
        raise ValueError("Segmentation features cannot be negative")

    if len(data) < NUMBER_OF_CLUSTERS:
        raise ValueError("There must be at least four customers")


def build_segmentation_pipeline():
    log_pipeline = Pipeline(steps=[("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[("log_features", log_pipeline, MODEL_LOG_FEATURES), ("scale_only", StandardScaler(), ["tenure_days"])], remainder="drop", verbose_feature_names_out=False)

    segmentation_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("kmeans", KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=20))])

    return segmentation_pipeline


def identify_cluster_names(data):
    cluster_profiles = data.groupby("cluster_id").agg(
        median_total_spend=("total_spend", "median"),
        median_recency_days=("recency_days", "median"),
        median_tenure_days=("tenure_days", "median")
    )

    available_clusters = set(cluster_profiles.index)

    high_value_cluster = cluster_profiles["median_total_spend"].idxmax()

    available_clusters.remove(high_value_cluster)

    low_activity_cluster = cluster_profiles.loc[sorted(available_clusters), "median_recency_days"].idxmax()

    available_clusters.remove(low_activity_cluster)

    recent_development_cluster = cluster_profiles.loc[sorted(available_clusters), "median_tenure_days"].idxmin()

    available_clusters.remove(recent_development_cluster)

    intermediate_value_cluster = available_clusters.pop()

    cluster_names = {
        high_value_cluster: "Alto valor consolidado",
        low_activity_cluster: "Baja actividad prolongada",
        recent_development_cluster: "Desarrollo reciente",
        intermediate_value_cluster: "Valor intermedio en seguimiento"
    }

    return cluster_names


def fit_customer_segmentation(data):
    validate_segmentation_input(data)

    model = build_segmentation_pipeline()

    result = data.copy()

    result["cluster_id"] = model.fit_predict(result[MODEL_FEATURES])

    cluster_names = identify_cluster_names(result)

    result["ml_segment"] = result["cluster_id"].map(cluster_names)

    result["recommended_action"] = result["ml_segment"].map(SEGMENT_ACTIONS)

    return result, model