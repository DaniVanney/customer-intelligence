import pandas as pd
import pytest

from customer_intelligence.scoring.clustering import fit_customer_segmentation


def create_customer_data():
    customer_data = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "recency_days": [5, 10, 160, 200, 650, 700, 15, 25],
            "order_count": [25, 20, 5, 4, 1, 1, 2, 2],
            "total_spend": [15000.0, 10000.0, 1500.0, 1200.0, 150.0, 100.0, 300.0, 250.0],
            "tenure_days": [720, 700, 650, 600, 500, 600, 60, 30],
            "unique_products": [250, 200, 70, 60, 8, 5, 15, 10]
        }
    )

    return customer_data


def test_fit_customer_segmentation_assigns_every_customer():
    data = create_customer_data()

    result, model = fit_customer_segmentation(data)

    assert len(result) == len(data)
    assert result["customer_id"].tolist() == data["customer_id"].tolist()
    assert result["customer_id"].is_unique
    assert result["cluster_id"].nunique() == 4
    assert result["cluster_id"].notna().all()
    assert result["ml_segment"].notna().all()
    assert result["recommended_action"].notna().all()
    assert set(model.named_steps) == {"preprocessor", "kmeans"}

def test_fit_customer_segmentation_identifies_business_profiles():
    data = create_customer_data()

    result, model = fit_customer_segmentation(data)

    segments = result.set_index("customer_id")["ml_segment"]

    assert segments["A"] == "Alto valor consolidado"
    assert segments["B"] == "Alto valor consolidado"
    assert segments["C"] == "Valor intermedio en seguimiento"
    assert segments["D"] == "Valor intermedio en seguimiento"
    assert segments["E"] == "Baja actividad prolongada"
    assert segments["F"] == "Baja actividad prolongada"
    assert segments["G"] == "Desarrollo reciente"
    assert segments["H"] == "Desarrollo reciente"


def test_fit_customer_segmentation_is_reproducible():
    data = create_customer_data()

    first_result, first_model = fit_customer_segmentation(data)

    second_result, second_model = fit_customer_segmentation(data)

    assert first_result["cluster_id"].tolist() == second_result["cluster_id"].tolist()
    assert first_result["ml_segment"].tolist() == second_result["ml_segment"].tolist()

def test_fit_customer_segmentation_rejects_missing_columns():
    data = create_customer_data().drop(columns=["total_spend"])

    with pytest.raises(ValueError, match="Missing required columns"):
        fit_customer_segmentation(data)


def test_fit_customer_segmentation_rejects_duplicate_customers():
    data = create_customer_data()

    data.loc[1, "customer_id"] = data.loc[0, "customer_id"]

    with pytest.raises(ValueError, match="Customer IDs must be unique"):
        fit_customer_segmentation(data)


def test_fit_customer_segmentation_rejects_missing_values():
    data = create_customer_data()

    data.loc[0, "recency_days"] = None

    with pytest.raises(ValueError, match="Segmentation values cannot be missing"):
        fit_customer_segmentation(data)


def test_fit_customer_segmentation_rejects_negative_values():
    data = create_customer_data()

    data.loc[0, "total_spend"] = -100.0

    with pytest.raises(ValueError, match="Segmentation features cannot be negative"):
        fit_customer_segmentation(data)