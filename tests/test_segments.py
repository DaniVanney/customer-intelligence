import pandas as pd

import pytest

from customer_intelligence.scoring.segments import (
    assign_customer_segments,
)


def test_assign_customer_segments_covers_business_profiles():
    data = pd.DataFrame(
        {
            "customer_id": [
                "strategic",
                "recent",
                "established",
                "high-value",
                "developing",
                "at-risk",
                "inactive",
                "general",
            ],
            "recency_score": [5, 5, 3, 3, 5, 1, 1, 3],
            "frequency_score": [5, 1, 5, 2, 3, 5, 1, 2],
            "monetary_score": [5, 2, 2, 5, 2, 4, 1, 2],
        }
    )

    result = assign_customer_segments(data)
    segments = result.set_index("customer_id")["segment"]

    assert segments["strategic"] == "Valor estratégico"
    assert segments["recent"] == "Incorporación reciente"
    assert segments["established"] == "Lealtad consolidada"
    assert segments["high-value"] == "Alto valor"
    assert segments["developing"] == "Potencial de desarrollo"
    assert segments["at-risk"] == "Riesgo de inactividad"
    assert segments["inactive"] == "Inactividad prolongada"
    assert segments["general"] == "Base general"

def test_assign_customer_segments_rejects_missing_columns():
    incomplete_data = pd.DataFrame(
        {
            "customer_id": ["A"],
            "recency_score": [5],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        assign_customer_segments(incomplete_data)


def test_assign_customer_segments_rejects_invalid_scores():
    data = pd.DataFrame(
        {
            "customer_id": ["A"],
            "recency_score": [0],
            "frequency_score": [3],
            "monetary_score": [6],
        }
    )

    with pytest.raises(
        ValueError,
        match="Scores must be between 1 and 5",
    ):
        assign_customer_segments(data)