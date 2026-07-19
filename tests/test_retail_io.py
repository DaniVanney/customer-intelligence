from pathlib import Path

import pandas as pd

from customer_intelligence.data.retail_io import (
    load_retail_excel,
    save_parquet,
)


def test_load_retail_excel_combines_sheets(tmp_path: Path) -> None:
    input_path = tmp_path / "retail.xlsx"

    with pd.ExcelWriter(input_path) as writer:
        pd.DataFrame({"value": [1, 2]}).to_excel(
            writer,
            sheet_name="First",
            index=False,
        )
        pd.DataFrame({"value": [3]}).to_excel(
            writer,
            sheet_name="Second",
            index=False,
        )

    result = load_retail_excel(input_path)

    assert result["value"].tolist() == [1, 2, 3]


def test_save_parquet_creates_readable_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "retail.parquet"
    expected = pd.DataFrame(
        {
            "invoice_id": ["100", "101"],
            "line_total": [10.0, 20.0],
        }
    )

    save_parquet(expected, output_path)
    result = pd.read_parquet(output_path)

    assert output_path.exists()
    pd.testing.assert_frame_equal(result, expected)