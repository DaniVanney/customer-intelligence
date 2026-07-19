import pandas as pd


ACCOUNT_COLUMNS = frozenset(
    {
        "account",
        "sector",
        "year_established",
        "revenue",
        "employees",
        "office_location",
        "subsidiary_of",
    }
)

PRODUCT_COLUMNS = frozenset(
    {
        "product",
        "series",
        "sales_price",
    }
)

SALES_TEAM_COLUMNS = frozenset(
    {
        "sales_agent",
        "manager",
        "regional_office",
    }
)


def validate_reference_table(
    data: pd.DataFrame,
    expected_columns: frozenset[str],
    key: str,
    table_name: str,
) -> None:
    """Validate a CRM reference-table schema and primary key."""
    actual_columns = set(data.columns)
    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Invalid {table_name} schema. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )

    if data[key].isna().any():
        raise ValueError(f"{table_name} contains missing values in {key}.")

    if data[key].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate values in {key}.")


def prepare_accounts(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare the CRM account reference table."""
    validate_reference_table(
        data,
        ACCOUNT_COLUMNS,
        key="account",
        table_name="accounts",
    )
    result = data.copy()

    string_columns = [
        "account",
        "sector",
        "office_location",
        "subsidiary_of",
    ]
    for column in string_columns:
        result[column] = result[column].astype("string")

    result["year_established"] = pd.to_numeric(
        result["year_established"], errors="raise"
    ).astype("int64")
    result["revenue"] = pd.to_numeric(
        result["revenue"], errors="raise"
    ).astype("float64")
    result["employees"] = pd.to_numeric(
        result["employees"], errors="raise"
    ).astype("int64")

    return result


def prepare_products(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare the CRM product reference table."""
    validate_reference_table(
        data,
        PRODUCT_COLUMNS,
        key="product",
        table_name="products",
    )
    result = data.copy()

    result["product"] = result["product"].astype("string")
    result["series"] = result["series"].astype("string")
    result["sales_price"] = pd.to_numeric(
        result["sales_price"], errors="raise"
    ).astype("float64")

    return result


def prepare_sales_teams(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare the CRM sales-team reference table."""
    validate_reference_table(
        data,
        SALES_TEAM_COLUMNS,
        key="sales_agent",
        table_name="sales_teams",
    )
    result = data.copy()

    for column in SALES_TEAM_COLUMNS:
        result[column] = result[column].astype("string")

    return result