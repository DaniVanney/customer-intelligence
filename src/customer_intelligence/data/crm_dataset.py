import pandas as pd


def find_unmatched_values(
    data: pd.DataFrame,
    reference: pd.DataFrame,
    column: str,
) -> list[str]:
    """Return non-missing values absent from a reference table."""
    values = data[column].dropna()
    unmatched = values.loc[~values.isin(reference[column])].unique()

    return sorted(unmatched.tolist())


def validate_relationship(
    data: pd.DataFrame,
    reference: pd.DataFrame,
    column: str,
) -> None:
    """Raise an error when a known value has no reference-table match."""
    unmatched = find_unmatched_values(data, reference, column)

    if unmatched:
        raise ValueError(
            f"Unmatched values in column '{column}': {unmatched}"
        )


def build_opportunity_dataset(
    pipeline: pd.DataFrame,
    accounts: pd.DataFrame,
    products: pd.DataFrame,
    sales_teams: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich opportunities with product, account and sales-team data."""
    validate_relationship(pipeline, products, "product")
    validate_relationship(pipeline, sales_teams, "sales_agent")
    validate_relationship(pipeline, accounts, "account")

    result = pipeline.merge(
        products,
        on="product",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        sales_teams,
        on="sales_agent",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        accounts,
        on="account",
        how="left",
        validate="many_to_one",
    )

    return result.reset_index(drop=True)