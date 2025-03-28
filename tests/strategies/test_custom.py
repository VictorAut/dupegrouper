import pytest
import pandas as pd
import polars as pl
from dupegrouper.strategies.custom import Custom


# Custom callable function
def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    my_map = {}
    for irow, _ in df.iterrows():
        left: str = df.at[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right: str = df.at[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break

    print(my_map)
    return my_map


def test_custom_dedupe(df_pandas):

    deduper = Custom(my_func, "address", match_str="Navarra")
    deduper._set_df(df_pandas)

    updated_df = deduper.dedupe()

    expected_group_ids = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13] 

    assert list(updated_df["group_id"]) == expected_group_ids