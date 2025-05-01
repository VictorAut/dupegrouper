import pytest

from dupegrouper.base import _dispatch_dataframe
from dupegrouper.strategies.fuzzy import Fuzzy


def do_fuzzy(df, fuzzy_params, group_id):
    fuzzy = Fuzzy(**fuzzy_params)
    fuzzy._set_df(_dispatch_dataframe(df))

    updated_df = fuzzy.dedupe("address")

    assert list(updated_df["group_id"]) == group_id


fuzzy_parametrize_data = [
    # i.e. no deduping
    ({"tolerance": 0}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive deduping
    ({"tolerance": 0.05}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.15}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.25}, [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.35}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.45}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.55}, [1, 2, 3, 3, 5, 5, 3, 2, 1, 1, 5, 1, 13]),
    ({"tolerance": 0.65}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 2, 12, 13]),
    ({"tolerance": 0.75}, [1, 2, 3, 3, 3, 3, 3, 3, 1, 1, 3, 12, 3]),
    ({"tolerance": 0.85}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 1]),
    ({"tolerance": 0.95}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
]

# i.e. pandas


@pytest.mark.parametrize("fuzzy_params, expected_group_id", fuzzy_parametrize_data)
def test_fuzzy_dedupe_pandas(fuzzy_params, expected_group_id, df_pandas):
    do_fuzzy(df_pandas, fuzzy_params, expected_group_id)


# i.e. polars


@pytest.mark.parametrize("fuzzy_params, expected_group_id", fuzzy_parametrize_data)
def test_fuzzy_dedupe_polars(fuzzy_params, expected_group_id, df_polars):
    do_fuzzy(df_polars, fuzzy_params, expected_group_id)
