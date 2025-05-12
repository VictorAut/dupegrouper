"""Tests for dupegrouper.strategy"""

import pandas as pd

import pytest

from dupegrouper.base import _wrap
from dupegrouper.strategy import DeduplicationStrategy
from dupegrouper.wrappers.dataframes import WrappedPandasDataFrame, WrappedPolarsDataFrame


###########################
# `DeduplicationStrategy` #
###########################


class DummyStrategy(DeduplicationStrategy):
    def dedupe(self, attr: str):
        return self.assign_group_id(attr).unwrap()


def testwith_frame_pandas(df_pandas):
    strategy = DummyStrategy()
    strategy.with_frame(_wrap(df_pandas))

    assert isinstance(strategy.wrapped_df, WrappedPandasDataFrame)


def testwith_frame_polars(df_polars):
    strategy = DummyStrategy()
    strategy.with_frame(_wrap(df_polars))

    assert isinstance(strategy.wrapped_df, WrappedPolarsDataFrame)


# all length 6 arrays
@pytest.mark.parametrize(
    "attribute_array, expected_group_id",
    [
        # standard: matches
        (["Alice", "Bob", "Alice", "Charlie", "Bob", "Charlie"], [1, 2, 1, 4, 2, 4]),
        # Mixed casing: no matches
        (["Alice", "Bob", "alice", "charlie", "Bob", "Charlie"], [1, 2, 3, 4, 2, 6]),
        # int numbers
        ([111, 123, 321, 999, 654, 999], [1, 2, 3, 4, 5, 4]),
        # floats numbers
        ([111.0, 123.0, 321.0, 999.0, 654.0, 999.0], [1, 2, 3, 4, 5, 4]),
        # mixed numbers
        ([111.0, 123.0, 321.0, 999, 654.0, 999.0], [1, 2, 3, 4, 5, 4]),
        # white space: no matches
        (["Alice", "Bob", "Alice     ", "Charlie", "   Bob", "Charlie"], [1, 2, 3, 4, 5, 4]),
    ],
)
def test_assign_group_id(attribute_array, expected_group_id):
    df = pd.DataFrame({"name": attribute_array, "group_id": [1, 2, 3, 4, 5, 6]})

    strategy = DummyStrategy()
    strategy.with_frame(_wrap(df))

    updated_df = strategy.assign_group_id("name").unwrap()

    assert list(updated_df["group_id"]) == expected_group_id


def test_dedupe():
    """In a way, this essentially mimics testing `dupegrouper.strategies.Exact`"""

    df = pd.DataFrame(
        {
            "name": [
                "Alice",
                "Bob",
                "Alice",
                "Charlie",
                "Bob",
                "Charlie",
            ],
            "group_id": [1, 2, 3, 4, 5, 6],
        }
    )

    strategy = DummyStrategy()
    strategy.with_frame(_wrap(df))

    deduped_df = strategy.dedupe("name")  # Uses assign_group_id internally

    expected_groups = [1, 2, 1, 4, 2, 4]
    assert list(deduped_df["group_id"]) == expected_groups
