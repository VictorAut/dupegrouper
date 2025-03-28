"""Tests for dupegrouper.strategy"""

import pandas as pd

import pytest

from dupegrouper.strategy import DeduplicationStrategy, _DataFrameDispatcher
from dupegrouper.frames.methods import PandasMethods, PolarsMethods


#############################
# TEST _DataFrameDispatcher #
#############################


def test_dataframe_dispatcher_pandas(df_pandas):
    dispatcher = _DataFrameDispatcher(df_pandas)
    assert isinstance(dispatcher.frame_methods, PandasMethods)


def test_dataframe_dispatcher_polars(df_polars):
    dispatcher = _DataFrameDispatcher(df_polars)
    assert isinstance(dispatcher.frame_methods, PolarsMethods)


def test_dataframe_dispatcher_unsupported():
    class FakeDataFrame:
        pass

    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        _DataFrameDispatcher(FakeDataFrame())


###########################
# `DeduplicationStrategy` #
###########################


class DummyStrategy(DeduplicationStrategy):
    def dedupe(self, attr: str):
        return self.assign_group_id(attr).frame


def test_set_df_pandas(df_pandas):
    strategy = DummyStrategy()
    strategy._set_df(df_pandas)

    assert isinstance(strategy.frame_methods, PandasMethods)


def test_set_df_polars(df_polars):
    strategy = DummyStrategy()
    strategy._set_df(df_polars)

    assert isinstance(strategy.frame_methods, PolarsMethods)


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
    strategy._set_df(df)

    updated_df = strategy.assign_group_id("name").frame

    assert list(updated_df["group_id"]) == expected_group_id


def test_dedupe():
    """In a way, this essentially mimics testing `dupegrouper.strategies.Exact`"""

    df = pd.DataFrame({"name": ["Alice", "Bob", "Alice", "Charlie", "Bob", "Charlie"], "group_id": [1, 2, 3, 4, 5, 6]})

    strategy = DummyStrategy()
    strategy._set_df(df)

    deduped_df = strategy.dedupe("name")  # Uses assign_group_id internally

    expected_groups = [1, 2, 1, 4, 2, 4]
    assert list(deduped_df["group_id"]) == expected_groups


# deduplication.strategies.Exact is wrapper of `assign_group_id` so above is valid
