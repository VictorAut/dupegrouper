"""Tests for dupegrouper.base"""

import importlib
import os
from unittest.mock import ANY, Mock, patch

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row
import pytest

from dupegrouper.base import (
    DupeGrouper,
    DeduplicationStrategy,
    StrategyTypeError,
    _StrategyManager,
    _wrap,
)

import dupegrouper.definitions
from dupegrouper.strategies import Exact, Fuzzy, TfIdf
from dupegrouper.wrappers import WrappedDataFrame
import dupegrouper.wrappers.dataframes
from dupegrouper.wrappers.dataframes import (
    WrappedPandasDataFrame,
    WrappedPolarsDataFrame,
    WrappedSparkDataFrame,
    WrappedSparkRows,
)


###############
#  TEST _wrap #
###############

DATAFRAME_TYPES = {
    pd.DataFrame: WrappedPandasDataFrame,
    pl.DataFrame: WrappedPolarsDataFrame,
    SparkDataFrame: WrappedSparkDataFrame,
    list[Row]: WrappedSparkRows,
}


def test_wrap_dataframe(dataframe):
    df, _, id = dataframe

    expected_type = DATAFRAME_TYPES.get(type(df))

    df_wrapped: WrappedDataFrame = _wrap(df, id)

    assert isinstance(df_wrapped, expected_type)


def test_dataframe_dispatcher_unsupported():
    class FakeDataFrame:
        pass

    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        _wrap(FakeDataFrame())


######################
#  TEST set group_id #
######################


def reload():
    importlib.reload(dupegrouper.definitions)  # reset constant
    importlib.reload(dupegrouper.wrappers.dataframes._pandas)
    importlib.reload(dupegrouper.wrappers.dataframes._polars)
    importlib.reload(dupegrouper.wrappers.dataframes._spark)


@pytest.mark.parametrize(
    "env_var_value, expected_value",
    [
        # i.e. the default
        ("group_id", "group_id"),
        # null override to default, simulates unset
        (None, "group_id"),
        # arbitrary: different value
        ("beep_boop_id", "beep_boop_id"),
        # arbitrary: supported (but bad!) column naming with whitespace
        ("bad group id", "bad group id"),
    ],
    ids=["default", "null", "default-override", "default-override-bad-format"],
)
def test_group_id_env_var(env_var_value, expected_value, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    if env_var_value:
        os.environ["GROUP_ID"] = env_var_value
    else:
        os.environ.pop("GROUP_ID", None)  # remove it if exists

    reload()

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        assert expected_value not in df.columns  # no change
    elif isinstance(df, WrappedSparkRows):
        for row in df.unwrap():
            print(row.asDict().keys())
            assert expected_value in row.asDict().keys()
    else:
        assert expected_value in df.columns

    # clean up
    os.environ["GROUP_ID"] = "group_id"

    reload()


##############################################
#  TEST _StrategyManager + StrategyTypeError #
##############################################


class DummyClass:
    pass


DEFAULT_ERROR_MSG = "Input is not valid"
CLASS_ERROR_MSG = "Input class is not valid: must be an instance of `DeduplicationStrategy`"
TUPLE_ERROR_MSG = "Input tuple is not valid: must be a length 2 [callable, dict]"
DICT_ERROR_MSG = "Input dict is not valid: items must be a list of `DeduplicationStrategy` or tuples"


@pytest.mark.parametrize(
    "strategy, expected_to_pass, base_msg",
    [
        # correct base inputs
        (Mock(spec=DeduplicationStrategy), True, None),
        ((lambda x: x, {"key": "value"}), True, None),
        (
            {
                "address": [
                    Mock(spec=DeduplicationStrategy),
                    (lambda x: x, {"key": "value"}),
                ],
                "email": [
                    Mock(spec=DeduplicationStrategy),
                    Mock(spec=DeduplicationStrategy),
                ],
            },
            True,
            None,
        ),
        # incorrect inputs
        (DummyClass, False, CLASS_ERROR_MSG),
        (lambda x: x, False, DEFAULT_ERROR_MSG),
        ((lambda x: x, [1, 2, 3]), False, TUPLE_ERROR_MSG),
        (("foo",), False, TUPLE_ERROR_MSG),
        (["bar", "baz"], False, DEFAULT_ERROR_MSG),
        ("foobar", False, DEFAULT_ERROR_MSG),
        (
            {
                "address": [DummyClass()],
                "email": [
                    "random string",
                    ("tuple too short",),
                ],
            },
            False,
            DICT_ERROR_MSG,
        ),
    ],
    ids=[
        "valid dedupe class",
        "valid callable",
        "valid dict",
        "invalid class",
        "invalid callable not in tuple",
        "invalid callable positional args",
        "invalid tuple",
        "invalid list",
        "invalid str",
        "invalid dict"
    ],
)
def test_strategy_manager_validate_addition_strategy(strategy, expected_to_pass, base_msg):
    """validates that the input 'strtagey' is legit, against `StrategyTypeError`"""
    manager = _StrategyManager()
    if expected_to_pass:
        if isinstance(strategy, dict):
            for k, value in strategy.items():
                for v in value:
                    manager.add(k, v)
                    assert (k in manager.get()) is expected_to_pass
        else:
            manager.add("default", strategy)
            assert ("default" in manager.get()) is expected_to_pass
    else:
        with pytest.raises(StrategyTypeError) as e:
            manager.add("default", strategy)
            assert base_msg in str(e)


def test_strategy_manager_reset():
    manager = _StrategyManager()
    strategy = Mock(spec=DeduplicationStrategy)
    manager.add("name", strategy)
    manager.reset()
    assert manager.get() == {}


##################################
# TEST DupeGrouper - public API! #
##################################


def test_dupegrouper_df_attribute_pandas(df_pandas):
    grouper = DupeGrouper(df_pandas)
    assert isinstance(grouper.df, pd.DataFrame)
    assert "group_id" in grouper.df.columns


def test_dupegrouper_df_attribute_polars(df_polars):
    grouper = DupeGrouper(df_polars)
    assert isinstance(grouper.df, pl.DataFrame)
    assert "group_id" in grouper.df.columns


def test_dupegrouper_add_strategy(df_pandas):
    grouper = DupeGrouper(df_pandas)
    mock_strategy = Mock(spec=DeduplicationStrategy)
    grouper.add_strategy(mock_strategy)
    assert "default" in grouper._strategy_manager.get()


def my_dummy_func():
    pass


def patch_helper_reset(grouper: DupeGrouper):
    with patch.object(grouper, "_dedupe") as mock_dedupe, patch.object(
        grouper._strategy_manager, "reset"
    ) as mock_reset:

        mock_dedupe.side_effect = mock_reset

        grouper.dedupe("address")

        mock_dedupe.assert_called_once_with("address", ANY)

        grouper._strategy_manager = _StrategyManager()
        print(grouper.strategies)

    assert not grouper.strategies


def test_dupegrouper_strategies_attribute_inline(df_pandas):
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(Mock(spec=Exact))
    grouper.add_strategy(Mock(spec=Fuzzy))
    grouper.add_strategy((my_dummy_func, {"str": "random"}))

    assert grouper.strategies == tuple(["Exact", "Fuzzy", "my_dummy_func"])

    patch_helper_reset(grouper)


def test_dupegrouper_strategies_attribute_dict(df_pandas):
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(
        {
            "address": [
                Mock(spec=Exact),
                (my_dummy_func, {"key": "value"}),
            ],
            "email": [
                Mock(spec=Exact),
                Mock(spec=Fuzzy),
            ],
        }
    )

    assert grouper.strategies == dict({"address": ("Exact", "my_dummy_func"), "email": ("Exact", "Fuzzy")})

    patch_helper_reset(grouper)


def test_dupegrouper_add_strategy_equal_execution(df_pandas):
    """strategies can be added in various ways

    This tests that given the differing addition mechanisms, the output is the
    same"""

    expected_group_ids = [1, 2, 3, 3, 2, 6, 2, 2, 1, 1, 11, 12, 13]

    dg_inline = DupeGrouper(df_pandas)
    dg_inline.add_strategy(Exact())
    dg_inline.add_strategy(Fuzzy(tolerance=0.3))
    dg_inline.add_strategy(TfIdf(tolerance=0.7, ngram=3, topn=3))

    dg_inline.dedupe("address")

    dg_inline.add_strategy(Exact())

    dg_inline.dedupe("email")

    inline_group_ids = list(dg_inline.df["group_id"])

    assert inline_group_ids == expected_group_ids

    dg_asdict = DupeGrouper(df_pandas)
    dg_asdict.add_strategy(
        {
            "address": [
                Exact(),
                Fuzzy(tolerance=0.3),
                TfIdf(tolerance=0.7, ngram=3, topn=3),
            ],
            "email": [Exact()],
        }
    )

    dg_asdict.dedupe()

    asdict_group_ids = list(dg_asdict.df["group_id"])

    assert asdict_group_ids == expected_group_ids

    # Q.E.D
    assert inline_group_ids == asdict_group_ids


def test_iterative_deduplication(df_pandas):
    """tests that deduplication can be iteratively re-applied"""

    def dedupe_iteration(input):

        dg = DupeGrouper(input)
        dg.add_strategy(Fuzzy(tolerance=0.3))
        dg.dedupe("address")
        return dg.df

    # fresh data

    df_pandas = df_pandas[["id", "address", "email"]]

    # dedupe once

    output_iter1 = dedupe_iteration(df_pandas)

    # now we mimic an additional row being added i.e. iterative deduplication
    # But this addition results in a re-ordering!
    # So we add at the start

    print(list(output_iter1["group_id"]))

    output_iter1 = pd.concat(
        [
            output_iter1,
            pd.DataFrame(
                data={
                    "id": [99],
                    "address": ["Calle Sueco, 56, 05688, Rioja, Navarra"],
                    "email": ["hellothere@example.com"],
                    "group_id": [14],
                }
            ),
        ]
    )

    print(output_iter1)

    # note we now expect group_id 14 -> 3 via deduplication and record selection

    expected_group_ids = [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13, 3]

    # dedupe again

    output_iter2 = dedupe_iteration(output_iter1)

    print(list(output_iter2["group_id"]))

    assert expected_group_ids == list(output_iter2["group_id"])
