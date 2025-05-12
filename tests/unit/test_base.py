"""Tests for dupegrouper.base"""

import importlib
import os
from unittest.mock import ANY, Mock, patch

import pandas as pd
import polars as pl
import pytest

from dupegrouper.base import (
    DupeGrouper,
    DeduplicationStrategy,
    StrategyTypeError,
    _StrategyManager,
    _wrap,
)
from dupegrouper.wrappers import WrappedDataFrame
import dupegrouper.definitions
import dupegrouper.wrappers
import dupegrouper.wrappers.dataframes
from dupegrouper.strategies import Exact, Fuzzy, TfIdf


#############################
#  TEST _wrap #
#############################


def test_init_dataframe_pandas(df_pandas_raw: pd.DataFrame):
    df_container: WrappedDataFrame = _wrap(df_pandas_raw)
    df: pd.DataFrame = df_container.unwrap()  # type: ignore
    assert "group_id" in df.columns
    assert df["group_id"].tolist() == [i for i in range(1, 14)]


def test_init_dataframe_polars(df_polars_raw: pl.DataFrame):
    df_container: WrappedDataFrame = _wrap(df_polars_raw)
    df: pl.DataFrame = df_container.unwrap()  # type: ignore
    assert "group_id" in df.columns
    assert df["group_id"].to_list() == [i for i in range(1, 14)]


def test_dataframe_dispatcher_unsupported():
    class FakeDataFrame:
        pass

    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        _wrap(FakeDataFrame())


######################
#  TEST set group_id #
######################


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
)
def test_different_group_id_env_var(env_var_value, expected_value, df_pandas_raw):
    if env_var_value:
        os.environ["GROUP_ID"] = env_var_value
    else:
        os.environ.pop("GROUP_ID", None)  # remove it if exists

    importlib.reload(dupegrouper.definitions)  # reset constant
    importlib.reload(dupegrouper.wrappers.dataframes._pandas)  # final value in `base`
    df_init = _wrap(df_pandas_raw).unwrap()
    assert expected_value in df_init.columns

    # clean up
    os.environ["GROUP_ID"] = "group_id"
    importlib.reload(dupegrouper.definitions)
    importlib.reload(dupegrouper.wrappers.dataframes._pandas)


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
    "test_input, expected_to_pass, base_msg",
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
)
def test_strategy_manager_validate_addition_strategy(test_input, expected_to_pass, base_msg):
    """validates that the input 'strtagey' is legit, against `StrategyTypeError`"""
    manager = _StrategyManager()
    mock_strategy = test_input
    if expected_to_pass:
        if isinstance(mock_strategy, dict):
            for k, value in mock_strategy.items():
                for v in value:
                    manager.add(k, v)
                    assert (k in manager.get()) is expected_to_pass
        else:
            manager.add("default", mock_strategy)
            assert ("default" in manager.get()) is expected_to_pass
    else:
        with pytest.raises(StrategyTypeError) as e:
            raise (StrategyTypeError(mock_strategy))
        assert base_msg in str(e)


def test_strategy_manager_reset():
    manager = _StrategyManager()
    mock_strategy = Mock(spec=DeduplicationStrategy)
    manager.add("name", mock_strategy)
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
