"""Tests for dupegrouper.base"""

from unittest.mock import ANY, Mock, patch

import pandas as pd
import polars as pl
import pytest

from dupegrouper.base import (
    _InitDataFrame,
    _StrategyManager,
    DupeGrouper,
    StrategyTypeError,
)
from dupegrouper.strategies import Exact, Fuzzy, TfIdf
from dupegrouper.strategy import DeduplicationStrategy


########################
#  TEST _InitDataFrame #
########################


def test_init_dataframe_pandas(df_pandas):
    df_init = _InitDataFrame(df_pandas).choose
    assert "group_id" in df_init.columns
    assert df_init["group_id"].tolist() == [i for i in range(1, 14)]


def test_init_dataframe_polars(df_polars):
    df_init = _InitDataFrame(df_polars).choose
    assert "group_id" in df_init.columns
    assert df_init["group_id"].to_list() == [i for i in range(1, 14)]


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
        mock_reset.assert_called_once()

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
