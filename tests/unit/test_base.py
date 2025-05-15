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


# dummy

class DummyClass:
    pass

def dummy_func():
    pass


###############
#  TEST _wrap #
###############

DATAFRAME_TYPES = {
    pd.DataFrame: WrappedPandasDataFrame,
    pl.DataFrame: WrappedPolarsDataFrame,
    SparkDataFrame: WrappedSparkDataFrame,
    list[Row]: WrappedSparkRows,
}


def test__wrap_dataframe(dataframe):
    df, _, id = dataframe

    expected_type = DATAFRAME_TYPES.get(type(df))

    df_wrapped: WrappedDataFrame = _wrap(df, id)

    assert isinstance(df_wrapped, expected_type)


def test__wrap_dataframe_raises():
    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        _wrap(DummyClass())


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
        "invalid dict",
    ],
)
def test__strategy_manager_validate_addition_strategy(strategy, expected_to_pass, base_msg):
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


def test__strategy_manager_reset():
    manager = _StrategyManager()
    strategy = Mock(spec=DeduplicationStrategy)
    manager.add("name", strategy)
    manager.reset()
    assert manager.get() == {}


###############################
# TEST _call_strategy_deduper #
###############################


def test__call_strategy_deduper_deduplication_strategy(mocked_dupegrouper):
    attr = "address"

    strategy = Mock(spec=DeduplicationStrategy)
    deduped_df_mock = Mock()
    strategy.with_frame.return_value.dedupe.return_value = deduped_df_mock

    result = mocked_dupegrouper._call_strategy_deduper(strategy, attr)

    # assert

    strategy.with_frame.assert_called_once_with(mocked_dupegrouper._df)
    strategy.with_frame.return_value.dedupe.assert_called_once_with(attr)

    assert result == deduped_df_mock


def test__call_strategy_deduper_tuple(mocked_dupegrouper):
    attr = "address"

    mock_callable = Mock()
    mock_callable.__name__ = "mock_func"

    mock_kwargs = {"tolerance": 0.8}

    deduped_df_mock = Mock()

    with patch("dupegrouper.base.Custom") as Custom:

        # Mock instance that Custom returns
        instance = Mock()
        Custom.return_value = instance

        # Ensure full method chain is mocked
        instance.with_frame.return_value = instance
        instance.dedupe.return_value = deduped_df_mock

        result = mocked_dupegrouper._call_strategy_deduper(
            (mock_callable, mock_kwargs),  # tuple!
            attr,
        )

        # assert

        Custom.assert_called_once_with(mock_callable, attr, **mock_kwargs)
        instance.with_frame.assert_called_once_with(mocked_dupegrouper._df)
        instance.with_frame.return_value.dedupe.assert_called_once_with()

        assert result == deduped_df_mock


@pytest.mark.parametrize(
    "input, type",
    [
        (42, r".*int.*"),
        (DummyClass(), r".*DummyClass.*"),
        (["a"], r".*list.*"),
        ({"a": "b"}, r".*dict.*"),
    ],
    ids=["invalid int", "invalid class", "invalid list", "invalid dict"],
)
def test__call_strategy_deduper_raises(input, type, mocked_dupegrouper):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        mocked_dupegrouper._call_strategy_deduper(input, "address")


################
# TEST _dedupe #
################


def test__dedupe_str_attr(mocked_dupegrouper):
    attr = "address"

    strat1 = Mock(spec=DeduplicationStrategy)
    strat2 = Mock(spec=DeduplicationStrategy)
    strat3 = Mock(spec=DeduplicationStrategy)

    strategy_collection = {"default": [strat1, strat2, strat3]}

    with patch.object(mocked_dupegrouper, "_call_strategy_deduper") as call_deduper:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # after third

        call_deduper.side_effect = [
            df1,
            df2,
            df3,
        ]

        mocked_dupegrouper._dedupe(attr, strategy_collection)

        assert call_deduper.call_count == 3

        call_deduper.assert_any_call(strat1, attr)
        call_deduper.assert_any_call(strat2, attr)
        call_deduper.assert_any_call(strat3, attr)

        assert mocked_dupegrouper._df == df3


def test__dedupe_nonetype_attr(mocked_dupegrouper):

    attr = None  # Important!

    strat1 = Mock(spec=DeduplicationStrategy)
    strat2 = Mock(spec=DeduplicationStrategy)
    strat3 = Mock(spec=DeduplicationStrategy)
    strat4 = Mock(spec=DeduplicationStrategy)

    strategy_collection = {
        "attr1": [strat1, strat2],
        "attr2": [strat3, strat4],
    }

    with patch.object(mocked_dupegrouper, "_call_strategy_deduper") as call_deduper:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # ...
        df4 = (Mock(),)  # after fourth dedupe

        call_deduper.side_effect = [df1, df2, df3, df4]

        mocked_dupegrouper._dedupe(attr, strategy_collection)

        assert call_deduper.call_count == 4

        call_deduper.assert_any_call(strat1, "attr1")
        call_deduper.assert_any_call(strat2, "attr1")
        call_deduper.assert_any_call(strat3, "attr2")
        call_deduper.assert_any_call(strat3, "attr2")

        assert mocked_dupegrouper._df == df4


@pytest.mark.parametrize(
    "attr_input, type",
    [
        (42, r".*int.*"),
        ([42], r".*list.*"),
        ((42,), r".*tuple.*"),
        ({"a": 42}, r".*dict.*"),
        (42.0, r".*float.*"),
    ],
    ids=["invalid int", "invalid list", "invalid tuple", "invalid dict", "invalid float"],
)
def test__dedupe_raises(attr_input, type, mocked_dupegrouper):
    with pytest.raises(NotImplementedError, match=f"Unsupported attribute type: {type}"):
        mocked_dupegrouper._dedupe(attr_input, {}) # any dict


#####################
# TEST add_strategy #
#####################

@pytest.mark.parametrize(
        "strategy",
        [
            (dummy_func, {'tolerance': 0.8}),
            Mock(spec=DeduplicationStrategy)
        ],
        ids = ["tuple", "DeduplicationStrategy"]
)
def test_add_strategy_deduplication_strategy_or_tuple(strategy, mocked_dupegrouper):

    with patch.object(mocked_dupegrouper, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            mocked_dupegrouper.add_strategy(strategy)

            assert add.call_count == 1

            add.assert_any_call("default", strategy)

def test_add_strategy_dict(mocked_dupegrouper):
    strat1 = Mock(spec=DeduplicationStrategy)
    strat2 = Mock(spec=DeduplicationStrategy)
    strat3 = Mock(spec=DeduplicationStrategy)
    strat4 = Mock(spec=DeduplicationStrategy)

    strategy = {
        "attr1": [strat1, strat2],
        "attr2": [strat3, strat4],
    }

    with patch.object(mocked_dupegrouper, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            mocked_dupegrouper.add_strategy(strategy)

            assert add.call_count == 4

            add.assert_any_call("attr1", strat1)
            add.assert_any_call("attr1", strat2)
            add.assert_any_call("attr2", strat3)
            add.assert_any_call("attr2", strat3)

     

@pytest.mark.parametrize(
    "strategy, type",
    [
        (DummyClass(), r".*DummyClass.*"),
        ([42], r".*list.*"),
    ],
    ids=["invalid class", "invalid list"],
)
def test_add_strategy_raises(strategy, type, mocked_dupegrouper):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        mocked_dupegrouper.add_strategy(strategy)


#####################
# TEST dedupe #
#####################


# TODO


##################################
# TEST DupeGrouper - public API! #
##################################


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
    grouper.add_strategy((dummy_func, {"str": "random"}))

    assert grouper.strategies == tuple(["Exact", "Fuzzy", "dummy_func"])

    patch_helper_reset(grouper)


def test_dupegrouper_strategies_attribute_dict(df_pandas):
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(
        {
            "address": [
                Mock(spec=Exact),
                (dummy_func, {"key": "value"}),
            ],
            "email": [
                Mock(spec=Exact),
                Mock(spec=Fuzzy),
            ],
        }
    )

    assert grouper.strategies == dict({"address": ("Exact", "dummy_func"), "email": ("Exact", "Fuzzy")})

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
