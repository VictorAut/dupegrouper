from unittest.mock import ANY, Mock, patch

import pytest
import pandas as pd
import polars as pl

from dupegrouper.base import (
    _InitDataFrame,
    _StrategyManager,
    DupeGrouper,
    StrategyTypeError,
)
from dupegrouper.strategies import Exact, Fuzzy, TfIdf
from dupegrouper.strategy import DeduplicationStrategy


dataframe_data = [
    [1, "123ab, OL5 9PL, UK", "bbab@example.com"],
    [
        2,
        "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom",
        "bb@example.com",
    ],
    [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com"],
    [
        4,
        "Calle Sueco, 56, 05688, Rioja, Navarra",
        "hellothere@example.com",
    ],
    [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com"],
    [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com"],
    [7, "C. Ancho 49, 05687, Navarra", "b@example.com"],
    [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com"],
    [9, "123ab, OL5 9PL, UK", "hellathere@example.com"],
    [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com"],
    [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com"],
    [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com"],
    [13, "totally random non existant address", "fictitious@never.co.uk"],
]

# fictitious addresses.
df_pandas = pd.DataFrame(
    columns=["id", "address", "email"],
    data=dataframe_data,
)

df_polars = pl.DataFrame(
    schema=["id", "address", "email"], data=dataframe_data, orient="row"
)


########################
#  TEST _InitDataFrame #
########################


def test_init_dataframe_pandas():
    df_init = _InitDataFrame(df_pandas).choose
    assert "group_id" in df_init.columns
    assert df_init["group_id"].tolist() == [i for i in range(1, 14)]


def test_init_dataframe_polars():
    df_init = _InitDataFrame(df_polars).choose
    assert "group_id" in df_init.columns
    assert df_init["group_id"].to_list() == [i for i in range(1, 14)]


##############################################
#  TEST _StrategyManager + StrategyTypeError #
##############################################


class DummyClass:
    pass


DEFAULT_ERROR_MSG = "Input is not valid"
CLASS_ERROR_MSG = (
    "Input class is not valid: must be an instance of `DeduplicationStrategy`"
)
TUPLE_ERROR_MSG = "Input tuple is not valid: must be a length 2 [callable, dict]"
DICT_ERROR_MSG = (
    "Input dict is not valid: items must be a list of `DeduplicationStrategy` or tuples"
)


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
def test_strategy_manager_validate_addition_strategy(
    test_input, expected_to_pass, base_msg
):
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


def test_dupegrouper_df_attribute_pandas():
    grouper = DupeGrouper(df_pandas)
    assert isinstance(grouper.df, pd.DataFrame)
    assert "group_id" in grouper.df.columns


def test_dupegrouper_df_attribute_polars():
    grouper = DupeGrouper(df_polars)
    assert isinstance(grouper.df, pl.DataFrame)
    assert "group_id" in grouper.df.columns


def test_dupegrouper_add_strategy():
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


def test_dupegrouper_strategies_attribute_inline():
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(Mock(spec=Exact))
    grouper.add_strategy(Mock(spec=Fuzzy))
    grouper.add_strategy((my_dummy_func, {"str": "random"}))

    assert grouper.strategies == tuple(["Exact", "Fuzzy", "my_dummy_func"])

    patch_helper_reset(grouper)


def test_dupegrouper_strategies_attribute_dict():
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

    assert grouper.strategies == dict(
        {"address": ("Exact", "my_dummy_func"), "email": ("Exact", "Fuzzy")}
    )

    patch_helper_reset(grouper)


def test_dupegrouper_add_strategy_equal_execution():
    """strategies can be added in various ways

    This tests that given the differing addition mechanisms, the output is the
    same"""

    expected_group_ids = [1, 2, 3, 3, 2, 1, 2, 2, 1, 1, 11, 12, 13]

    dg_inline = DupeGrouper(df_pandas)
    dg_inline.add_strategy(Exact())
    dg_inline.add_strategy(Fuzzy(tolerance=0.3))
    dg_inline.add_strategy(TfIdf(tolerance=0.7, ngram=3, topn=3))

    dg_inline.dedupe("address")

    dg_inline.add_strategy(Exact())

    dg_inline.dedupe("email")

    inline_group_ids = list(dg_inline.df["group_id"])

    assert expected_group_ids == inline_group_ids

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

    dg_asdict.dedupe("email")

    asdict_group_ids = list(dg_asdict.df["group_id"])

    assert expected_group_ids == asdict_group_ids

    # Q.E.D
    assert inline_group_ids == asdict_group_ids
