import pytest
import pandas as pd
import polars as pl
from dupegrouper.base import (
    _InitDataFrame,
    _StrategyManager,
    DupeGrouper,
    StrategyTypeError,
)
from dupegrouper.strategy import DeduplicationStrategy
from unittest.mock import Mock

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
    schema=["id", "address", "email"],
    data=dataframe_data
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


# ---- TEST _StrategyManager ----

##########################
#  TEST _StrategyManager #
##########################


def test_strategy_manager_add_valid_strategy():
    manager = _StrategyManager()
    mock_strategy = Mock(spec=DeduplicationStrategy)
    manager.add("name", mock_strategy)
    assert "name" in manager.get()


# def test_strategy_manager_invalid_strategy():
#     manager = _StrategyManager()
#     with pytest.raises(StrategyTypeError):
#         manager.add("name", "invalid_strategy")  # Not a valid type


# def test_strategy_manager_reset():
#     manager = _StrategyManager()
#     mock_strategy = Mock(spec=DeduplicationStrategy)
#     manager.add("name", mock_strategy)
#     manager.reset()
#     assert manager.get() == {}


# # ---- TEST DupeGrouper ----


# def test_dupegrouper_initialization():
#     df = pd.DataFrame({"name": ["Alice", "Bob"]})
#     grouper = DupeGrouper(df)
#     assert isinstance(grouper.df, pd.DataFrame)
#     assert "group_id" in grouper.df.columns


# def test_dupegrouper_add_strategy():
#     df = pd.DataFrame({"name": ["Alice", "Bob"]})
#     grouper = DupeGrouper(df)
#     mock_strategy = Mock(spec=DeduplicationStrategy)
#     grouper.add_strategy(mock_strategy)
#     assert "default" in grouper._strategy_manager.get()


# def test_dupegrouper_dedupe():
#     df = pd.DataFrame({"name": ["Alice", "Bob"]})
#     grouper = DupeGrouper(df)
#     mock_strategy = Mock(spec=DeduplicationStrategy)
#     mock_strategy.dedupe = Mock(return_value=df)

#     grouper.add_strategy(mock_strategy)
#     grouper.dedupe("name")

#     mock_strategy.dedupe.assert_called_with("name")
#     assert isinstance(grouper.df, pd.DataFrame)
