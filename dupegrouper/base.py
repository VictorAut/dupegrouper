import collections.abc
import collections
from functools import singledispatchmethod
import logging
from types import NoneType
import typing

import pandas as pd
import polars as pl
from multipledispatch import dispatch

from dupegrouper.definitions import GROUP_ID, strategy_list_collection, strategy_map_collection, frames
from dupegrouper.strategies.custom import Custom
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# DATAFRAME CONSTRUCTOR:


class _InitDataFrame:

    def __init__(self, df):
        self._df = self._init_dispatch(df)

    @singledispatchmethod
    @staticmethod
    def _init_dispatch(df: frames):
        raise NotImplementedError(f"Unsupported data frame: {type(df)}")

    @_init_dispatch.register(pd.DataFrame)
    def _(self, df):
        return df.assign(**{GROUP_ID: range(1, len(df) + 1)})

    @_init_dispatch.register(pl.DataFrame)
    def _(self, df):
        return df.with_columns(**{GROUP_ID: range(1, len(df) + 1)})

    @property
    def choose(self):
        return self._df
    

# STRATEGY MANAGER:


class _StrategyManager:
    def __init__(self):
        self._strategies: strategy_map_collection | strategy_list_collection = []
    


    


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self._df = _InitDataFrame(df).choose
        # self._strategy_manager = _StrategyManager()
        DeduplicationStrategy._tally = collections.defaultdict(list) # i.e. reset

    @singledispatchmethod
    def _call_strategy_deduper(
        self,
        strategy: DeduplicationStrategy | tuple[typing.Callable, typing.Any],
        attr: str,
    ):
        del attr  # Unused
        return NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @_call_strategy_deduper.register(DeduplicationStrategy)
    def _(self, strategy, attr):
        return strategy.dedupe(self._df, attr)

    @_call_strategy_deduper.register(tuple)
    def _(self, strategy: tuple[typing.Callable, typing.Any], attr):
        func, kwargs = strategy
        return Custom(func, self._df, attr, **kwargs).dedupe()

    @dispatch(list, str)
    def _dedupe(self, strategy_collection: strategy_list_collection, attr):
        for strategy in strategy_collection:
            self._df = self._call_strategy_deduper(strategy, attr)
        self._tally: dict = DeduplicationStrategy._tally

    @dispatch(dict, NoneType)  # type: ignore[no-redef]
    def _dedupe(self, strategy_collection: strategy_map_collection, attr):
        del attr  # Unused
        for attr, strategies in strategy_collection.items():
            for strategy in strategies:
                self._df = self._call_strategy_deduper(strategy, attr)
        self._tally: dict = DeduplicationStrategy._tally  # type: ignore[no-redef]

    def _report(self):
        return {k: v for k, v in self._tally.items() if len(v) > 1}

    # PUBLIC API:

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def strategies(self) -> None | list[str] | dict[str, tuple[str, ...]]:
        def name_conditional(iter):
            return (
                iter[0].__name__ if isinstance(iter, tuple) else iter.__class__.__name__
            )
        if not hasattr(self, "_strategy_collection"):
            return None
        if isinstance(self._strategy_collection, list):
            return [name_conditional(strat) for strat in self._strategy_collection]
        return {
            k: tuple([(name_conditional(vx)) for vx in v])
            for k, v in self._strategy_collection.items()
        }

    @singledispatchmethod
    def add_strategy(self, strategy: DeduplicationStrategy | tuple | strategy_map_collection):
        return NotImplementedError(f"Unsupported strategy: {type(strategy())}")

    @add_strategy.register(DeduplicationStrategy)
    @add_strategy.register(tuple)
    def _(self, strategy):
        if not hasattr(self, "_strategy_collection"):
            self._strategy_collection = []
        self._strategy_collection.append(strategy)

    @add_strategy.register(dict)
    def _(self, strategy: strategy_map_collection):
        self._strategy_collection = strategy

    def dedupe(self, attr: str | None = None):
        self._dedupe(self._strategy_collection, attr)
        del self._strategy_collection # i.e. reset

    @property
    def report(self) -> dict:
        raise NotImplementedError
