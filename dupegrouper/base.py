import collections
import collections.abc
from functools import singledispatchmethod
from types import NoneType
import typing

import pandas as pd
import polars as pl
from multipledispatch import dispatch

from dupegrouper.strategies.custom import Custom
from dupegrouper.strategy import DeduplicationStrategy


# TYPES:


strategies_map = collections.abc.Mapping[
    str,
    tuple[
        DeduplicationStrategy,
        ...,
    ],
]


class _InitDataFrame:

    def __init__(self, df):
        self._df = self._init_dispatch(df)

    @singledispatchmethod
    def _init_dispatch(df):
        raise NotImplementedError(f"Unsupported data frame: {type(df)}")

    @_init_dispatch.register(pd.DataFrame)
    def _(self, df):
        return df.assign(group_id=range(1, len(df) + 1))

    @_init_dispatch.register(pl.DataFrame)
    def _(self, df):
        return df.with_columns(group_id=range(1, len(df) + 1))

    @property
    def choose(self):
        return self._df


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self._df = _InitDataFrame(df).choose
        self._strategy_collection: list[DeduplicationStrategy] | strategies_map = []
        DeduplicationStrategy._tally = collections.defaultdict(list)

    @dispatch(DeduplicationStrategy, str)
    def _call_strategy_deduper(self, strategy, _attr):
        return strategy.dedupe(self._df, _attr)

    @dispatch(tuple, str)
    def _call_strategy_deduper(
        self,
        strategy: tuple[typing.Callable, typing.Any],
        _attr,
    ):
        func, kwargs = strategy
        return Custom(func=func, df=self._df, attr=_attr, **kwargs).dedupe()

    @dispatch(list, str)
    def _dedupe(self, strategy_collection, attr):
        for strategy in strategy_collection:
            self._df = self._call_strategy_deduper(strategy, attr)
        self._tally: dict = DeduplicationStrategy._tally

    @dispatch(dict, NoneType)
    def _dedupe(self, strategy_collection: strategies_map, attr):
        del attr  # Unused
        for attr, strategies in strategy_collection.items():
            for strategy in strategies:
                self._df = self._call_strategy_deduper(strategy, attr)
        self._tally: dict = DeduplicationStrategy._tally

    def _report(self):
        return {k: v for k, v in self._tally.items() if len(v) > 1}

    # PUBLIC API:

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def strategies(self) -> list[DeduplicationStrategy] | strategies_map:
        return {
            k: tuple(
                [
                    (vx[0].__name__ if isinstance(vx, tuple) else vx.__class__.__name__)
                    #
                    for vx in v
                ]
            )
            for k, v in self._strategy_collection.items()
        }

    @singledispatchmethod
    def add_strategy(self, strategy: DeduplicationStrategy | tuple | strategies_map):
        return NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @add_strategy.register(DeduplicationStrategy | tuple)
    def _(self, strategy: tuple[typing.Callable, typing.Any]):
        self._strategy_collection.append(strategy)

    @add_strategy.register(dict)
    def _(self, strategy: strategies_map):
        self._strategy_collection = strategy

    def dedupe(self, attr: str | None = None):
        self._dedupe(self._strategy_collection, attr)

    @property
    def report(self) -> dict:
        raise NotImplementedError
