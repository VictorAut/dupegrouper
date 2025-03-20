import collections.abc
import functools
from types import NoneType
import typing

import pandas as pd
from multipledispatch import dispatch

from deduplication import Custom
from strategy import DeduplicationStrategy



# TYPES:


strategies_map = collections.abc.Mapping[
    str,
    tuple[
        DeduplicationStrategy,
        ...,
    ],
]


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        # self.df = df
        # self.df["group_id"] = range(1, len(df) + 1)
        self._get_df(df)
        self._strategy_collection: list[DeduplicationStrategy] | strategies_map = []

    def _get_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self._df = df.assign(group_id = range(1, len(df) + 1))

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

    @dispatch(dict, NoneType)
    def _dedupe(self, strategy_collection: strategies_map, attr):
        del attr  # Unused
        for attr, strategies in strategy_collection.items():
            for strategy in strategies:
                self._df = self._call_strategy_deduper(strategy, attr)

    # PUBLIC API:

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def strategies(self) -> list[DeduplicationStrategy] | strategies_map:
        return self._strategy_collection

    @functools.singledispatchmethod
    def add_strategy(self, strategy: DeduplicationStrategy | tuple | strategies_map):
        return NotImplementedError(f"Not implemented with type {type(strategy)}")

    @add_strategy.register(DeduplicationStrategy | tuple)
    def _(self, strategy: tuple[typing.Callable, typing.Any]):
        self._strategy_collection.append(strategy)

    @add_strategy.register(dict)
    def _(self, strategy: strategies_map):
        self._strategy_collection = strategy

    def dedupe(self, attr: str | None = None):
        self._dedupe(self._strategy_collection, attr)
