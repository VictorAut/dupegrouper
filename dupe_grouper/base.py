import collections.abc
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
        self.df = df
        self.df["group_id"] = range(1, len(df) + 1)
        self._strategy_collection: list[DeduplicationStrategy] | strategies_map = []

    @dispatch(DeduplicationStrategy, str)
    def _call_strategy_deduper(self, strategy, _attr):
        return strategy.dedupe(self.df, _attr)

    @dispatch(tuple, str)
    def _call_strategy_deduper(
        self,
        strategy: tuple[typing.Callable, typing.Any],
        _attr,
    ):
        func, kwargs = strategy
        return Custom(func=func, df=self.df, attr=_attr, **kwargs).dedupe()

    @dispatch(list, str)
    def _dedupe(self, strategy_collection, attr):
        for strategy in strategy_collection:

            self.df = strategy.dedupe(self.df, attr)

        self._strategy_collection = []  # re-initialise

    @dispatch(dict, NoneType)
    def _dedupe(self, strategy_collection: strategies_map, attr):
        del attr  # Unused when strategies are mapping
        for _attr, strategies in strategy_collection.items():
            for strategy in strategies:
                # self.df = strategy.dedupe(self.df, _attr)
                self.df = self._call_strategy_deduper(strategy, _attr)

        self._strategy_collection = []  # re-initialise

    # PUBLIC API:

    @property
    def strategies(self) -> list[DeduplicationStrategy] | strategies_map:
        return self._strategy_collection

    @dispatch(DeduplicationStrategy)
    def add_strategy(self, strategy):
        self._strategy_collection.append(strategy)

    @dispatch(dict)
    def add_strategy(self, strategy: strategies_map):
        self._strategy_collection = strategy

    def dedupe(self, attr: str | None = None):
        self._dedupe(self._strategy_collection, attr)
