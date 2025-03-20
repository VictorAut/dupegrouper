import collections.abc
import functools
from types import NoneType
import typing

import pandas as pd
from multipledispatch import dispatch

from strategy import DeduplicationStrategy


# TYPES:


strategy_map = collections.abc.Mapping[
    str,
    tuple[
        #
        tuple[
            #
            DeduplicationStrategy,
            collections.abc.Mapping[str, typing.Any],
        ],
        ...,
    ],
]


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["group_id"] = range(1, len(df) + 1)
        self._strategies: list[DeduplicationStrategy] | strategy_map = []

    @property
    def strategies(self) -> list[DeduplicationStrategy] | strategy_map:
        return self._strategies

    @strategies.setter
    def strategies(self, value: list[DeduplicationStrategy] | strategy_map):
        if not isinstance(value, (list, dict)):
            raise TypeError("strategies must be a list of subclasses of base.DeduplicationStrategy or a `strategy_map` dict.")
        self._strategies = value

    def add_strategy(self, strategy: DeduplicationStrategy):
        self._strategies.append(strategy)

    def map_strategies(self, strategy_map: strategy_map):
        self._strategies = strategy_map

    @dispatch(list, str)
    def _dedupe(self, strategies, attr):
        for strategy in strategies:

            self.df = strategy.dedupe(self.df, attr)

        self._strategies = []  # re-initialise

    @dispatch(dict, NoneType)
    def _dedupe(self, strategies: strategy_map, attr):
        del attr  # Unused when strategies are mapped
        for _attr, dedupers_and_kwargs in strategies.items():
            for deduper, kwargs in dedupers_and_kwargs:
                self.df = deduper(**kwargs).dedupe(self.df, _attr)

        self._strategies = []  # re-initialise

    def dedupe(self, attr: str | None = None):
        self._dedupe(self._strategies, attr)

