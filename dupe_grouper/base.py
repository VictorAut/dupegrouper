import functools
from types import NoneType
import typing

import pandas as pd
from multipledispatch import dispatch

from strategy import DeduplicationStrategy


# TYPES:


strategy_map = typing.Mapping[
    str,
    tuple[
        #
        tuple[
            #
            DeduplicationStrategy,
            typing.Mapping[str, typing.Any],
        ],
        ...,
    ],
]


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["group_id"] = range(1, len(df) + 1)
        self.strategies = []

    def add_strategy(self, strategy: DeduplicationStrategy):
        self.strategies.append(strategy)

    def map_strategies(self, strategy_map: strategy_map):
        self.strategies = strategy_map

    @dispatch(list, str)
    def _dedupe(self, strategies, attr):
        for strategy in strategies:

            self.df = strategy.dedupe(self.df, attr)

        self.strategies = []  # re-initialise

    @dispatch(dict, NoneType)
    def _dedupe(self, strategies: strategy_map, attr):
        del attr  # Unused when strategies are mapped
        for _attr, dedupers_and_kwargs in strategies.items():
            for deduper, kwargs in dedupers_and_kwargs:
                self.df = deduper(**kwargs).dedupe(self.df, _attr)

        self.strategies = []  # re-initialise

    @typing.override
    def dedupe(self, attr: str | None = None):
        self._dedupe(self.strategies, attr)



import data
import deduplication

df1 = data.df3

dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=0.05))
dg.add_strategy(deduplication.TfIdf(tolerance=0.7))

dg.dedupe("address")

dg.strategies

dg.df

######################

df1 = data.df3

dg = DupeGrouper(df1)

strategies: strategy_map = {
    "address": (
        (deduplication.Exact, {}),
        (deduplication.Fuzzy, {"tolerance": 0.05}),
        (deduplication.TfIdf, {"tolerance": 0.5, "ngram": 3, "topn": 4}),
    )
}

dg.map_strategies(strategies)

dg.dedupe()

dg.strategies

dg.df
