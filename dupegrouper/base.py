import collections.abc
import collections
from functools import singledispatchmethod
import inspect
import logging
from types import NoneType
import typing

import pandas as pd
import polars as pl

from dupegrouper.definitions import (
    GROUP_ID,
    strategy_map_collection,
    frames,
)
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


# STRATEGY MANAGMENT:


class StrategyTypeError(Exception):
    """raise strategy type errors"""

    def __init__(self, strategy):
        msg = "Input is not valid"  # i.e. default
        if inspect.isclass(strategy):
            msg = f"Input class must be an instance of `DeduplicationStrategy`, not: {type(strategy())}"
        if isinstance(strategy, tuple):
            msg = f"Input tuple is not valid: must be a length 2 [callable, dict], not {strategy}"
        if isinstance(strategy, dict):
            msg = "Input dict is not valid: items must be a list of `DeduplicationStrategy` or tuples"
        super(StrategyTypeError, self).__init__(msg)


class _StrategyManager:
    def __init__(self):
        self._strategies: strategy_map_collection = collections.defaultdict(list)

    def add(
        self,
        attr_key: str,
        strategy: DeduplicationStrategy | tuple,
    ):
        if self.validate(strategy):
            self._strategies[attr_key].append(strategy)  # type: ignore[attr-defined]
            return
        raise StrategyTypeError(strategy)

    def get(self):
        return self._strategies

    def validate(self, strategy):
        if isinstance(strategy, DeduplicationStrategy):
            return True
        if isinstance(strategy, tuple) and len(strategy) == 2:
            func, kwargs = strategy
            return callable(func) and isinstance(kwargs, dict)
        if isinstance(strategy, dict):
            for _, v in strategy.items():
                if not self.validate(v) or not isinstance(v, list):
                    return False
        return False

    def reset(self):
        self.__init__()


# BASE:


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self._df = _InitDataFrame(df).choose
        self._strategy_manager = _StrategyManager()
        DeduplicationStrategy._tally = collections.defaultdict(list)  # i.e. reset

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

    @singledispatchmethod
    def _dedupe(
        self,
        attr: str | None,
        strategy_collection: strategy_map_collection,
    ):
        del strategy_collection  # Unused
        raise NotImplementedError(f"Unsupported type: {type(attr)}")

    @_dedupe.register(str)
    def _(self, attr, strategy_collection):
        for strategy in strategy_collection["default"]:
            self._df = self._call_strategy_deduper(strategy, attr)
        self._tally: dict = DeduplicationStrategy._tally

    @_dedupe.register(NoneType)
    def _(self, attr, strategy_collection):
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
    def strategies(self) -> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
        strategies = self._strategy_manager.get()
        if not strategies:
            return None

        def parse_strategies(dict_values):
            return tuple(
                [
                    (vx[0].__name__ if isinstance(vx, tuple) else vx.__class__.__name__)
                    for vx in dict_values
                ]
            )

        if "default" in strategies:
            return tuple([parse_strategies(v) for _, v in strategies.items()])[0]
        return {k: parse_strategies(v) for k, v in strategies.items()}

    @singledispatchmethod
    def add_strategy(
        self, strategy: DeduplicationStrategy | tuple | strategy_map_collection
    ):
        return NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @add_strategy.register(DeduplicationStrategy)
    @add_strategy.register(tuple)
    def _(self, strategy):
        self._strategy_manager.add("default", strategy)

    @add_strategy.register(dict)
    def _(self, strategy: strategy_map_collection):
        for attr, strat_list in strategy.items():
            for strat in strat_list:
                self._strategy_manager.add(attr, strat)

    def dedupe(self, attr: str | None = None):
        self._dedupe(attr, self._strategy_manager.get())
        self._strategy_manager.reset()

    @property
    def report(self) -> dict:
        raise NotImplementedError
