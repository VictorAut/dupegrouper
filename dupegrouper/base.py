"""dupegrouper main entrypoint

This module contains `DupeGrouper`, at the core of all 'dupe and group'
functionality provided by dupegrouper.
"""

from __future__ import annotations
import collections
from functools import singledispatch, singledispatchmethod
import inspect
import logging

try:
    from types import NoneType
except ImportError:
    NoneType = type(None)  # type: ignore
import typing

import pandas as pd
import polars as pl
from pyspark.sql import (
    SparkSession,
    Row,
    DataFrame as SparkDataFrame,  # i.e. no clash with generic DataFrame definition
)
from pyspark.sql.types import StringType, StructField, StructType

from dupegrouper.definitions import StrategyMapCollection, DataFrame, GROUP_ID
from dupegrouper.wrappers.dataframes import (
    WrappedPandasDataFrame,
    WrappedPolarsDataFrame,
    WrappedSparkDataFrame,
    WrappedSparkRows,
)
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategies.custom import Custom
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


_logger = logging.getLogger(__name__)


# CORE:


class DupeGrouper:
    """Top-level entrypoint for grouping duplicates

    This class handles initialisation of a dataframe, dispatching appropriately
    given the supported dataframe libraries (e.g. Pandas). An instance of this
    class can then accept a variety of strategies for deduplication and
    grouping.

    Upon initialisation, `DupeGrouper` sets a new column, usually `"group_id"`
    — but you can control this by setting an environment variable `GROUP_ID` at
    runtime. The group_id is a monotonically increasing, numeric id column
    starting at 1 to the length of the dataframe provided.
    """

    def __init__(
        self,
        df: DataFrame,
        spark_session: SparkSession = None,
        id: str | None = None,
    ):
        self._df: WrappedDataFrame = _wrap(df, id)
        self._strategy_manager = _StrategyManager()
        self._id = id
        self.spark_session = spark_session

    @singledispatchmethod
    def _call_strategy_deduper(
        self,
        strategy: DeduplicationStrategy | tuple[typing.Callable, typing.Any],
        attr: str,
    ):
        """Dispatch the appropriate strategy deduplication method.

        If the strategy is an instance of a dupegrouper `DeduplicationStrategy`
        the strategy will have been added as such, with it's parameters. In the
        case of a custom implementation of a Callable, passed as a tuple, we
        pass this *directly* to the `Custom` class and initialise that.

        Args:
            strategy: A `dupegrouper` deduplication strategy or a tuple
                containing a (customer) callable and its parameters.
            attr: The attribute used for deduplication.

        Returns:
            A deduplicated dataframe

        Raises:
            NotImplementedError.
        """
        del attr  # Unused
        return NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @_call_strategy_deduper.register(DeduplicationStrategy)
    def _(self, strategy, attr) -> WrappedDataFrame:
        return strategy.with_frame(self._df).dedupe(attr)

    @_call_strategy_deduper.register(tuple)
    def _(self, strategy: tuple[typing.Callable, typing.Any], attr) -> WrappedDataFrame:
        func, kwargs = strategy
        return Custom(func, attr, **kwargs).with_frame(self._df).dedupe()

    @singledispatchmethod
    def _dedupe(
        self,
        attr: str | None,
        strategy_collection: StrategyMapCollection,
    ):
        """Dispatch the appropriate deduplication logic.

        If strategies have been added individually, they are stored under a
        "default" key and retrived as such when the public `.dedupe` method is
        called _with_ the attribute label. In the case of having added
        strategies in one go with a direct dict (mapping) object, the attribute
        label is first extracted from strategy collection dictionary keys.
        Upon completing deduplication the strategy collection is wiped for
        (any) subsequent deduplication.

        Args:
            attr: The attribute used for deduplication; or None in the case
                of strategies being a mapping object

        Returns:
            None; internal `_df` attribute is updated.

        Raises:
            NotImplementedError.
        """
        del strategy_collection  # Unused
        raise NotImplementedError(f"Unsupported type: {type(attr)}")

    @_dedupe.register(str)
    def _(self, attr, strategy_collection):
        for strategy in strategy_collection["default"]:
            self._df = self._call_strategy_deduper(strategy, attr)
        self._strategy_manager.reset()

    @_dedupe.register(NoneType)
    def _(self, attr, strategy_collection):
        del attr  # Unused
        for attr, strategies in strategy_collection.items():
            for strategy in strategies:
                self._df = self._call_strategy_deduper(strategy, attr)
        self._strategy_manager.reset()

    # PUBLIC API:

    @singledispatchmethod
    def add_strategy(self, strategy: DeduplicationStrategy | tuple | StrategyMapCollection):
        """
        Add a strategy to the strategy manager.

        Instances of `DeduplicationStrategy` or tuple are added to the
        "default" key. Mapping objects update the manager directly

        Args:
            strategy: A deduplication strategy, tuple, or strategy collection
                (mapping) to add.

        Returns:
            self is updated

        Raises:
            NotImplementedError
        """
        return NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @add_strategy.register(DeduplicationStrategy)
    @add_strategy.register(tuple)
    def _(self, strategy):
        self._strategy_manager.add("default", strategy)

    @add_strategy.register(dict)
    def _(self, strategy: StrategyMapCollection):
        for attr, strat_list in strategy.items():
            for strat in strat_list:
                self._strategy_manager.add(attr, strat)

    def dedupe(self, attr: str | None = None):
        """dedupe, and group, the data based on the provided attribute

        Args:
            attr: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        if not isinstance(self._df, WrappedSparkDataFrame):
            self._dedupe(attr, self._strategy_manager.get())
        else:

            def _process_partition(partition_iter: typing.Iterator[Row], strategies, id: str) -> typing.Iterator[Row]:
                # handle empty partitions
                rows = list(partition_iter)
                if not rows:
                    return iter([])

                # re-instantiate strategies based on driver's
                reinstantiated_strategies = {}
                for key, values in strategies.items():
                    reinstantiated_strategies[key] = [
                        v if isinstance(v, tuple) else v.reinstantiate()
                        #
                        for v in values
                    ]

                # Core API reused per partition, per worker node
                dg = DupeGrouper(rows, id=id)
                dg.add_strategy(strategies)
                dg.dedupe()

                return iter(dg.df)

            strategies = self._strategy_manager.get()
            id = self._id

            deduped_rdd = self._df.rdd.mapPartitions(
                lambda partition_iter: _process_partition(partition_iter, strategies, id)
            )
            self._df = WrappedSparkDataFrame(
                self.spark_session.createDataFrame(
                    deduped_rdd,
                    schema=StructType(self._df.schema.fields + [StructField(GROUP_ID, StringType(), True)]),
                )
            )

    @property
    def strategies(self) -> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
        """
        Returns the strategies currently stored in the strategy manager.

        If no strategies are stored, returns `None`. Otherwise, returns a tuple
        of strategy names or a dictionary mapping attributes to their
        respective strategies.

        Returns:
            The stored strategies, formatted
        """
        strategies = self._strategy_manager.get()
        if not strategies:
            return None

        def parse_strategies(dict_values):
            return tuple(
                [
                    (vx[0].__name__ if isinstance(vx, tuple) else vx.__class__.__name__)
                    #
                    for vx in dict_values
                ]
            )

        if "default" in strategies:
            return tuple([parse_strategies(v) for _, v in strategies.items()])[0]
        return {k: parse_strategies(v) for k, v in strategies.items()}

    @property
    def df(self) -> DataFrame:
        return self._df.unwrap()


# STRATEGY MANAGER:


class _StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Strategies are collected into a dictionary-like collection where keys are
    attribute names, and values are lists of strategies. Validation is provided
    upon addition allowing only the following stratgies types:
        - `DeduplicationStrategy`
        - a tuple, typed as tuple[callable, dict[str, str]]
    A public property exposes stratgies upon successul addition and validation.
    A `StrategyTypeError` is thrown, otherwise.
    """

    def __init__(self) -> None:
        self._strategies: StrategyMapCollection = collections.defaultdict(list)

    def add(
        self,
        attr_key: str,
        strategy: DeduplicationStrategy | tuple,
    ):
        """Adds a strategy to the collection under a specific attribute key.

        Validates the strategy before adding it to the collection. If the
        strategy is not valid, a `StrategyTypeError` is raised.

        Args:
            attr_key: The key representing the attribute the strategy applies
                to.
            strategy: The deduplication strategy or a tuple containing a
                callable and its associated keyword arguments, as a mapping

        Raises:
            StrategyTypeError: If the strategy is not valid according to
            validation rules.
        """
        if self.validate(strategy):
            self._strategies[attr_key].append(strategy)  # type: ignore[attr-defined]
            return
        raise StrategyTypeError(strategy)

    def get(self) -> StrategyMapCollection:
        return self._strategies

    def validate(self, strategy) -> bool:
        """
        Validates a strategy

        The strategy to validate. Can be a `DeduplicationStrategy`, a tuple, or
        a dict of the aforementioned strategies types i.e.
        dict[str, DeduplicationStrategy | tuple]. As such the function checks
        such dict instances via recursion.

        Args:
            strategy: The strategy to validate. `DeduplicationStrategy`, tuple,
            or a dict of such

        Returns:
            bool: strategy is | isn't valid

        A valid strategy is one of the following:
            - A `DeduplicationStrategy` instance.
            - A tuple where the first element is a callable and the second
                element is a dictionary.
            - A dictionary where each item is a valid strategy.
        """
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
        """Reset strategy collection to empty default dictionary"""
        self.__init__()


# EXCEPTION CLASS


class StrategyTypeError(Exception):
    """Strategy type not valid errors"""

    def __init__(self, strategy: DeduplicationStrategy | tuple):
        base_msg = "Input is not valid"  # i.e. default
        context = ""
        if inspect.isclass(strategy):
            base_msg = "Input class is not valid: must be an instance of `DeduplicationStrategy`"
            context = f"not: {type(strategy())}"
        if isinstance(strategy, tuple):
            base_msg = "Input tuple is not valid: must be a length 2 [callable, dict]"
            context = f"not: {strategy}"
        if isinstance(strategy, dict):
            base_msg = "Input dict is not valid: items must be a list of `DeduplicationStrategy` or tuples"
            context = ""
        super().__init__(base_msg + context)


# WRAP DATAFRAME DISPATCHER:


@singledispatch
def _wrap(df: DataFrame, id: str | None = None) -> WrappedDataFrame:
    """
    Dispatch the dataframe to the appropriate wrapping handler.

    Args:
        df: The dataframe to dispatch to the appropriate handler.

    Returns:
        WrappedDataFrame, a DataFrame wrapped with a uniform interface.

    Raises:
        NotImplementedError
    """
    del id # Unused
    raise NotImplementedError(f"Unsupported data frame: {type(df)}")


@_wrap.register(pd.DataFrame)
def _(df, id: str | None = None):
    del id  # TODO implement
    return WrappedPandasDataFrame(df)


@_wrap.register(pl.DataFrame)
def _(df, id: str | None = None):
    del id  # TODO implement
    return WrappedPolarsDataFrame(df)


@_wrap.register(SparkDataFrame)
def _(df, id: str | None = None):
    del id  # TODO implement
    return WrappedSparkDataFrame(df)


@_wrap.register(list)
def _(df, id: str):
    return WrappedSparkRows(df, id)
