"""Abstract base class for any deduplication atrategy

This module contains `DeduplicationStrategy` which provides `assign_group_id`,
which is at the core functionality of `dupegrouper` and is used for any
deduplication that requires _grouping_. Additionally, the overrideable `dedupe`
is defined."""

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatchmethod
import logging

import numpy as np
import pandas as pd
import polars as pl

from dupegrouper.definitions import GROUP_ID, frames
from dupegrouper.frame.methods import PandasMethods, PolarsMethods
from dupegrouper.frame import DFMethods


# LOGGER:


logger = logging.getLogger(__name__)


# DATAFRAME DISPATCHER


class _DataFrameDispatcher:
    """Dispatcher to collect methods for the given dataframe"""

    def __init__(self, df):
        self._frame_methods: DFMethods = self._df_dispatch(df)

    @singledispatchmethod
    @staticmethod
    def _df_dispatch(df: frames) -> DFMethods:
        """
        Dispatch the dataframe to the appropriate handler.

        Args:
            df: The dataframe to dispatch to the appropriate handler.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(f"Unsupported data frame: {type(df)}")

    @_df_dispatch.register(pd.DataFrame)
    def _(self, df) -> DFMethods:
        return PandasMethods(df)

    @_df_dispatch.register(pl.DataFrame)
    def _(self, df) -> DFMethods:
        return PolarsMethods(df)

    @property
    def frame_methods(self) -> DFMethods:
        return self._frame_methods


# STRATEGY:


class DeduplicationStrategy(ABC):
    """Abstract base class for defining deduplication strategies.

    This class provides the structure and methods for implementing different
    deduplication strategies. _Any_ deduplication strategy must inherit from 
    here.
    """

    def set_df(self, df):
        """assign the dataframe data and initializes the corresponding methods.

        Args:
            df: The dataframe to set

        Returns:
            self: i.e. allow for further chaining
        """
        self.frame_methods: DFMethods = _DataFrameDispatcher(df).frame_methods
        return self

    def assign_group_id(self, attr: str):
        """assign new group ids according to duplicated instances of attribute.

        Array-like contents of the dataframe's attributes are collected as a
        numpy array, along with the group id. unique instances are found, and
        the __first__ group id of that attribute is identified. This allows to
        then assign this "first" group id to all subsequent instances of a
        given unique attribute thus "flooring" the group ids.

        This implementation is akin to

            df.groupby(attr).transform("first").fill_null("group_id")

        Where the null backfill is implemented to handle instances where data
        in the attribute `attr` is incomplete — which happens in instances of
        iterative application of this function, or, when the function is
        applied to an attribute `attr` that contains only matches, i.e., a
        partial map of matches.
        
        Args:
            attr: the dataframe label of the attribute

        Returns:
            frame_methods; i.e. an instance `DFMethods` i.e. container of data
            **and** linked dataframe methods; ready for further downstream 
            processing.
        """
        logger.debug(
            f'Re-assigning new "group_id" per duped instance of attribute "{attr}"'
        )

        frame_methods: DFMethods = self.frame_methods  # type according to future ABC

        attrs = np.asarray(frame_methods.get_col(attr))
        groups = np.asarray(frame_methods.get_col(GROUP_ID))

        unique_attrs, unique_indices = np.unique(
            attrs,
            return_index=True,
        )

        first_groups = groups[unique_indices]

        attr_group_map = dict(zip(unique_attrs, first_groups))

        # iteratively: attrs -> value param; groups -> default param
        new_groups: np.ndarray = np.vectorize(
            lambda value, default: attr_group_map.get(
                value,
                default,
            )
        )(
            attrs,
            groups,
        )

        frame_methods: DFMethods = frame_methods.put_col(GROUP_ID, new_groups)  # type: ignore[no-redef]

        return frame_methods  # i.e. has `frame` and `methods`

    @abstractmethod
    def dedupe(self, attr: str) -> frames:
        """Method to define the deduplication logic for a specific strategy.

        This method must be implemented by subclasses to perform deduplication
        based on a specific attribute — and has been designed such that it will
        _always_ be a wrapper of `assign_group_id`

        Args:
            attr: The attribute to use for deduplication.

        Returns:
            frames: the deduplicated dataframe
        """
        pass
