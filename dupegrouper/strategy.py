from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatchmethod
import logging

import numpy as np
import pandas as pd
import polars as pl

from dupegrouper.definitions import GROUP_ID, frames
from dupegrouper.dataframes import PandasMethods, PolarsMethods


# LOGGER:


logger = logging.getLogger(__name__)


# DATAFRAME DISPATCHER


class _DataFrameDispatcher:

    def __init__(self, df):
        self._frame_methods = self._df_dispatch(df)

    @singledispatchmethod
    @staticmethod
    def _df_dispatch(df: frames):
        raise NotImplementedError(f"Unsupported data frame: {type(df)}")

    @_df_dispatch.register(pd.DataFrame)
    def _(self, df):
        return PandasMethods(df)

    @_df_dispatch.register(pl.DataFrame)
    def _(self, df):
        return PolarsMethods(df)

    @property
    def frame_methods(self):
        return self._frame_methods


# STRATEGY:


class DeduplicationStrategy(ABC):

    def set_df(self, df):
        self.frame_methods = _DataFrameDispatcher(df).frame_methods
        return self

    def _assign_group_id(self, attr: str):
        logger.debug(
            f'Re-assigning new "group_id" per duped instance of attribute "{attr}"'
        )

        frame_methods = self.frame_methods  # type according to future ABC

        attrs = np.asarray(frame_methods.get_col(attr))
        groups = np.asarray(frame_methods.get_col(GROUP_ID))

        print(attrs)

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

        frame_methods = frame_methods.put_col(GROUP_ID, new_groups)

        return frame_methods

    @abstractmethod
    def dedupe(self, attr: str) -> frames:
        pass
