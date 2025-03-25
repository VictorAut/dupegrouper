from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
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

    @singledispatchmethod
    def __init__(self, df: frames):
        raise NotImplementedError(f"Not supported for dataframe of type {type(df)}")

    @__init__.register(pd.DataFrame)
    def _(self, df):
        self.df = PandasMethods(df)
        # self.df = df

    @__init__.register(pl.DataFrame)
    def _(self, df):
        self.df = PolarsMethods(df)
        # self.df = df


# class _DataFrameDispatcher:


#     @singledispatchmethod
#     def _put_col(self, df: frames, column: str, array):
#         del column, array  # Unused
#         raise NotImplementedError(
#             f"No create column series method supported for {type(df)}"
#         )

#     @_put_col.register(pd.DataFrame)
#     def _(self, df, column, array):
#         return df.assign(**{column: array})

#     @_put_col.register(pl.DataFrame)
#     def _(self, df, column, array):
#         return df.with_columns(**{column: array})

#     @singledispatchmethod
#     def _get_col(self, df: frames, column: str):
#         del column  # Unused
#         raise NotImplementedError(
#             f"No create column series method supported for {type(df)}"
#         )

#     @_get_col.register(pd.DataFrame)
#     @_get_col.register(pl.DataFrame)
#     def _(self, df, column):
#         return df[column]

#     @singledispatchmethod
#     def _map_dict(self, df: frames, column: str, mapping: dict):
#         del column, mapping  # Unused
#         raise NotImplementedError(
#             f"No create column series method supported for {type(df)}"
#         )

#     @_map_dict.register(pd.DataFrame)
#     def _(self, df, column, mapping) -> pd.Series:
#         return self._get_col(df, column).map(mapping)

#     @_map_dict.register(pl.DataFrame)
#     def _(self, df, column, mapping) -> pl.Series:
#         return self._get_col(df, column).replace(mapping)

#     @singledispatchmethod
#     def _drop_col(self, df: frames, column: str):
#         del column  # Unused
#         raise NotImplementedError(
#             f"No create column series method supported for {type(df)}"
#         )

#     @_drop_col.register(pd.DataFrame)
#     def _(self, df, column):
#         return df.drop(columns=column)

#     @_drop_col.register(pl.DataFrame)
#     def _(self, df, column):
#         return df.drop(column)  # i.e. positional only

#     @singledispatchmethod
#     def _fill_na(self, series, array):
#         del array  # Unused
#         raise NotImplementedError(
#             f"No create column series method supported for {type(series)}"
#         )

#     @_fill_na.register(pd.Series)
#     def _(self, series, array):
#         return series.fillna(array)

#     @_fill_na.register(pl.Series)
#     def _(self, series, array):
#         return series.fill_null(array)


# STRATEGY:


class DeduplicationStrategy(ABC):

    _tally: dict[int | str, list[int]] = defaultdict(list)

    def set_df(self, df):
        self._df = _DataFrameDispatcher(df)

    def _update_tally(
        self,
        id,
        old_group_id,
        new_group_id,
    ):
        if not (key := self._tally[id]):
            return key.append(int(old_group_id))
        if new_group_id != old_group_id:
            return key.append(int(new_group_id))

    def _assign_group_id(self, attr: str):
        logger.debug(
            f'Re-assigning new group_id per duped instance of attribute "{attr}"'
        )
        # ids = np.asarray(self._get_col(df, "id")) TODO
        attrs = np.asarray(self._df._get_col(attr))
        groups = np.asarray(self._df._get_col(GROUP_ID))

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

        # np.vectorize(self._update_tally)(ids, groups, new_groups) TODO

        self._df._put_col(GROUP_ID, new_groups)

        return self._df

    @abstractmethod
    def dedupe(self, attr: str) -> pd.DataFrame:
        pass
