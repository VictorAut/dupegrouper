"""Defines methods after Polars API"""

from typing_extensions import override
import typing

import polars as pl

from dupegrouper.frames.dataframe import DFMethods


class PolarsMethods(DFMethods):

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.with_columns(**{column: array})  # type: ignore[operator]
        return self

    @override
    def get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)  # type: ignore[operator]

    @override
    def map_dict(self, column: str, mapping: dict) -> pl.Series:
        return self.get_col(column).replace_strict(mapping, default=None)

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(column)  # i.e. positional only
        return self

    @staticmethod
    @override
    def fill_na(series: pl.Series, array) -> pl.Series:  # type: ignore[override]
        return series.fill_null(array)

    @property
    @override
    def frame(self):
        return self._df
