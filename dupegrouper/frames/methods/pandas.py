"""Defines methods after Pandas API"""

from __future__ import annotations
from typing_extensions import override
import typing

import pandas as pd

from dupegrouper.frames.dataframe import DFMethods


class PandasMethods(DFMethods):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self._df: pd.DataFrame = df

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.assign(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    @override
    def map_dict(self, column: str, mapping: dict) -> pd.Series:
        return self.get_col(column).map(mapping)

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(columns=column)
        return self

    @staticmethod
    @override
    def fill_na(series: pd.Series, array) -> pd.Series:
        return series.fillna(array)

    @property
    @override
    def frame(self):
        return self._df
