import typing

import polars as pl


class PolarsMethods:

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.with_columns(**{column: array})
        return self

    def get_col(self, column: str) -> pl.Series:
        return self._df[column]

    def map_dict(self, column: str, mapping: dict) -> pl.Series:
        return self.get_col(column).replace(mapping)

    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(column)  # i.e. positional only
        return self

    @staticmethod
    def fill_na(series: pl.Series, array) -> pl.Series:
        return series.fill_null(array)

    @property
    def frame(self):
        return self._df
