import typing

import pandas as pd


class PandasMethods:

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.assign(**{column: array})
        return self

    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    def map_dict(self, column: str, mapping: dict) -> pd.Series:
        return self.get_col(column).map(mapping)

    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(columns=column)
        return self

    @staticmethod
    def fill_na(series: pd.Series, array) -> pd.Series:
        return series.fillna(array)

    @property
    def frame(self):
        return self._df
