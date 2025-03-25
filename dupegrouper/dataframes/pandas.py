import pandas as pd


class PandasMethods:

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _put_column(self, column: str, array):
        self._df = self._df.assign(**{column: array})

    def _get_col(self, column: str):
        return self._df[column]

    def _map_dict(self, column: str, mapping: dict) -> pd.Series:
        return self._get_col(self._df, column).map(mapping)

    def _drop_col(self, column: str):
        self._df = self._df.drop(columns=column)

    @staticmethod
    def _fill_na(series: pd.Series, array):
        return series.fillna(array)
