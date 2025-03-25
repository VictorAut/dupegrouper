import polars as pl


class PolarsMethods:

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _put_column(self, column: str, array):
        return self._df.with_columns(**{column: array})

    def _get_col(self, column: str):
        return self._df[column]

    def _map_dict(self, column: str, mapping: dict) -> pl.Series:
        self._get_col(self._df, column).replace(mapping)

    def _drop_col(self, column: str):
        return self._df.drop(column)  # i.e. positional only

    @staticmethod
    def _fill_na(series: pl.Series, array):
        return series.fill_null(array)
