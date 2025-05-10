"""Defines methods after Spark API"""

from __future__ import annotations
from typing_extensions import override
import typing

import numpy as np
from pyspark.sql import DataFrame, Row

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframe import WrappedDataFrame


class WrappedSparkDataFrame(WrappedDataFrame):

    def __init__(self, df: DataFrame):
        super().__init__(df)
        # self._df: DataFrame = self._add_group_id(df)


    @staticmethod
    @override
    def _add_group_id(df) -> DataFrame:
        return [
            Row(**{**row.asDict(), "group_id": value})
            #
            for row, value in zip(df, list([i + 1 for i in range(len(df))]))
        ]

    # SPARK API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    @override
    def get_col(self, column: str) -> list:
        return [row[column] for row in self._df]

    @override
    def map_dict(self, column: str, mapping: dict) -> list:
        return [mapping.get(row[column]) for row in self._df]

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = [Row(**{k: v for k, v in row.asDict().items() if k != column}) for row in self._df]
        return self

    @staticmethod
    @override
    def fill_na(series: list, array) -> list:
        return [i[-1] if not i[0] else i[0] for i in zip(series, array)]

    # THIN TRANSPARENCY DELEGATION

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)


class WrappedSparkRows(WrappedDataFrame):

    def __init__(self, df: list[Row], id: str):
        super().__init__(df)
        self._df: list[Row] = self._add_group_id(df, id)

    @staticmethod
    @override
    def _add_group_id(df, id: str) -> list[Row]:
        # Monotonic increase:
        # return [
        #     Row(**{**row.asDict(), GROUP_ID: value})
        #     for row, value in zip(df, list([i + 1 for i in range(len(df))]))
        # ]
        return [Row(**{**row.asDict(), GROUP_ID: row[id]}) for row in df]

    # SPARK API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    @override
    def get_col(self, column: str) -> list:
        return [row[column] for row in self._df]

    @override
    def map_dict(self, column: str, mapping: dict) -> list:
        return [mapping.get(row[column]) for row in self._df]

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = [Row(**{k: v for k, v in row.asDict().items() if k != column}) for row in self._df]
        return self

    @staticmethod
    @override
    def fill_na(series: list, array) -> list:
        return [i[-1] if not i[0] else i[0] for i in zip(series, array)]

    # THIN TRANSPARENCY DELEGATION

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)
