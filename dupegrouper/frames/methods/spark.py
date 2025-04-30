"""Defines methods after Polars API"""

from __future__ import annotations
from typing_extensions import override
import typing

from pyspark.sql import DataFrame

from dupegrouper.frames.dataframe import DataFrameContainer


class SparkMethods(DataFrameContainer):

    def __init__(self, df: DataFrame):
        super().__init__(df)
        self._df: DataFrame = df

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.withColumn(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> Series:
        return self._df.get_column(column)

    @override
    def map_dict(self, column: str, mapping: dict) -> Series:
        return self.get_col(column).replace_strict(mapping, default=None)

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(column)  # i.e. positional only
        return self

    @staticmethod
    @override
    def fill_na(series: Series, array) -> Series:
        return series.fill_null(array)

    @property
    @override
    def frame(self):
        return self._df

data = [
        [1, "123ab, OL5 9PL, UK", "bbab@example.com"],
        [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "bb@example.com"],
        [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com"],
        [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com"],
        [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com"],
        [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com"],
        [7, "C. Ancho 49, 05687, Navarra", "b@example.com"],
        [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com"],
        [9, "123ab, OL5 9PL, UK", "hellathere@example.com"],
        [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com"],
        [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com"],
        [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com"],
        [13, "totally random non existant address", "fictitious@never.co.uk"],
    ]

schema=["id", "address", "email"]


from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit

spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

df = spark.createDataFrame(data=data, schema=schema)

df.withColumn()