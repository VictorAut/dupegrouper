"""Defines methods after Spark API"""

from __future__ import annotations
from typing_extensions import override
import typing

import numpy as np
from pyspark.sql import DataFrame, SparkSession, functions, Row

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframe import WrappedDataFrame


class WrappedSparkDataFrame(WrappedDataFrame):

    def __init__(self, df: DataFrame, spark: SparkSession):
        super().__init__(df)
        self._df: DataFrame = self._add_group_id(df, spark)
        self._spark = spark

    @staticmethod
    @override
    def _add_group_id(df, spark: SparkSession) -> DataFrame:
        # DATAFRAME
        # id_array = [i + 1 for i in range(df.count())]
        # new_rows = _new_rows_from_new_column(df, id_array)
        # return spark.createDataFrame(new_rows, df.columns + [GROUP_ID])

        return [Row(**{**row.asDict(), 'group_id': value}) for row, value in zip(df, list([i + 1 for i in range(len(df))]))]

    # SPARK API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        # DATAFRAME:
        # if column in self._df.columns:
        #     self._df = self._df.drop(column)
        # new_rows = _new_rows_from_new_column(self._df, array)
        # self._df = self._spark.createDataFrame(data=new_rows, schema=self._df.columns + [column])
        # return self

        # list of Rows
        array = [i.item() for i in array]
        self._df = [Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    @override
    def get_col(self, column: str) -> list:
        # DataFrame
        # return self._df.select(column).rdd.flatMap(lambda x: x).collect()
        # list of Rows
        return [row[column] for row in self._df]

    @override
    def map_dict(self, column: str, mapping: dict) -> list:
        # DATAFRMAE
        # Turn the dictionary into a Spark map literal
        # mapping_expr = functions.create_map(
        #     [
        #         functions.lit(x)
        #         #
        #         for pair in mapping.items()
        #         for x in pair
        #     ]
        # )

        # # retrieve map of column
        # return (
        #     self._df.withColumn("TMP_MAPPING", mapping_expr.getItem(functions.col(column)))
        #     .select("TMP_MAPPING")
        #     .rdd.flatMap(lambda x: x)
        #     .collect()
        # )  # "default" is still None
    
        return [mapping.get(row[column]) for row in self._df]

    @override
    def drop_col(self, column: str) -> typing.Self:
        # DataFrame:
        # self._df = self._df.drop(column)
        # return self
        # list of Rows
        self._df = [Row(**{k: v for k, v in row.asDict().items() if k != column}) for row in self._df]
        return self

    @staticmethod
    @override
    def fill_na(series: list, array) -> list:
        return [i[-1] if not i[0] else i[0] for i in zip(series, array)]

    # THIN TRANSPARENCY DELEGATION

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)


# SPARK HELPERS:


def _safe_convert_array(array: typing.Sequence | np.ndarray) -> list:
    return array.tolist() if isinstance(array, np.ndarray) else array


def _new_rows_from_new_column(df: DataFrame, array: typing.Sequence | np.ndarray) -> list[tuple]:
    return [
        tuple(list(row) + [new_val])
        for row, new_val in zip(
            df.collect(),
            _safe_convert_array(array),
        )
    ]


##############################################################################

# data = [
#     [1, "123ab, OL5 9PL, UK", "bbab@example.com"],
#     [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "bb@example.com"],
#     [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com"],
#     [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com"],
#     [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com"],
#     [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com"],
#     [7, "C. Ancho 49, 05687, Navarra", "b@example.com"],
#     [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com"],
#     [9, "123ab, OL5 9PL, UK", "hellathere@example.com"],
#     [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com"],
#     [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com"],
#     [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com"],
#     [13, "totally random non existant address", "fictitious@never.co.uk"],
# ]

# columns = ["id", "address", "email"]

# spark = (
#     SparkSession.builder.master("local[1]")
#     .appName("local-tests")
#     .config("spark.executor.cores", "1")
#     .config("spark.executor.instances", "1")
#     .config("spark.sql.shuffle.partitions", "1")
#     .config("spark.driver.bindAddress", "127.0.0.1")
#     .getOrCreate()
# )

# df = spark.createDataFrame(data=data, schema=columns)

# wdf = WrappedSparkDataFrame(df, spark)
# wdf.show()
