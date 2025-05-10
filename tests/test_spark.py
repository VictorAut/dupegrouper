"""This test module covers the same type of tests found in
- test_base.py
- test_strategy.py

But, collected in one place for the purpose of uniquely testing Spark cases.
"""

"""Tests for dupegrouper.base"""

from pyspark.sql import DataFrame
import pytest


from dupegrouper.base import DupeGrouper

from dupegrouper.strategies import Exact


def spark_column_of_rows_as_list(df: DataFrame, column: str) -> list:
    return [value[column] for value in df.select(column).collect()]


def test_spark_1_partitions(df_spark_raw: DataFrame, spark):

    df_spark_raw = df_spark_raw.repartition(1, "blocking_key")

    strategies = {
        "address": [Exact()],
        "email": [Exact()],
    }
    dg = DupeGrouper(df_spark_raw, spark_session=spark, id="id")
    dg.add_strategy(strategies)
    dg.dedupe()

    assert spark_column_of_rows_as_list(dg.df, "group_id") == [1, 2, 3, 4, 5, 6, 5, 8, 1, 1, 11, 12, 13]


def test_spark_2_partitions(df_spark_raw: DataFrame, spark):

    df_spark_raw = df_spark_raw.repartition(2, "blocking_key")

    strategies = {
        "address": [Exact()],
        "email": [Exact()],
    }
    dg = DupeGrouper(df_spark_raw, spark_session=spark, id="id")
    dg.add_strategy(strategies)
    dg.dedupe()

    assert spark_column_of_rows_as_list(dg.df, "group_id") == [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13]
