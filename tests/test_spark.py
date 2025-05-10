"""This test module covers the same type of tests found in
- test_base.py
- test_strategy.py

But, collected in one place for the purpose of uniquely testing Spark cases.
"""

import pytest

from pyspark.sql import DataFrame


def spark_column_of_rows_as_list(df: DataFrame, column: str) -> list:
    return [value[column] for value in df.select(column).collect()]

def test_random(df_spark_raw):
    assert spark_column_of_rows_as_list(df_spark_raw, 'id') == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


