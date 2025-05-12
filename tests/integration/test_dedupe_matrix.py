"""
Full integration test for each backend wrapper and each strategy via
a cartesian product of (backend X strategy). For respective lower-level tests
of backend wrappers and deduplication strategies, please see unit tests.
"""

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
import pytest

from dupegrouper import DupeGrouper
from dupegrouper.strategies import exact, fuzzy, tfidf


STRATEGY_CLASSES = (
    exact.Exact(),
    # fuzzy.Fuzzy(tolerance=0.45),
    # tfidf.TfIdf(ngram=(1, 1), tolerance=0.20, topn=2),
)

EXPECTED_GROUP_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13],
    [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 11, 12, 13],
    [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13],
)


@pytest.fixture(
    params=[
        # "pandas",
        # "polars",
        "spark",
    ]
)
def dataframe(request, df_pandas_raw, df_polars_raw, df_spark_raw, spark) -> tuple:
    """return a tuple of positionally ordered input parameters of DupeGrouper
    i.e.
        - df
        - spark_session
        - id
    """
    match request.param:
        case "pandas":
            return df_pandas_raw, None, None
        case "polars":
            return df_polars_raw, None, None
        case "spark":
            return df_spark_raw, spark, "id"


def get_group_id_as_list(df):
    if isinstance(df, pd.DataFrame | pl.DataFrame):
        return list(df["group_id"])
    if isinstance(df, SparkDataFrame):
        return [value["group_id"] for value in df.select("group_id").collect()]


@pytest.mark.parametrize(
    "strategy, expected_group_id",
    zip(STRATEGY_CLASSES, EXPECTED_GROUP_ID),
    ids=[instance.__class__.__name__ for instance in STRATEGY_CLASSES],
)
def test_dedup_matrix(strategy, expected_group_id, dataframe):

    df, spark_session, id = dataframe

    dg = DupeGrouper(df=df, spark_session=spark_session, id=id)

    # single strategy item addition
    # dg.add_strategy(strategy)
    # dg.dedupe("address")
    # assert get_group_id_as_list(dg.df) == expected_group_id

    # dictionary straegy addition
    dg.add_strategy({"address": [strategy]})
    dg.dedupe()
    assert get_group_id_as_list(dg.df) == expected_group_id


# def test_spark_single(df_spark_raw, spark):

#     strategies = {"address": [exact.Exact()]}
#     dg = DupeGrouper(df_spark_raw, spark_session=spark, id="id")
#     dg.add_strategy(strategies)
#     dg.dedupe()

#     assert get_group_id_as_list(dg.df) == [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]
