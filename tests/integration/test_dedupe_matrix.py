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
    exact.Exact,
    fuzzy.Fuzzy,
    tfidf.TfIdf,
)

STRATEGY_PARAMS = (
    {},  # for exact
    {"tolerance": 0.45},  # for fuzzy
    {"ngram": (1, 1), "tolerance": 0.20, "topn": 2},  # for tfidf
)

EXPECTED_GROUP_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13],  # for exact
    [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 11, 12, 13],  # for fuzzy
    [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13],  # for tfidf
)


@pytest.mark.parametrize(
    "strategy_class, strategy_params, expected_group_id",
    zip(STRATEGY_CLASSES, STRATEGY_PARAMS, EXPECTED_GROUP_ID),
    ids=[cls.__name__ for cls in STRATEGY_CLASSES],
)
def test_dedupe_matrix(strategy_class, strategy_params, expected_group_id, dataframe, helpers):

    df, spark_session, id = dataframe

    dg = DupeGrouper(df=df, spark_session=spark_session, id=id)

    # single strategy item addition
    dg.add_strategy(strategy_class(**strategy_params))
    dg.dedupe("address")
    assert helpers.get_group_id_as_list(dg.df) == expected_group_id

    # dictionary straegy addition
    dg.add_strategy({"address": [strategy_class(**strategy_params)]})
    dg.dedupe()
    assert helpers.get_group_id_as_list(dg.df) == expected_group_id
