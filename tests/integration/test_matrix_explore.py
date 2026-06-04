"""Integration tests for exploratory duplicate-rate profiling."""

from __future__ import annotations

import pytest

import liken as lk


SUPPORTED_BACKENDS = {"pandas", "polars", "modin"}
THRESHOLDS = [0.5, 0.75]

# fmt: off

PARAMS = [
    (
        ["address", "email"],
        ["address", "email"],
        {
            "exact": {"address": 0.2, "email": 0.1},
            "0.5": {"address": 0.5, "email": 0.8},
            "0.75": {"address": 0.3, "email": 0.6},
        },
    ),
    (
        {"address": lk.tfidf(ngram=(1, 2))},
        ["address"],
        {
            "exact": {"address": 0.2},
            "0.5": {"address": 0.6},
            "0.75": {"address": 0.4},
        },
    ),
]

IDS = [
    "list-default-fuzzy",
    "dict-tfidf",
]

# fmt: on


@pytest.mark.parametrize("columns, expected_columns, expected_rates", PARAMS, ids=IDS)
def test_matrix_explore(columns, expected_columns, expected_rates, dataframe, helpers):
    if helpers.backend not in SUPPORTED_BACKENDS:
        pytest.skip("explore currently supports pandas, polars and modin")

    result = lk.dedupe(dataframe).explore(columns, thresholds=THRESHOLDS)

    assert _columns(result, helpers.backend) == expected_columns
    assert _metrics(result, helpers.backend) == list(expected_rates)

    for metric, rates in expected_rates.items():
        for column, expected_rate in rates.items():
            assert _rate(result, helpers.backend, metric, column) == pytest.approx(expected_rate)


def _columns(df, backend):
    if backend == "polars":
        return [col for col in df.columns if col != "metric"]

    return list(df.columns)


def _metrics(df, backend):
    if backend == "polars":
        return df.get_column("metric").to_list()

    return list(df.index)


def _rate(df, backend, metric, column):
    if backend == "polars":
        return df.filter(df["metric"] == metric).select(column).item()

    return df.loc[metric, column]
