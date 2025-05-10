import pytest

from dupegrouper.base import _wrap
from dupegrouper.strategies.tfidf import TfIdf


def do_tfidf(df, tfidf_params, group_id):
    tfidf = TfIdf(**tfidf_params)
    tfidf.with_frame(_wrap(df))

    updated_wrapped_df = tfidf.dedupe("address")
    updated_df = updated_wrapped_df.unwrap()

    assert list(updated_df["group_id"]) == group_id


tfidf_parametrize_data = [
    # i.e. no deduping, by definition
    ({"ngram": (1, 1), "tolerance": 0, "topn": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive tolerance
    ({"ngram": (1, 1), "tolerance": 0.05, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"ngram": (1, 1), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    ({"ngram": (1, 1), "tolerance": 0.35, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 6]),
    ({"ngram": (1, 1), "tolerance": 0.50, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    ({"ngram": (1, 1), "tolerance": 0.65, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    ({"ngram": (1, 1), "tolerance": 0.85, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    # progressive ngram @ 0.2
    ({"ngram": (1, 2), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 2), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (3, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    # progressive ngram @ 0.4
    ({"ngram": (1, 2), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 2), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (3, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
]

# i.e. pandas


@pytest.mark.parametrize("tfidf_params, expected_group_id", tfidf_parametrize_data)
def test_tfidf_dedupe_pandas(tfidf_params, expected_group_id, df_pandas):
    do_tfidf(df_pandas, tfidf_params, expected_group_id)


# i.e. polars


@pytest.mark.parametrize("tfidf_params, expected_group_id", tfidf_parametrize_data)
def test_tfidf_dedupe_polars(tfidf_params, expected_group_id, df_polars):
    do_tfidf(df_polars, tfidf_params, expected_group_id)
