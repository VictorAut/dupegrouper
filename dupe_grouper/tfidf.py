import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

import data

from strategy import DeduplicationStrategy


df = data.df3

N = 3
tolerance = 0.7  # i.e. how much deviation

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(N, N))

similarities = sp_matmul_topn(
    (mat := vectorizer.fit_transform(df["address"])),
    mat.T,
    top_n=4,
    threshold=1 - tolerance,
    sort=True,
)


def get_matches_df(sparse, array):

    sparse_coo = sparse.tocoo()  # i.e. COO format

    mask = ~np.isclose(
        sparse_coo.data, 1.0
    )  # i.e. floating point precision handling for 'perfect' match filter

    rows, cols = sparse_coo.row[mask], sparse_coo.col[mask]

    n = len(rows)

    lookup = np.empty(n, dtype=object)
    match = np.empty(n, dtype=object)
    similarity = np.zeros(n)

    for i in range(0, n):

        lookup[i] = array[rows[i]]
        match[i] = array[cols[i]]
        similarity[i] = sparse[rows[i], cols[i]]

    return lookup, match, similarity


matches = get_matches_df(similarities, df["address"])

pd.DataFrame({"lookup": matches[0], "match": matches[1], "similarity": matches[2]})


def gen_map(matches: tuple[np.ndarray, np.ndarray, np.ndarray]):
    seen = set()
    for i, j, _ in zip(*matches):
        if {(i, j), (j, i)}.isdisjoint(seen):  # i.e. not self-reference; not repeated
            seen.add((i, j))
            yield {i: j}


tf = gen_map(matches)

for kv in tf:
    print(kv)

tfidf_map = {i: j for i, j in zip(lookup, match)}


def _assign_group_id(df: pd.DataFrame, attr: str):
    return df.assign(
        group_id=df.groupby(attr)["group_id"].transform("first").fillna(df["group_id"])
    )


_assign_group_id(df, df["address"].map(tfidf_map))
