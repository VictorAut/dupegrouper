import typing

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from strategy import DeduplicationStrategy


# TFIDF:


class TfIdf(DeduplicationStrategy):

    def __init__(self, ngram: int = 3, tolerance: float = 0.05):
        self.ngram = ngram
        self.tolerance = tolerance

    def _vectorize(self, **kwargs) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.ngram, self.ngram),
            **kwargs,
        )

    def _get_similarities_matrix(
        self,
        vectorizer: TfidfVectorizer,
        array: np.ndarray,
        /,
    ) -> csr_matrix:
        return sp_matmul_topn(
            (mat := vectorizer.fit_transform(array)),
            mat.T,
            top_n=4,
            threshold=1 - self.tolerance,
            sort=True,
        )

    @staticmethod
    def _get_matches_array(
        sparse: csr_matrix,
        array: np.ndarray,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        sparse_coo = sparse.tocoo()  # i.e. COO format

        # floating point precision handling for 'perfect' match filter
        mask = ~np.isclose(sparse_coo.data, 1.0)

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

    @staticmethod
    def _gen_map(
        matches: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> typing.Iterator[dict[str, str]]:

        seen = set()

        # iter in ascending order of similarities
        for i, j, _ in zip(*map(np.flip, matches)):

            # not self-reference; not repeated
            if {(i, j), (j, i)}.isdisjoint(seen):
                seen.add((i, j))
                yield {i: j}

    def dedupe(self, df: pd.DataFrame, attr: str, /):
        print("deduping...")

        vectorizer = self._vectorize()

        similarities = self._get_similarities_matrix(vectorizer, df[attr])

        matches = self._get_matches_array(similarities, df[attr])

        tfidf_maps = self._gen_map(matches)

        print(type(tfidf_maps))

        for tfidf_map in tfidf_maps:
            print("gen")
            print(tfidf_map)
            df =  self._assign_group_id(df, df[attr].map(tfidf_map).fillna(df[attr]))

        return df


import data

df = data.df3

deduper = TfIdf()

dir(deduper)

deduper.dedupe(df, "address")