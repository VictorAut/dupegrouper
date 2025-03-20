import functools
import typing
from typing_extensions import override

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from strategy import DeduplicationStrategy


# TFIDF:


class TfIdf(DeduplicationStrategy):

    def __init__(
        self,
        ngram: int | tuple[int, int] = 3,
        tolerance: float = 0.05,
        topn: int = 4,
        **kwargs,
    ):
        self.ngram = ngram
        self.tolerance = tolerance
        self.topn = topn
        self.kwargs = kwargs

    @functools.singledispatchmethod
    def _vectorize(self, ngram) -> TfidfVectorizer:
        del ngram  # Unused by generic
        return TypeError("ngram must be of type int or a length 2 tuple of integers")

    @_vectorize.register(int)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(ngram, ngram),
            **self.kwargs,
        )

    @_vectorize.register(tuple)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram,
            **self.kwargs,
        )

    def _get_vectorizer(self):
        return self._vectorize(self.ngram)

    def _get_similarities_matrix(
        self,
        vectorizer: TfidfVectorizer,
        array: np.ndarray,
        /,
    ) -> csr_matrix:
        return sp_matmul_topn(
            (mat := vectorizer.fit_transform(array)),
            mat.T,
            top_n=self.topn,
            threshold=1 - self.tolerance,
            sort=True,
        )

    @staticmethod
    def _get_matches_array(
        sparse: csr_matrix,
        array: np.ndarray,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sparse_coo = sparse.tocoo()

        # floating point precision handling of perfect match filter
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

        # iter: reorder to similarities score ascending
        for i, j, _ in zip(*map(np.flip, matches)):

            # not inversed; not repeated
            if {(i, j), (j, i)}.isdisjoint(seen):
                seen.add((i, j))
                yield {i: j}

    @override
    def dedupe(self, df: pd.DataFrame, attr: str, /) -> pd.DataFrame:
        print(f"evaluating {self.__class__.__name__}")
        vectorizer = self._get_vectorizer()

        similarities = self._get_similarities_matrix(vectorizer, df[attr])

        matches = self._get_matches_array(similarities, df[attr])

        for tfidf_map in self._gen_map(matches):

            df = self._assign_group_id(df, df[attr].map(tfidf_map).fillna(df[attr]))

        return df


# import data

# df = data.df1


# deduper = TfIdf(ngram=3, tolerance=0.7)

# deduper.dedupe(df.reset_index(drop=True), "email")

# TODO check why these results are weird
# Also need to actually cast to arrays e.g. with .to_numpy()
# Also need to think about dfs that come in with unordered indexes for above but also in general
