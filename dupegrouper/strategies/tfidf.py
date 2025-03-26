import functools
import logging
from typing_extensions import override
import typing

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sparse_dot_topn import sp_matmul_topn  # type: ignore

from dupegrouper.definitions import TMP_ATTR_LABEL, frames
from dupegrouper.frame import DFMethods
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# TFIDF:


class TfIdf(DeduplicationStrategy):

    def __init__(
        self,
        ngram: int | tuple[int, int] = 3,
        tolerance: float = 0.05,
        topn: int = 2,
        **kwargs,
    ):
        self._ngram = ngram
        self._tolerance = tolerance
        self._topn = topn
        self._kwargs = kwargs

    @functools.singledispatchmethod
    def _vectorize(self, ngram) -> TfidfVectorizer:
        del ngram  # Unused by generic
        return TypeError("ngram must be of type int or a length 2 tuple of integers")

    @_vectorize.register(int)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(ngram, ngram),
            **self._kwargs,
        )

    @_vectorize.register(tuple)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram,
            **self._kwargs,
        )

    def _get_vectorizer(self):
        return self._vectorize(self._ngram)

    def _get_similarities_matrix(
        self,
        vectorizer: TfidfVectorizer,
        array: np.ndarray,
        /,
    ) -> csr_matrix:
        return sp_matmul_topn(
            (mat := vectorizer.fit_transform(array)),
            mat.T,
            top_n=self._topn,
            threshold=1 - self._tolerance,
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
        seen: set[tuple[int, int]] = set()

        # iter: reorder to similarities score ascending
        for i, j, _ in zip(*map(np.flip, matches)):

            # not inversed; not repeated
            if {(i, j), (j, i)}.isdisjoint(seen):
                seen.add((i, j))
                yield {i: j}

    @override
    def dedupe(self, attr: str, /) -> frames:
        logger.debug(
            f'Deduping attribute "{attr}" with {self.__class__.__name__}('
            f"ngram={self._ngram}, "
            f"tolerance={self._tolerance}, "
            f"topn={self._topn}"
            ")"
        )

        frame_methods: DFMethods = self.frame_methods

        tmp_attr: str = attr + TMP_ATTR_LABEL

        vectorizer = self._get_vectorizer()

        similarities = self._get_similarities_matrix(
            vectorizer, frame_methods.get_col(attr)
        )

        matches = self._get_matches_array(
            similarities, np.array(frame_methods.get_col(attr))
        )

        logger.debug(
            f'Assigning duplicated "{attr}" instances to attribute "{tmp_attr}"'
        )
        for tfidf_map in self._gen_map(matches):

            attr_map = frame_methods.map_dict(attr, tfidf_map)  # i.e. "Series" like

            new_attr = frame_methods.fill_na(
                attr_map, frame_methods.get_col(attr)
            )  # i.e. "Series" like

            frame_methods: DFMethods = frame_methods.put_col(tmp_attr, new_attr)  # type: ignore[no-redef]

            frame_methods: DFMethods = self._assign_group_id(tmp_attr).drop_col(  # type: ignore[no-redef]
                tmp_attr
            )

        df: frames = frame_methods.frame
        return df
