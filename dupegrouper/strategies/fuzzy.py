import functools
from typing_extensions import override

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from dupegrouper.strategy import DeduplicationStrategy, TMP_ATTR_LABEL


# FUZZY:


class Fuzzy(DeduplicationStrategy):

    def __init__(self, tolerance: float = 0.05):
        self._ratio = 100 * (1 - tolerance)

    @staticmethod
    @functools.cache
    def _fuzz_ratio(s1, s2):
        return fuzz.ratio(s1, s2)

    @override
    def dedupe(self, df: pd.DataFrame, attr: str, /) -> pd.DataFrame:
        print(f"evaluating {self.__class__.__name__}")

        tmp_attr: str = TMP_ATTR_LABEL

        uattrs = np.unique(self._get_col(df, attr))

        similarity_matrix = np.array([[self._fuzz_ratio(s1, s2) for s1 in uattrs] for s2 in uattrs])

        match_indices = np.where(similarity_matrix >= self._ratio)

        fuzzy_map = {uattrs[i]: uattrs[j] for i, j in zip(*match_indices)}

        attr_map = self._map_dict(df, attr, fuzzy_map)

        df = self._put_col(df, tmp_attr, attr_map)

        return self._drop_col(self._assign_group_id(df, tmp_attr), tmp_attr)
