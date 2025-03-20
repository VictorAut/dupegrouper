import functools
from typing_extensions import override

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from strategy import DeduplicationStrategy


# FUZZY:


class Fuzzy(DeduplicationStrategy):

    def __init__(self, tolerance: float = .05):
        self.ratio = 100 * (1 - tolerance)

    @staticmethod
    @functools.cache
    def _fuzz_ratio(s1, s2):
        return fuzz.ratio(s1, s2)

    @override
    def dedupe(self, df: pd.DataFrame, attr: str, /) -> pd.DataFrame:
        print(f"evaluating {self.__class__.__name__}")
        uattrs = df[attr].unique()

        similarity_matrix = np.array(
            [[self._fuzz_ratio(s1, s2) for s1 in uattrs] for s2 in uattrs]
        )

        match_indices = np.where(similarity_matrix >= self.ratio)

        fuzzy_map = {uattrs[i]: uattrs[j] for i, j in zip(*match_indices)}

        return self._assign_group_id(df, df[attr].map(fuzzy_map))