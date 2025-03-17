import typing
import functools
import itertools
from time import time
from rapidfuzz import fuzz
import numpy as np
import pandas as pd
import helpers
import data


class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        df["group_id"] = range(1, len(df) + 1)

    @staticmethod
    def _assign_group_id(df: pd.DataFrame, attr: str):
        return df.assign(
            group_id=df.groupby(attr)["group_id"]
            .transform("first")
            .fillna(df["group_id"])
        )

    def exact_group(self, attr: str):
        self.df = self._assign_group_id(self.df, attr)
        return self

    @staticmethod
    @functools.cache
    def _fuzz_ratio(s1, s2):
        return fuzz.ratio(s1, s2)

    def fuzz_group(self, attr: str, /, *, tolerance: float = 0.05):
        ratio = 100 * (1 - tolerance)

        uattrs = self.df[attr].unique()

        similarity_matrix = np.array(
            [[self._fuzz_ratio(s1, s2) for s1 in uattrs] for s2 in uattrs]
        )

        match_indices = np.where(similarity_matrix >= ratio)

        fuzzy_map = {uattrs[i]: uattrs[j] for i, j in zip(*match_indices)}

        self.df = self._assign_group_id(self.df, self.df[attr].map(fuzzy_map))

        return self
    
    def tfidf_group(self):
        pass

    def lsh_group(self):
        pass