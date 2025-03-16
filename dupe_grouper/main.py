import typing
import functools
import itertools
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd

df = pd.read_csv("mock_data.csv")
df["phone"] = df["phone"].astype("Int64").astype("string")
df["address_number"] = df["address_number"].astype("Int64").astype("string")


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

        self.df = assign_group_id(self.df, self.df[attr].map(fuzzy_map))

        return self


deduper = DupeGrouper(df)

deduper.exact_group("email").fuzz_group("email")

deduper.df


def assign_group_id(df: pd.DataFrame, attribute: str):

    return df.assign(
        group_id=df.groupby(attribute)["group_id"]
        .transform("first")
        .fillna(df["group_id"])
    )


df = pd.DataFrame(
    {
        "id": {44: 1, 55: 2, 31: 3, 9: 4, 6: 5, 12: 6, 89: 7, 19: 8, 76: 9, 7: 10},
        "group_id": {
            44: 1,
            55: 2,
            31: 3,
            9: 4,
            6: 5,
            12: 6,
            89: 7,
            19: 8,
            76: 9,
            7: 10,
        },
        "email": {
            44: "bbab@example.com",
            55: "bb@example.com",
            31: "a@example.com",
            9: "hellothere@example.com",
            6: "b@example.com",
            12: "bab@example.com",
            89: "b@example.com",
            19: "hellthere@example.com",
            76: "hellathere@example.com",
            7: "irrelevant@hotmail.com",
        },
    }
)

df = pd.DataFrame(
    {
        "id": range(1, 1001),
        "group_id": range(1, 1001),
        "email": np.random.choice(
            [
                "bbab@example.com",
                "b@example.com",
                "a@example.com",
                "hellothere@example.com",
                "bb@example.com",
                "bab@example.com",
                "bb@example.com",
                "hellthere@example.com",
                "heythere@example.com",
                "irrelevant@hotmail.com",
            ],
            1000,
        ),
    }
)

df = df.reset_index(drop=True)

df = assign_group_id(df, "email")

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"TIMING: function: {f.__name__}, time: {te - ts}")
        return result

    return wrap
