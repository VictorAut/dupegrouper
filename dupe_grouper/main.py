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

    def __init__(self, df):
        self.df = df
        df["group_id"] = range(1, len(df) + 1)

    def group_exact(self, df, attribute: str):
        pass


deduper = DupeGrouper(df)

df["group_id"] = range(1, len(df) + 1)


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


# @functools.cache
def _fuzz_ratio(val1, val2):
    return fuzz.ratio(val1, val2)


@timing
def fuzz_group(
    df: pd.DataFrame,
    attr: str,
    /,
    *,
    lock_group: bool = True,
    tolerance: float = 0.05,
):

    df = df.copy()  # TODO remove and WTF

    ratio = 100 * (1 - tolerance)

    deduped_ids = set()

    uattrs = df[attr].unique()

    similarity_matrix = {
        (e1, e2): _fuzz_ratio(e1, e2) for e1 in uattrs for e2 in uattrs
    }

    for val in uattrs:

        df["_fuzzy_match"] = None

        lock_mask = ~df["group_id"].isin(deduped_ids) if lock_group else slice(None)

        df.loc[lock_mask, "_fuzzy_match"] = df.loc[lock_mask, attr].map(
            lambda row: val if similarity_matrix.get((val, row), 0) > ratio else None
        )

        df = assign_group_id(df, "_fuzzy_match")

        if len(matches := df.dropna(subset="_fuzzy_match")) > 1:
            deduped_ids.add(int(matches.reset_index(drop=True).at[0, "group_id"]))

    return df.astype({"group_id": int}).drop(columns="_fuzzy_match")


@timing
def fuzz_group2(
    df: pd.DataFrame,
    attr: str,
    /,
    *,
    tolerance: float = 0.05,
):
    ratio = 100 * (1 - tolerance)

    uattrs = df[attr].unique()

    similarity_matrix = np.array(
        [[_fuzz_ratio(s1, s2) for s1 in uattrs] for s2 in uattrs]
    )

    match_indices = np.where(similarity_matrix >= ratio)

    fuzzy_map = {uattrs[i]: uattrs[j] for i, j in zip(*match_indices)}

    df.loc[:, "_fuzzy_match"] = df[attr].map(fuzzy_map)

    return assign_group_id(df, "_fuzzy_match").drop(columns="_fuzzy_match")


a = fuzz_group(df, "email", lock_group=False)
b = fuzz_group2(df, "email")

a.equals(b)
