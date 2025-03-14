import typing
import functools
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
        "id": {44: 1, 55: 2, 31: 3, 9: 4, 6: 5, 89: 6, 19: 7, 76: 8, 7: 9},
        "group_id": {44: 1, 55: 2, 31: 3, 9: 4, 6: 5, 89: 6, 19: 7, 76: 8, 7: 9},
        "email": {
            44: "a@example.com",
            55: "b@example.com",
            31: "a@example.com",
            9: "hellothere@example.com",
            6: "bb@example.com",
            89: "bb@example.com",
            19: "hellthere@example.com",
            76: "heythere@example.com",
            7: "irrelevant@hotmail.com",
        },
    }
)

df = df.reset_index(drop=True)

df = assign_group_id(df, "email")


for i in range(0, len(df)):
    df["_attr"] = df["email"].at[i]
    df["score"] = df.apply(
        lambda row: fuzz.ratio(row["email"], row["_attr"]), axis=1
    )
    df["_attr"] = df.apply(
        lambda row: row["_attr"] if row["score"] > 95 else None, axis=1
    )
    print(df)
    print('------------------------------')
    df = assign_group_id(df, "_attr")
    df = df.astype({'group_id': int})
    print(df)
    print('------------------------------')
    print('------------------------------')
    print('------------------------------')

for i, val in enumerate(df['email']):
    print(val)
    df["_attr"] = df.apply(
        lambda row: val if fuzz.ratio(row["email"], val) > 95 else None, axis=1
    )
    print(df)
    print('------------------------------')
    df = assign_group_id(df, "_attr")
    df = df.astype({'group_id': int})
    print(df)
    print('------------------------------')
    print('------------------------------')
    print('------------------------------')

df
