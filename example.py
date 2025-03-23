import pandas as pd

import dupegrouper
from dupegrouper.strategies import Exact, Fuzzy, TfIdf

######################


def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    my_map = {}
    for irow, _ in df.iterrows():
        left = df.loc[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right = df.loc[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break
    return my_map


######################

df = pd.read_csv("multi_df.csv")

######################

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy(Exact())
dg.add_strategy(Fuzzy(tolerance=0.3))
dg.add_strategy(TfIdf(tolerance=0.6))
# dg.add_strategy((my_func, {"match_str": "london"}))

dg.dedupe("address")

print(dg.df)

######################

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": (
        Exact(),
        Fuzzy(tolerance=0.2),
        (my_func, {"match_str": "london"}),
    ),
    "email": (
        Exact(),
        TfIdf(tolerance=0.7, ngram=3, topn=4),
    ),
}

dg.add_strategy(strategies)

print(dg.strategies)

dg.dedupe()

print(dg.df)
