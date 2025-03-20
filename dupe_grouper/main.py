import pandas as pd
import data
import deduplication

from base import DupeGrouper, strategies_map

df1 = data.df3

dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=0.05))
dg.add_strategy(deduplication.TfIdf(tolerance=0.7))

dg.dedupe("address")

dg.strategies

dg.df

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

df1 = data.df3

dg = DupeGrouper(df1)

strategies: strategies_map = {
    "address": (
        deduplication.Exact(),
        deduplication.Fuzzy(tolerance=0.05),
        deduplication.TfIdf(tolerance=0.5, ngram=(3, 3), topn=4),
        (my_func, {'match_str': "london"})
    )
}


dg.add_strategy(strategies)

dg.dedupe()

dg.df

# TODO deal with this with a specialised class
# dg.strategies = ['poo']

# dg.df

# dg.add_strategy('poo')

# df = data.df3
