import logging

import pandas as pd
import polars as pl

import dupegrouper


######################


logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# logger = logging.getLogger("testing_dupegrouper")
# logger.info('test string')

######################


def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    my_map = {}
    for irow, _ in df.iterrows():
        left: str = df.at[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right: str = df.at[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break
    return my_map


######################

print(
    "------------------------------------------------------------Pandas, LIST like------------------------------------------------------------"
)

######################

df = pd.read_csv("multi_df.csv")

dg = dupegrouper.DupeGrouper(df)


dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))
dg.add_strategy(dupegrouper.strategies.TfIdf(tolerance=0.4, ngram=3, topn=3))
dg.add_strategy((my_func, {"match_str": "london"}))

print(dg.strategies)

dg.dedupe("address")

print(dg.strategies)

print(dg.df)

######################

print(
    "------------------------------------------------------------Pandas, DICT like------------------------------------------------------------"
)

######################

df = pd.read_csv("multi_df.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
        # (my_func, {"match_str": "london"}),
    ],
    "email": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.TfIdf(tolerance=0.7, ngram=3, topn=4),
    ],
}

dg.add_strategy(strategies)

print(dg.strategies)

dg.dedupe()

print(dg.strategies)

print(dg.df)

######################

print(
    "------------------------------------------------------------Polars, LIST like------------------------------------------------------------"
)

######################

df = pl.read_csv("multi_df.csv")

df

dg = dupegrouper.DupeGrouper(df)


dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))
dg.add_strategy(dupegrouper.strategies.TfIdf(tolerance=0.4, ngram=3, topn=3))
# dg.add_strategy((my_func, {"match_str": "london"}))

print(dg.strategies)

dg.dedupe("address")

print(dg.strategies)

print(pd.DataFrame(dg.df))

######################

print(
    "------------------------------------------------------------Polars, DICT like------------------------------------------------------------"
)

######################

df = pl.read_csv("multi_df.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
        # (my_func, {"match_str": "london"}),
    ],
    "email": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.TfIdf(tolerance=0.7, ngram=3, topn=4),
    ],
}

dg.add_strategy(strategies)

print(dg.strategies)

dg.dedupe()

print(dg.strategies)

print(pd.DataFrame(dg.df))
