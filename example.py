import logging

import pandas as pd

import dupegrouper


######################


logging.basicConfig(
    level=logging.DEBUG,
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

df = pd.read_csv("multi_df.csv")

######################

dg = dupegrouper.DupeGrouper(df)


dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))
dg.add_strategy(dupegrouper.strategies.TfIdf(tolerance=0.6))
dg.add_strategy((my_func, {"match_str": "london"}))


dg.dedupe("address")

print(dg._strategy_manager.get())

dg.strategies

# print(dg.df)

######################


# dg.add_strategy(('poo',)) # this should not work
# dg.add_strategy({}) # this should not work

######################

df = pd.read_csv("multi_df.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.2),
        (my_func, {"match_str": "london"}),
    ],
    "email": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.TfIdf(tolerance=0.7, ngram=3, topn=4),
    ],
}

dg.add_strategy(strategies)

print(dg.strategies)

dg.dedupe()

# print(dg.df)
