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

# df = pd.read_csv("multi_df2.csv")


df = pd.DataFrame(
    columns=["id", "address", "email", "group_id"],
    data=[
        [1, "123ab, OL5 9PL, UK", "bbab@example.com", 1],
        [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "bb@example.com", 2],
        [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com", 3],
        [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com", 4],
        [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com", 5],
        [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com", 6],
        [7, "C. Ancho 49, 05687, Navarra", "b@example.com", 7],
        [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com", 8],
        [9, "123ab, OL5 9PL, UK", "hellathere@example.com", 9],
        [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com", 10],
        [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com", 11],
        [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com", 12],
        [13, "totally random non existant address", "fictitious@never.co.uk", 13],
    ],
)

# my_func(df, "address", match_str="navarra")

dg = dupegrouper.DupeGrouper(df)


# dg.add_strategy(dupegrouper.strategies.Exact())
# dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))
# dg.add_strategy(dupegrouper.strategies.TfIdf(tolerance=0.7, ngram=3, topn=3))
dg.add_strategy((my_func, {"match_str": "navarra"}))

dg.dedupe("address")

dg.df

dg.add_strategy(dupegrouper.strategies.Exact())
# dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.1))
# dg.add_strategy((my_func, {"match_str": "london"}))

dg.dedupe("email")

dg.df


######################

print(
    "------------------------------------------------------------Pandas, DICT like------------------------------------------------------------"
)

######################

df = pd.read_csv("multi_df2.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
        dupegrouper.strategies.TfIdf(tolerance=0.7, ngram=3, topn=3),
    ],
    "email": [
        dupegrouper.strategies.Exact(),
    ],
}

dg.add_strategy(strategies)

dg.dedupe()

dg.df


######################

print(
    "------------------------------------------------------------Polars, LIST like------------------------------------------------------------"
)

######################

df = pl.read_csv("multi_df2.csv")

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

df = pl.read_csv("multi_df2.csv")

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
