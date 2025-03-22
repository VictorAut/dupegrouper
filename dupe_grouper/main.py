import polars as pl
import pandas as pd

import data
import deduplication

from base import DupeGrouper, strategies_map

from custom import my_func


# pandas

df = pd.read_csv("multi_df.csv")

dg = DupeGrouper(df)

dg.df

# polars

df  = pl.read_csv("multi_df.csv")

dg = DupeGrouper(df)

dg.df








######################
import polars as pl
import pandas as pd
import data
import deduplication

from base import DupeGrouper, strategies_map

from custom import my_func

# df = data.df4
# df  = pl.read_csv("multi_df.csv")
df  = pd.read_csv("multi_df.csv")

dg = DupeGrouper(df)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=0.3))
dg.add_strategy(deduplication.TfIdf(tolerance=0.6))
dg.add_strategy((my_func, {'match_str': "london"}))

dg.dedupe("address")

# dg.strategies

# dg.report

dg.df

######################

df1 = data.df3

dg = DupeGrouper(df1)

strategies: strategies_map = {
    "address": (
        deduplication.Exact(),
        deduplication.Fuzzy(tolerance=0.2),
        deduplication.TfIdf(tolerance=0.7, ngram=3, topn=4),
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
