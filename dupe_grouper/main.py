import pandas as pd
import data
import deduplication

from base import DupeGrouper, strategies_map

from custom import my_func

df1 = data.df3

dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=0.05))
dg.add_strategy(deduplication.TfIdf(tolerance=0.7))
dg.add_strategy((my_func, {'match_str': "london"}))

dg.dedupe("address")

dg.strategies

dg.df

######################

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
