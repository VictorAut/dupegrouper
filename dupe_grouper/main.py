import data
import deduplication

from base import DupeGrouper

df1 = data.df3

dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=0.05))
dg.add_strategy(deduplication.TfIdf(tolerance=0.7))

dg.dedupe("address")

dg.strategies

strategies = {
    "email": {
        (deduplication.Exact, {}),
        (deduplication.Fuzzy, {"tolerance": 0.05}),
        (deduplication.TfIdf, {"tolerance": 0.5, "ngram": 3, "topn": 4}),
    }
}


