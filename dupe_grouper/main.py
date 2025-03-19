import data
import deduplication

from base import DupeGrouper

df1 = data.df1

dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=.05))

dg.dedupe("email")

dg.df