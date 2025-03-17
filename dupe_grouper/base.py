import pandas as pd

import data
import deduplication
from strategy import DeduplicationStrategy



df1 = data.df1



class DupeGrouper:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["group_id"] = range(1, len(df) + 1)
        self.strategies = []

    def add_strategy(self, strategy: DeduplicationStrategy):

        self.strategies.append(strategy)

    def dedupe(self, attr: str):

        for strategy in self.strategies:
            self.df = strategy.dedupe(self.df, attr)


dg = DupeGrouper(df1)

dg.add_strategy(deduplication.Exact())
dg.add_strategy(deduplication.Fuzzy(tolerance=.05))

dg.dedupe("email")

dg.df
