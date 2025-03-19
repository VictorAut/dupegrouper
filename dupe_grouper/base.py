import pandas as pd

from strategy import DeduplicationStrategy


# BASE:


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
