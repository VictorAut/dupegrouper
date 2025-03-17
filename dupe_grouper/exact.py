import pandas as pd

from strategy import DeduplicationStrategy

class Exact(DeduplicationStrategy):

    def dedupe(self, df: pd.DataFrame, attr: str, /):
        return self._assign_group_id(df, attr)