from abc import ABC, abstractmethod

import pandas as pd

class DeduplicationStrategy(ABC):

    @staticmethod
    def _assign_group_id(df: pd.DataFrame, attr: str):
        return df.assign(
            group_id=df.groupby(attr)["group_id"]
            .transform("first")
            .fillna(df["group_id"])
        )

    @abstractmethod
    def dedupe(self, df: pd.DataFrame, attr: str) -> pd.DataFrame:
        pass
