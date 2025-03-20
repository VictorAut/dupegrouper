from typing_extensions import override

import pandas as pd

from strategy import DeduplicationStrategy


# EXACT:


class Exact(DeduplicationStrategy):

    @override
    def dedupe(self, df: pd.DataFrame, attr: str, /) -> pd.DataFrame:
        print(f"evaluating {self.__class__.__name__}")
        return self._assign_group_id(df, attr)