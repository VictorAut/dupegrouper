import typing
from typing_extensions import override

import pandas as pd

from dupegrouper.strategy import DeduplicationStrategy, TMP_ATTR_LABEL


# TYPES:


_T = typing.TypeVar("_T")


# CUSTOM:


class Custom(DeduplicationStrategy):

    def __init__(
        self,
        func: typing.Callable[..., dict[_T, _T]],
        df: pd.DataFrame,
        attr: str,
        **kwargs,
    ):
        self._func = func
        self._df = df
        self._attr = attr
        self._kwargs = kwargs

    @override
    def dedupe(self) -> pd.DataFrame:
        print(f"evaluating {self.__class__.__name__}")

        tmp_attr: str = TMP_ATTR_LABEL

        attr_map = self._map_dict(
            self._df,
            self._attr,
            self._func(
                self._df,
                self._attr,
                **self._kwargs,
            ),
        )

        self._df = self._put_col(self._df, tmp_attr, attr_map)

        return self._drop_col(self._assign_group_id(self._df, tmp_attr), tmp_attr)
