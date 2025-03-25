import logging
import typing
from typing_extensions import override

import pandas as pd

from dupegrouper.definitions import TMP_ATTR_LABEL
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# TYPES:


_T = typing.TypeVar("_T")


# CUSTOM:


class Custom(DeduplicationStrategy):

    def __init__(
        self,
        func: typing.Callable[..., dict[_T, _T]],
        df: pd.DataFrame,
        attr: str,
        /,
        **kwargs,
    ):
        self._func = func
        self._df = df
        self._attr = attr
        self._kwargs = kwargs

    @override
    def dedupe(self, df=None, attr=None) -> pd.DataFrame:
        del df, attr  # Unused: initialised as private equivalents
        logger.debug(
            f'Deduping attribute "{self._attr}" with {self._func.__name__}'
            f'({", ".join(f"{k}={v}" for k, v in self._kwargs.items())})'
        )

        tmp_attr: str = self._attr + TMP_ATTR_LABEL

        attr_map = self._map_dict(
            self._df,
            self._attr,
            self._func(
                self._df,
                self._attr,
                **self._kwargs,
            ),
        )

        logger.debug(
            f"Assigning duplicated {self._attr} instances to attribute {tmp_attr}"
        )
        self._df = self._put_col(self._df, tmp_attr, attr_map)

        df = self._drop_col(self._assign_group_id(self._df, tmp_attr), tmp_attr)

        logger.debug(f"Finished grouping dupes of attribute {self._attr}")
        return df
