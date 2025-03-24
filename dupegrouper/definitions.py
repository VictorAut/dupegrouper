from __future__ import annotations
import collections.abc
import os
import typing

import pandas as pd
import polars as pl

if typing.TYPE_CHECKING:
    from dupegrouper.strategy import DeduplicationStrategy


# CONSTANTS


GROUP_ID: typing.Final[str] = os.environ.get("GROUP_ID", "group_id")
TMP_ATTR_LABEL: typing.Final[str] = os.environ.get("TMP_ATTR_LABEL", "__tmp_attr")


# TYPES:

strategy_list_item: typing.TypeAlias = "DeduplicationStrategy | tuple[typing.Callable, dict[str, str]]"

strategy_list_collection = list[strategy_list_item]

strategy_map_collection = collections.abc.Mapping[
    str,
    tuple[
        strategy_list_item,
        ...,
    ],
]


frames = pd.DataFrame | pl.DataFrame  # | ...