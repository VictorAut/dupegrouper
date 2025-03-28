"""constants and types"""

from __future__ import annotations
import os
import typing

import pandas as pd
import polars as pl

if typing.TYPE_CHECKING:
    from dupegrouper.strategy import DeduplicationStrategy


# CONSTANTS


# the group_id label in the dataframe
GROUP_ID: typing.Final[str] = os.environ.get("GROUP_ID", "group_id")

# the ethereal dataframe label created whilst deduplicating
TMP_ATTR_LABEL: typing.Final[str] = os.environ.get("TMP_ATTR_LABEL", "__tmp_attr")


# TYPES:

strategy_list_item: typing.TypeAlias = "DeduplicationStrategy | tuple[typing.Callable, dict[str, str]]"

strategy_map_collection = typing.DefaultDict[
    str,
    list[strategy_list_item],
]


frames = pd.DataFrame | pl.DataFrame  # | ...
