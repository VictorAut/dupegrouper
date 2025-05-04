"""constants and types"""

from __future__ import annotations
import os
import typing

import pandas as pd
import polars as pl
import pyspark.sql as ps

if typing.TYPE_CHECKING:
    from dupegrouper.strategy import DeduplicationStrategy


# CONSTANTS


# the group_id label in the dataframe
GROUP_ID: typing.Final[str] = os.environ.get("GROUP_ID", "group_id")

# the ethereal dataframe label created whilst deduplicating
TMP_ATTR: typing.Final[str] = os.environ.get("TMP_ATTR", "__tmp_attr")


# TYPES:


StrategyMapCollection: typing.TypeAlias = typing.DefaultDict[
    str,
    list["DeduplicationStrategy | tuple[typing.Callable, dict[str, str]]"],
]


DataFrame: typing.TypeAlias = "pd.DataFrame | pl.DataFrame | ps.DataFrame"  # | ...
