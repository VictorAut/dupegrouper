"""constants and types"""

from __future__ import annotations
import os
import typing

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import (
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    FloatType,
    BooleanType,
    TimestampType,
    DateType,
)

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


DataFrame: typing.TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame"  # | ...


# PYSPARK SQL TYPES TO CLASS TYPE CONVERSION

PYSPARK_TYPES = {
    "string": StringType(),
    "int": IntegerType(),
    "bigint": LongType(),
    "double": DoubleType(),
    "float": FloatType(),
    "boolean": BooleanType(),
    "timestamp": TimestampType(),
    "date": DateType(),
    # Add others as needed
}
