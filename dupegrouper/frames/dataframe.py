"""Abstract base class container"""

from __future__ import annotations
from abc import ABC, abstractmethod
import typing

from dupegrouper.definitions import DataFrameType


class DataFrameContainer(ABC):
    """Container class for a dataframe and associated methods

    At runtime any instance of this class will also be a data container of the
    dataframe. The abstractmethods defined here are all the required
    implementations needed
    """

    def __init__(self, df: DataFrameType):
        self._df: DataFrameType = df

    @staticmethod
    @abstractmethod
    def _add_group_id(df: DataFrameType):
        """Return a dataframe with a group id column"""
        pass

    # DATAFRAME `LIBRARY` WRAPPERS:

    @abstractmethod
    def put_col(self, column: str, array) -> typing.Self:
        """assign i.e. write a column with array-like data

        No return; `_df` is updated
        """
        pass

    @abstractmethod
    def get_col(self, column: str):
        """Return a column array-like of data"""
        pass

    @abstractmethod
    def map_dict(self, column: str, mapping: dict):
        """Return a column array-like of data mapped with `mapping`"""
        pass

    @abstractmethod
    def drop_col(self, column: str) -> typing.Self:
        """delete a column with array-like data

        No return: `_df` is updated
        """
        pass

    @staticmethod
    @abstractmethod
    def fill_na(series, array):
        """Return a column array-like of data null-filled with `array`"""
        pass

    @abstractmethod
    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)

    @property
    def frame(self):
        return self._df
    
    # THIN TRANSPARENCY DELEGATION

    @frame.setter
    def frame(self, new_frame: DataFrameType):
        self._df = new_frame
