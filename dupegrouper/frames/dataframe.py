"""Abstract base class container"""

from abc import ABC, abstractmethod

import typing

import pandas as pd


class DFMethods(ABC):
    """Implements needed methodsgiven any instance of a dataframe

    At runtime any instance of this class will also be a data container of the
    dataframe. The abstractmethods defined here are all the required
    implementations needed
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @abstractmethod
    def put_col(self, column: str, array) -> typing.Self:
        """assign i.e. write a column with array-like data

        No return: `_df` is updated
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
    def fill_na(series: pd.Series, array):
        """Return a column array-like of data null-filled with `array`"""
        pass

    @property
    @abstractmethod
    def frame(self):
        pass
