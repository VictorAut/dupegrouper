from abc import ABC, abstractmethod

import typing

import pandas as pd


class DFMethods(ABC):

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @abstractmethod
    def put_col(self, column: str, array) -> typing.Self:
        pass

    @abstractmethod
    def get_col(self, column: str):
        pass

    @abstractmethod
    def map_dict(self, column: str, mapping: dict):
        pass

    @abstractmethod
    def drop_col(self, column: str) -> typing.Self:
        pass

    @staticmethod
    @abstractmethod
    def fill_na(series: pd.Series, array):
        pass

    @property
    @abstractmethod
    def frame(self):
        pass
