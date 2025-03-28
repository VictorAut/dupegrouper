import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def dataframe_data():
    return [
        [1, "123ab, OL5 9PL, UK", "bbab@example.com", 1],
        [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "bb@example.com", 2],
        [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com", 3],
        [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com", 4],
        [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com", 5],
        [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com", 6],
        [7, "C. Ancho 49, 05687, Navarra", "b@example.com", 7],
        [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com", 8],
        [9, "123ab, OL5 9PL, UK", "hellathere@example.com", 9],
        [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com", 10],
        [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com", 11],
        [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com", 12],
        [13, "totally random non existant address", "fictitious@never.co.uk", 13],
    ]


@pytest.fixture
def df_pandas(dataframe_data):
    return pd.DataFrame(columns=["id", "address", "email", "group_id"], data=dataframe_data)


@pytest.fixture
def df_polars(dataframe_data):
    return pl.DataFrame(schema=["id", "address", "email", "group_id"], data=dataframe_data, orient="row")
