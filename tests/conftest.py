import pandas as pd
import polars as pl
import pytest

@pytest.fixture
def base_data():
    return [
    [1, "123ab, OL5 9PL, UK", "bbab@example.com"],
    [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "bb@example.com"],
    [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com"],
    [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com"],
    [5, "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom", "b@example.com"],
    [6, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com"],
    [7, "C. Ancho 49, 05687, Navarra", "b@example.com"],
    [8, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com"],
    [9, "123ab, OL5 9PL, UK", "hellathere@example.com"],
    [10, "123ab, OL5 9PL, UK", "irrelevant@hotmail.com"],
    [11, "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK", "yet.another.email@msn.com"],
    [12, "37 GH9, UK", "awesome_surfer_77@yahoo.com"],
    [13, "totally random non existant address", "fictitious@never.co.uk"],
]


@pytest.fixture
def raw_data(base_data):
    return base_data


@pytest.fixture
def initialised_data(base_data):
    for i in range(len(base_data)):
        base_data[i].append(i + 1)
    return base_data


# raw data i.e. no "GROUP ID"


@pytest.fixture
def df_pandas_raw(raw_data):
    return pd.DataFrame(columns=["id", "address", "email"], data=raw_data)


@pytest.fixture
def df_polars_raw(raw_data):
    return pl.DataFrame(schema=["id", "address", "email"], data=raw_data, orient="row")


# "initialised" data


@pytest.fixture
def df_pandas(initialised_data):
    return pd.DataFrame(columns=["id", "address", "email", "group_id"], data=initialised_data)


@pytest.fixture
def df_polars(initialised_data):
    return pl.DataFrame(schema=["id", "address", "email", "group_id"], data=initialised_data, orient="row")
