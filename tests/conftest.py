import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
import pytest


@pytest.fixture(scope="function")
def id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="function")
def address():
    return [
        "123ab, OL5 9PL, UK",
        "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom",
        "Calle Ancho, 12, 05688, Rioja, Navarra, Espana",
        "Calle Sueco, 56, 05688, Rioja, Navarra",
        "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom",
        "66b Porters street, OL5 9PL, Newark, United Kingdom",
        "C. Ancho 49, 05687, Navarra",
        "Ambleside avenue Park Road ED3, UK",
        "123ab, OL5 9PL, UK",
        "123ab, OL5 9PL, UK",
        "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK",
        "37 GH9, UK",
        "totally random non existant address",
    ]


@pytest.fixture(scope="function")
def email():
    return [
        "bbab@example.com",
        "bb@example.com",
        "a@example.com",
        "hellothere@example.com",
        "b@example.com",
        "bab@example.com",
        "b@example.com",
        "hellthere@example.com",
        "hellathere@example.com",
        "irrelevant@hotmail.com",
        "yet.another.email@msn.com",
        "awesome_surfer_77@yahoo.com",
        "fictitious@never.co.uk",
    ]


@pytest.fixture(scope="function")
def blocking_key():
    return [
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
    ]


@pytest.fixture(scope="function")
def group_id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="function")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


# raw data i.e. no "GROUP ID"


@pytest.fixture(scope="function")
def df_pandas_raw(id, address, email):
    return pd.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="function")
def df_polars_raw(id, address, email):
    return pl.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="function")
def df_spark_raw(spark, id, address, email, blocking_key):
    return spark.createDataFrame(
        [[id[i], address[i], email[i], blocking_key[i]] for i in range(len(id))],
        schema=("id", "address", "email", "blocking_key"),
    )


# "initialised" data


@pytest.fixture(scope="function")
def df_pandas(id, address, email, group_id):
    return pd.DataFrame({"id": id, "address": address, "email": email, "group_id": group_id})


@pytest.fixture(scope="function")
def df_polars(id, address, email, group_id):
    return pl.DataFrame({"id": id, "address": address, "email": email, "group_id": group_id})
