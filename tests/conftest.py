import pandas as pd
import polars as pl
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
import pytest


@pytest.fixture(scope="session")
def id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def group_id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def df_pandas_raw(id, address, email):
    return pd.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="session")
def df_polars_raw(id, address, email):
    return pl.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="session")
def df_spark_raw(spark, id, address, email, blocking_key):
    """default is a single partition"""
    return spark.createDataFrame(
        [[id[i], address[i], email[i], blocking_key[i]] for i in range(len(id))],
        schema=("id", "address", "email", "blocking_key"),
    ).repartition(1, "blocking_key")


# "initialised" data


@pytest.fixture(scope="session")
def df_pandas(id, address, email, group_id):
    return pd.DataFrame({"id": id, "address": address, "email": email, "group_id": group_id})


@pytest.fixture(scope="session")
def df_polars(id, address, email, group_id):
    return pl.DataFrame({"id": id, "address": address, "email": email, "group_id": group_id})


@pytest.fixture(params=["pandas", "polars", "spark"], scope="session")
def dataframe(request, df_pandas_raw, df_polars_raw, df_spark_raw, spark) -> tuple:
    """return a tuple of positionally ordered input parameters of DupeGrouper
    i.e.
        - df
        - spark_session
        - id
    """
    match request.param:
        case "pandas":
            return df_pandas_raw, None, None
        case "polars":
            return df_polars_raw, None, None
        case "spark":
            return df_spark_raw, spark, "id"
        

# helpers

class Helpers():
    @staticmethod
    def get_group_id_as_list(df):
        if isinstance(df, pd.DataFrame | pl.DataFrame):
            return list(df["group_id"])
        if isinstance(df, SparkDataFrame):
            return [value["group_id"] for value in df.select("group_id").collect()]
        if isinstance(df, list):
            return [value["group_id"] for value in df]
        
@pytest.fixture(scope="session", autouse=True)
def helpers():
    return Helpers