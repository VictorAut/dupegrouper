# """This test module covers the same type of tests found in
# - test_base.py
# - test_strategy.py

# But, collected in one place for the purpose of uniquely testing Spark cases.
# """

# import pytest

# from pyspark.sql import SparkSession, DataFrame
# from pyspark.sql.functions import lit


# @pytest.fixture(scope="session")
# def spark():
#     spark = (
#         SparkSession.builder.master("local[1]")
#         .appName("local-tests")
#         .config("spark.executor.cores", "1")
#         .config("spark.executor.instances", "1")
#         .config("spark.sql.shuffle.partitions", "1")
#         .config("spark.driver.bindAddress", "127.0.0.1")
#         .getOrCreate()
#     )
#     yield spark
#     spark.stop()
