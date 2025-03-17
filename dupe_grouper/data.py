import pandas as pd
import numpy as np

mock = pd.read_csv("mock_data.csv")
mock["phone"] = mock["phone"].astype("Int64").astype("string")
mock["address_number"] = mock["address_number"].astype("Int64").astype("string")

df1 = pd.DataFrame(
    {
        "id": {44: 1, 55: 2, 31: 3, 9: 4, 6: 5, 12: 6, 89: 7, 19: 8, 76: 9, 7: 10},
        "group_id": {
            44: 1,
            55: 2,
            31: 3,
            9: 4,
            6: 5,
            12: 6,
            89: 7,
            19: 8,
            76: 9,
            7: 10,
        },
        "email": {
            44: "bbab@example.com",
            55: "bb@example.com",
            31: "a@example.com",
            9: "hellothere@example.com",
            6: "b@example.com",
            12: "bab@example.com",
            89: "b@example.com",
            19: "hellthere@example.com",
            76: "hellathere@example.com",
            7: "irrelevant@hotmail.com",
        },
    }
)


df2 = pd.DataFrame(
    {
        "id": range(1, 1001),
        "group_id": range(1, 1001),
        "email": np.random.choice(
            [
                "bbab@example.com",
                "b@example.com",
                "a@example.com",
                "hellothere@example.com",
                "bb@example.com",
                "bab@example.com",
                "bb@example.com",
                "hellthere@example.com",
                "heythere@example.com",
                "irrelevant@hotmail.com",
            ],
            1000,
        ),
    }
)