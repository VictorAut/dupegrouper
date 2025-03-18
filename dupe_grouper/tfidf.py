import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

import data

df = data.df3


vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(5, 5))

attr_matrix = vectorizer.fit_transform(df["address"])

similarities = sp_matmul_topn(attr_matrix, attr_matrix.T, top_n=10, sort=True)

sparse = similarities

sparse.tocoo().coords[sparse.data > 0.5]


def get_matches_df(sparse, array, tolerance: float = 0.05):

    sparse_coo = sparse.tocoo()  # i.e. COO format
    mask = sparse.data > (1 - tolerance)
    rows, cols = sparse_coo.row[mask], sparse_coo.col[mask]

    n = len(rows)

    attr = np.empty([n], dtype=object)
    match = np.empty([n], dtype=object)
    similarity = np.zeros([n])

    for i in range(0, n):

        attr[i] = array[rows[i]]
        match[i] = array[cols[i]]
        similarity[i] = sparse[rows[i], cols[i]]

    return attr, match, similarity


lookup, match, _ = get_matches_df(similarities, df["address"], tolerance=0.7)

my_dict = {k: v for k, v in zip(lookup, match)}
