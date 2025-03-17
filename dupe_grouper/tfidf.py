import scipy as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sparse_dot_topn

import data

df = data.df3


vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(5, 5))

attribute_matrix = vectorizer.fit_transform(df['address'])



def cosine_similarity_topN(attribute_matrix, ntop, lower_bound=0):
    
    mat1, mat2 = attribute_matrix, attribute_matrix.T

    M, N = attribute_matrix.shape # i.e. (document length, features length)

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=attribute_matrix.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype), np.asarray(A.indices, dtype=idx_dtype), A.data,
        np.asarray(B.indptr, dtype=idx_dtype), np.asarray(B.indices, dtype=idx_dtype), B.data,
        ntop, lower_bound, indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, original_vector, lookup_vector):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    nr_matches = sparsecols.size

    leftside = np.empty([nr_matches], dtype=object)
    rightside = np.empty([nr_matches], dtype=object)
    similarity = np.zeros([nr_matches])

    for i in range(0, nr_matches):

        leftside[i] = original_vector[sparserows[i]]
        rightside[i] = lookup_vector[sparsecols[i]]
        similarity[i] = sparse_matrix[sparserows[i], sparsecols[i]]

    return pd.DataFrame({'original': leftside,
                            'look_up': rightside,
                            'similarity': similarity})

#make tf_idf arrays of relevant column, by default the postal_address
original_arr = np.array(original[col_name].apply(str))
lookup_arr = np.array(lookup[col_name].apply(str))


vectorizer = TfidfVectorizer(analyzer=ngrams)

tf_idf_matrix = vectorizer.fit_transform(original_arr)
tf_idf_matrix2 = vectorizer.fit_transform(lookup_arr)

logger_tfidf_helper.info("making sparse cosine similarity matrix")
#make matches
matches_similarities = cosine_similarity_topN(tf_idf_matrix, tf_idf_matrix2.transpose(), top_n_matches, 0.1)

logger_tfidf_helper.info("making matches")
#get matches to original and lookup
matches_similarities = get_matches_df(matches_similarities, original_arr, lookup_arr)

#remove exact matches (i.e. self comparison)
matches_similarities = matches_similarities[matches_similarities.similarity < 0.99999]

#return with joined original dataframes
matches_similarities = matches_similarities.merge(lookup, left_on='look_up', right_on=col_name).rename(columns={'mc_id':'candidate_mc_id'}).drop([col_name], axis=1)\
                                            .merge(original, left_on='original', right_on=col_name).drop([col_name], axis=1).sort_values(['similarity'], ascending=False)

return matches_similarities

