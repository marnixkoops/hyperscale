import logging
import sys

# import nmslib
import numpy as np
import sklearn

from hyperscale import hyperscale

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger("hyperscale")

######################


# def cosine_similarities(matrix):
#     """Cosine similarity for continuous vectors."""
#     matrix = matrix.tocsc()
#     normalized_matrix = normalize(matrix, axis=0)
#     cosine_matrix = normalized_matrix.T * normalized_matrix
#     return cosine_matrix


# def jaccard_similarities(matrix):
#     """Jaccard similarity for binary vectors."""
#     column_sums = matrix.getnnz(axis=0)
#     ab = matrix.T * matrix

#     aa = np.repeat(column_sums, ab.getnnz(axis=0))
#     bb = column_sums[ab.indices]

#     jaccard_similarities = ab.copy()
#     jaccard_similarities.data /= aa + bb - ab.data

#     return jaccard_similarities


# matrix = scipy.sparse.rand(10 ** 2, 10 ** 2, 0.05, format="csc")

# ## %%timeit
# cosine_similarities(matrix)

# matrix.data[:] = 1
# ## %%timeit
# jaccard_similarities(matrix)

#################

"""
"nmslib_als": NMSLibAlternatingLeastSquares,
"annoy_als": AnnoyAlternatingLeastSquares,
"faiss_als": FaissAlternatingLeastSquares,
"""


# Now an NMS index
# NMSlib outperforms Annoy in terms of accuracy, but I also found that constructing
# indexes over a large dataset was slower. This becomes a problem if you have to
# rebuild your index frequently because of fast moving product catalogues etc.
# nms_member_idx = nmslib.init(method="hnsw", space="cosinesimil")
# nms_member_idx.addDataPointBatch(user_vectors)
# nms_member_idx.createIndex(print_progress=True)


########################################################################

# ratings = pd.read_csv(
#     "data/ml-1m/ratings.dat",
#     delimiter="::",
#     names=["userId", "movieId", "rating", "timestamp"],
#     engine="python",
# )

# movies = pd.read_table(
#     "data/ml-1m/movies.dat",
#     delimiter="::",
#     names=["movieId", "title", "genres"],
#     engine="python",
# )

from sklearn.decomposition import NMF

########################################################################
from hyperscale import hyperscale

matrix = np.random.rand(1000, 1000)

model = NMF(n_components=16)
model.fit(matrix)

user_vectors = model.transform(matrix)
item_vectors = model.components_.T

recommendations = hyperscale.recommend(user_vectors, item_vectors)
vector_index = hyperscale.build_vector_index(vectors=item_vectors, n_trees=5)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)

########################################################################
from implicit.als import AlternatingLeastSquares
from scipy import sparse

matrix = np.random.rand(1000, 1000)
sparse_matrix = sparse.csr_matrix(matrix)

model = AlternatingLeastSquares(factors=16)
model.fit(sparse_matrix)

user_vectors = model.user_factors
item_vectors = model.item_factors

recommendations = hyperscale.recommend(user_vectors, item_vectors)
vector_index = hyperscale.build_vector_index(vectors=item_vectors, n_trees=5)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)

########################################################################
from surprise import SVD, Dataset

data = Dataset.load_builtin("ml-100k")
data = data.build_full_trainset()

model = SVD(n_factors=16)
model.fit(data)

user_vectors = model.pu
item_vectors = model.qi

recommendations = hyperscale.recommend(user_vectors, item_vectors)
vector_index = hyperscale.build_vector_index(vectors=item_vectors, n_trees=5)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)

########################################################################
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

model = LightFM(loss="warp")
model.fit(data["train"])

_, user_vectors = model.get_user_representations(features=None)
_, item_vectors = model.get_item_representations(features=None)

recommendations = hyperscale.recommend(user_vectors, item_vectors)
vector_index = hyperscale.build_vector_index(vectors=item_vectors, n_trees=5)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)


########################################################################
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

model = Word2Vec(sentences=common_texts, vector_size=16, window=5, min_count=1)
gensim_vectors = model.wv
item_vectors = gensim_vectors.get_normed_vectors()

hyperscale = hyperscale()
vector_index = hyperscale.build_vector_index(vectors=item_vectors, n_trees=5)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)
