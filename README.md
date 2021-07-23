# ⚡ hyperscale
**Fast recommendations and vector similarity search**

## 👋 Hello

`hyperscale` mainly solves the bottleneck in recommender systems; querying recommendations fast at scale.

When the number of items is large, scoring and ranking all combinations is computationally expensive. `hyperscale` implements Approximate Nearest Neighbors (ANN) in high-dimensional space for fast (embedding) vector similarity search and maximum inner-product search (recommendation). This algorithm is computationally efficient and able to produce microsecond response-times across millions of vectors.

`hyperscale` can be used in combination with any (embedding) vector-based model. It is easy to scale up recommendation or vector similarity search without increasing engineering complexity. Simply extract the trained embeddings and feed them to `hyperscale` to handle querying.

For example, it can be used to quickly find the top item recommendations for a user after training a Collaborative Filtering model like Matrix Factorization. Or to quickly find the most similar items in a large set learned by an Embedding Neural Network. Some example code snippets for popular libraries are supplied below.

## 🪄 Install

```shell
$ pip3 install hyperscale
```

## 🚀 Quickstart

`hyperscale` offers a simple and lightweight Python API. While leveraging C++ under the hood for speed and scalability, it avoids engineering complexity to get up and running.

```python
from hyperscale import hyperscale
import numpy as np

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_most_similar(vector_index)
```

## ✨ Examples

###### [`sklearn`](https://github.com/scikit-learn/scikit-learn)
<details><summary><b>show code</b></summary>

```python
from hyperscale import hyperscale
from sklearn.decomposition import NMF

matrix = np.random.rand(1000, 1000)

model = NMF(n_components=16)
model.fit(matrix)

user_vectors = model.transform(matrix)
item_vectors = model.components_.T

recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_most_similar(vector_index=vector_index, n_vectors=10)
```

</details>

###### [`surprise`](https://github.com/NicolasHug/Surprise)
<details><summary><b>show code</b></summary>

```python
from hyperscale import hyperscale
from surprise import SVD, Dataset

data = Dataset.load_builtin("ml-100k")
data = data.build_full_trainset()

model = SVD(n_factors=16)
model.fit(data)

user_vectors = model.pu
item_vectors = model.qi

recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_most_similar(vectors=item_vectors, n_vectors=10)
```

</details>

###### [`lightfm`](https://github.com/lyst/lightfm)
<details><summary><b>show code</b></summary>

```python
from hyperscale import hyperscale
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

model = LightFM(loss="warp")
model.fit(data["train"])

_, user_vectors = model.get_user_representations(features=None)
_, item_vectors = model.get_item_representations(features=None)

recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_most_similar(vectors=item_vectors, n_vectors=10)
```

</details>

###### [`implicit`](https://github.com/benfred/implicit)
<details><summary><b>show code</b></summary>

```python
from hyperscale import hyperscale
from implicit.als import AlternatingLeastSquares
from scipy import sparse

matrix = np.random.rand(1000, 1000)
sparse_matrix = sparse.csr_matrix(matrix)

model = AlternatingLeastSquares(factors=16)
model.fit(sparse_matrix)

user_vectors = model.user_factors
item_vectors = model.item_factors

recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_most_similar(vectors=item_vectors, n_vectors=10)
```

</details>

###### [`gensim`](https://github.com/RaRe-Technologies/gensim)
<details><summary><b>show code</b></summary>

```python
from hyperscale import hyperscale
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

model = Word2Vec(sentences=common_texts, vector_size=16, window=5, min_count=1)
gensim_vectors = model.wv
item_vectors = gensim_vectors.get_normed_vectors()

hyperscale = hyperscale()
most_similar = hyperscale.find_most_similar(vectors=item_vectors, n_vectors=10)
```

</details>

## 🧮 Algorithm

`hyperscale` leverages random projections in high-dimensional space to build up a tree structure. Subspaces are created by repeatedly selecting two points at random and dividing them with a hyperplane. A tree is built `k` times to generate a forest of random hyperplanes and subspaces. This is controlled by the `n_trees` parameter when building the vector index. This parameter can be tuned to your needs based on the trade-off between precision and performance. In practice, it should probably be around the same order of vector dimensionality.

## 🔗 References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*

* Bern, Erik. [Annoy (Approximate Nearest Neighbors Oh Yeah)](https://github.com/spotify/annoy) *Spotify. 2015.*
