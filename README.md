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
import numpy as np
from hyperscale import hyperscale

item_vectors = np.random.rand(int(1e5), 16)
user_vectors = np.random.rand(int(1e4), 16)

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_similar_vectors(item_vectors)

recommendations
```

```
INFO:⚡hyperscale:👥 Augmenting user vectors with extra dimension
INFO:⚡hyperscale:📐 Augmenting vectors with Euclidean transformation
INFO:⚡hyperscale:🌲 Building vector index with 17 trees
100%|██████████| 100000/100000 [00:00<00:00, 349877.84it/s]
INFO:⚡hyperscale:🔍 Finding top 10 item recommendations for all 10000 users
100%|██████████| 10000/10000 [00:00<00:00, 27543.66it/s]

array([[24693, 93429, 84972, ..., 75432, 92763, 82794],
       [74375, 46553, 93429, ..., 92763, 32855, 42209],
       [42209, 64852, 52691, ..., 97284, 15103,  9693],
       ...,
       [ 6321, 42209, 15930, ...,  5718, 18849, 47896],
       [72212, 89112, 47896, ..., 49733,  9693, 93429],
       [82773, 72212, 34530, ..., 69396, 85292, 93429]], dtype=int32)
```

## ✨ Examples

###### [`sklearn`](https://github.com/scikit-learn/scikit-learn)
<details><summary><b>show code</b></summary>

```python
import numpy as np
from hyperscale import hyperscale
from sklearn.decomposition import NMF, TruncatedSVD

matrix = np.random.rand(1000, 1000)

model = NMF(n_components=16)
model.fit(matrix)

model = TruncatedSVD(n_components=16)
model.fit(matrix)

user_vectors = model.transform(matrix)
item_vectors = model.components_.T

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)
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

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_similar_vectors(vectors=item_vectors, n_vectors=10)
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

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_similar_vectors(vectors=item_vectors, n_vectors=10)
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

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)
most_similar = hyperscale.find_similar_vectors(vectors=item_vectors, n_vectors=10)
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
most_similar = hyperscale.find_similar_vectors(vectors=item_vectors, n_vectors=10)
```

</details>

## 🧮 Algorithm

`hyperscale` leverages random projections in high-dimensional space to build up a tree structure. Subspaces are created by repeatedly selecting two points at random and dividing them with a hyperplane. A tree is built `k` times to generate a forest of random hyperplanes and subspaces. This is controlled by the `n_trees` parameter when building the vector index. This parameter can be tuned to your needs based on the trade-off between precision and performance. In practice, it should probably be around the same order of vector dimensionality.

## 🔗 References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*

* Bern, Erik. [Annoy (Approximate Nearest Neighbors Oh Yeah)](https://github.com/spotify/annoy) *Spotify. 2015.*
