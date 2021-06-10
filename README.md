

**⚡ Lightning fast recommendations and similarity search.**

`turboquery` solves the main challenge in real world recommender systems: querying recommendations fast at scale.

When the number of items is large, scoring and ranking all of them is infeasible. `turboquery` implements algorithms to leverage fast Approximate Nearest Neighbors (ANN) in high-dimensional space for similarity and maximum inner-product search (recommendation). This approach can produce sub 50 µs response-times across millions of items.

`turboquery` can be used in synergy with any (embedding) vector-based machine learning model. For example, to quickly find the most similar embeddings learned by a Neural Network or to generate recommendations for users based on any type of Matrix Factorization model.

## ✨ Install

```bash
$ pip install turboquery
```

## 🚀 Quick Start

```python
import turboquery
import numpy as np

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

turboquery = turboquery()
vector_index = turboquery.build_vector_index(vectors)
recommendations = turboquery.recommend(user_vectors, vectors)
most_similar = turboquery.find_most_similar(vector_index)
```

```
>>> INFO:⚡turboquery:Augmenting user vectors with extra dimension
>>> INFO:⚡turboquery:Augmenting vectors with Euclidean transformation
>>> INFO:⚡turboquery:Building vector index with 5 trees
>>> 100%|██████████| 10000/10000 [00:00<00:00, 361652.76it/s]
>>> INFO:⚡turboquery:Searching top 10 items for each user
>>> 100%|██████████| 1000/1000 [00:00<00:00, 66724.53it/s]
>>> INFO:⚡turboquery:Building vector index with 5 trees
>>> 100%|██████████| 10000/10000 [00:00<00:00, 360304.44it/s]
>>> INFO:⚡turboquery:Querying most similar vectors for all 10000 vectors
>>> 100%|██████████| 10000/10000 [00:00<00:00, 91580.89it/s]
```

## 🖇️ References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*
