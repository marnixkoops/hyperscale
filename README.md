

**Lightning fast recommendations.**

## 👋 Introduction

`package` solves the main challenge in real world recommender systems: querying recommendations at scale.

When the number of items is large, scoring and ranking all of them is infeasible. `package` implements algorithms to leverage fast Approximate Nearest Neighbors (ANN) in high-dimensional space for similarity and maximum inner-product search (recommendation).

`package` can be used in synergy with any (embedding) vector-based machine learning model. For example, to quickly find the most similar embeddings learned by a Neural Network or to generate recommendations for user/item embeddings from any Matrix Factorization based model. You only need some vectors!

## ✨ Install

```bash
$ pip install package
```

## 🚀 Quick Start

```python
import package

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

package = package()
vector_index = package.build_vector_index(vectors)
recommendations = package.recommend(user_vectors, vectors)
most_similar = package.find_most_similar(vector_index)
```

```
>>> INFO:⚡package:Augmenting user vectors with extra dimension
>>> INFO:⚡package:Augmenting vectors with Euclidean transformation
>>> INFO:⚡package:Building vector index with 5 trees
>>> 100%|██████████| 10000/10000 [00:00<00:00, 361652.76it/s]
>>> INFO:⚡package:Searching top 10 items for each user
>>> 100%|██████████| 1000/1000 [00:00<00:00, 66724.53it/s]
>>> INFO:⚡package:Building vector index with 5 trees
>>> 100%|██████████| 10000/10000 [00:00<00:00, 360304.44it/s]
>>> INFO:⚡package:Querying most similar vectors for all 10000 vectors
>>> 100%|██████████| 10000/10000 [00:00<00:00, 91580.89it/s]
```

## 📎 References
