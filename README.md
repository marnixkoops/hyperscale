

**⚡ Lightning fast recommendations and vector similarity search**

`turboquery` solves the main challenge in real world recommender systems: querying recommendations fast at scale.

When the number of items is large, scoring and ranking all of them is infeasible. `turboquery` implements algorithms to leverage fast Approximate Nearest Neighbors (ANN) in high-dimensional space for similarity search and maximum inner-product search (recommendation). This method can produce microsecond response-times across millions of items.

Vector models are widely used across fields: NLP, recommender systems, computer vision, and more.
`turboquery` can be used in synergy with any (embedding) vector-based machine learning model. For example, to quickly find the most similar embeddings learned by a Neural Network or to generate recommendations for users based on any type of Matrix Factorization model.

## ✨ Install

```bash
$ pip install turboquery
```

## 🚀 Quick start

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

## 🖇️ References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*
