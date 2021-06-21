

**⚡ Lightning fast recommendations and vector similarity search**

`zift` solves the main challenge in real world recommender systems: querying recommendations fast at scale.

When the number of items is large, scoring and ranking all of them is infeasible. `zift` implements algorithms to leverage fast Approximate Nearest Neighbors (ANN) in high-dimensional space for similarity search and maximum inner-product search (recommendation). This method is computationally efficient and produces microsecond response-times across millions of items.

Furthermore, vector models are widely used in NLP, recommender systems, computer vision and other fields. `zift` can be used in synergy with any (embedding) vector-based machine learning model. For example, to quickly find the most similar embeddings learned by a Neural Network or to generate recommendations for users based on any type of Matrix Factorization model. Additionally, finding similar vectors can be effectively applied to increase customer experience by personalization of content, product upsell, cross-sell and other use-cases.

## ✨ Install

```bash
$ pip install zift
```

## 🚀 Quick start

```python
import zift
import numpy as np

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

zift = zift()
vector_index = zift.build_vector_index(vectors)
recommendations = zift.recommend(user_vectors, vectors)
most_similar = zift.find_most_similar(vector_index)
```

## 🖇️ References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*

* Bern, Erik. [Annoy (Approximate Nearest Neighbors Oh Yeah)](https://github.com/spotify/annoy)
