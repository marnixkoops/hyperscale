![](hyperscale.png)

### Fast recommendations and vector similarity search at scale

## 👋 Hello

`hyperscale` solves the main challenge in real world recommender systems; querying recommendations fast at scale. It offers a simple Python API and leverages C++ under the hood.

When the number of items is large, scoring and ranking all combinations is infeasible. `hyperscale` implements fast Approximate Nearest Neighbors (ANN) algorithms in high-dimensional space for vector similarity search and maximum inner-product search (recommendation). This method is computationally efficient and produces microsecond response-times across millions of items.

Furthermore, vector models are widely used in NLP, recommender systems, computer vision and other fields. `hyperscale` can be used in combination with any (embedding) vector-based model. For example, to quickly find the most similar embeddings learned by a Neural Network or to generate recommendations for users based on any type of Collaborative Filtering model such as Matrix Factorization.

Similar vector search can also be effectively applied to increase customer experience through personalization of content, product upsell, cross-sell and other use-cases.

## ✨ Install

```console
$ pip3 install hyperscale
```

## 🚀 Quick start

```python
import hyperscale
import numpy as np

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, item_vectors)

vector_index = hyperscale.build_vector_index(item_vectors)
most_similar = hyperscale.find_most_similar(vector_index)
```

## ✨ Examples

It is easy to scale up recommendation or similarity search after training a model with any popular library. Simply extract the embedding vectors and feed them to `hyperscale`. See examples below.

###### [`sklearn`](https://github.com/scikit-learn/scikit-learn)
<details><summary><b>show code</b></summary>

```python
import hyperscale
import numpy as np

item_vectors = np.random.rand(int(1e6), 32)
user_vectors = np.random.rand(int(1e4), 32)

hyperscale = hyperscale()
recommendations = hyperscale.recommend(user_vectors, vectors)
```

</details>

###### [`surprise`](https://github.com/NicolasHug/Surprise)
<details><summary><b>show code</b></summary>
xxx
</details>

###### [`lightfm`](https://github.com/lyst/lightfm)
<details><summary><b>show code</b></summary>
xxx
</details>

###### [`implicit`](https://github.com/benfred/implicit)
<details><summary><b>show code</b></summary>
xxx
</details>

###### [`tensorflow`](https://github.com/tensorflow/tensorflow)
<details><summary><b>show code<br> </b></summary>
xxx
</details>

###### [`pytorch`](https://github.com/pytorch/pytorch)
<details><summary><b>show code</b></summary>
xxx
</details>

###### [`umap-learn`](https://github.com/lmcinnes/umap)
<details><summary><b>show code<br> </b></summary>
xxx
</details>

## 🖇️ References

* Bachrach, Yoram, et al. ["Speeding up the Xbox Recommender System using a Euclidean transformation for inner-product spaces."](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) *Proceedings of the 8th ACM Conference on Recommender systems. 2014.*

* Bern, Erik. [Annoy (Approximate Nearest Neighbors Oh Yeah)](https://github.com/spotify/annoy) *Spotify. 2015.*
