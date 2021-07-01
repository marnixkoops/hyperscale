import logging
import sys
from typing import List

import numpy as np
from annoy import AnnoyIndex
from tqdm import trange

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger("⚡hyperscale")


class hyperscale:
    def build_vector_index(
        self, vectors: np.ndarray, n_trees: int = 5, save_index: bool = False
    ) -> AnnoyIndex:
        """Builds a vector index for Approximate Nearest Neighbor search.

        More trees is slower but gives higher precision when querying.
        This implementation is powered by https://github.com/spotify/annoy.

        Args:
            vectors: (embedding) vectors.
            n_trees: Number of trees to build for searching the index.
            save_index: Whether to save the index on disk.

        Returns:
            vector_index: Built vector index.
        """
        logger.info(f"Building vector index with {n_trees} trees")
        n_dimensions = vectors.shape[1]
        vector_index = AnnoyIndex(n_dimensions, metric="angular")

        for item in trange(vectors.shape[0]):
            vector = vectors[item]
            vector_index.add_item(item, vector)

        vector_index.build(n_trees=n_trees)

        if save_index:
            vector_index.save("vector_index.ann")

        return vector_index

    def find_most_similar(
        self, vector_index: AnnoyIndex, vector_id: int = None, n_vectors: int = 10,
    ) -> List:
        """Finds the most similar vectors using Approximate Nearest Neighbors.

        Given a vector id, searches most similar items in the index for that vector.
        If no vector id is supplied, finds the top N most similar vectors for each
        vector that is included in the index.

        Args:
            vector_index: Built vector index.
            vector_id: Index of vector to find the most similar vectors for.
            n_vectors: Number of most similar vectors to find.

        Returns:
            most_similar_vectors: top N most similar vectors.
        """
        if vector_id is not None:
            logger.info(f"Querying most similar vectors for vector id {vector_id}")
            most_similar_vectors = vector_index.get_nns_by_item(vector_id, n_vectors)
        else:
            n_vectors_in_index = vector_index.get_n_items()
            most_similar_vectors = np.empty(
                [n_vectors_in_index, n_vectors], dtype=np.int32
            )
            logger.info(
                f"Querying most similar vectors for all {n_vectors_in_index} vectors"
            )
            for vector in trange(n_vectors_in_index):
                most_similar_vectors[vector] = vector_index.get_nns_by_item(
                    vector, n_vectors
                )

        return most_similar_vectors

    def augment_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Augments vectors for fast (aproximate) maximum inner product search.

        Transforms each row of a vector matrix by adding an extra dimension giving
        equal norms. The cosine of the augmented vector is now proportional to the
        inner product. As a result, an angular nearest neighbours search will find
        vectors that result in the highest inner product.

        This approach was introduced in the paper: "Speeding Up the Xbox Recommender
        System Using a Euclidean Transformation for Inner-Product Spaces"
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

        Args:
            vectors: (Embedding) vectors.

        Returns:
            augmented_vectors: Augmented (embedding) vectors.
        """
        logger.info(
            "Augmenting vectors with Euclidean transformation for recommendation"
        )
        vector_norms = np.linalg.norm(vectors, axis=1)
        max_vector_norm = vector_norms.max()

        extra_dimension = np.sqrt(max_vector_norm ** 2 - vector_norms ** 2)
        augmented_vectors = np.append(
            vectors, extra_dimension.reshape(vector_norms.shape[0], 1), axis=1
        )

        return augmented_vectors

    def recommend(
        self, user_vectors: np.ndarray, item_vectors: np.ndarray, n_vectors: int = 10
    ) -> np.ndarray:
        """Approximate maximum inner product search for fast recommendations.

        After running an algorithm like Matrix Factorization, every user/item can be
        represented as an (embedding) vector in n-dimensional space. We increases the
        speed of recommendation and similar user/item search for large data-sets at low
        memory cost.

        Uses an implementation of Approximate Nearest Neighbours search.
        The massive speed-up comes at the cost of slighly reduced precision. In a
        typical recommendation setting this is not a problem because a large amount of
        top items are relevant, regardless of negligible differences in estimated score.

        Args:
            user_vectors: User embedding vectors.
            item_vectors: Item embedding vectors.
            n_trees: Number of trees to build for searching the index.
            n_vectors: Number of items to recommend for each input vector.

        Returns:
            recommendations: Top N item recommendations for each user.
        """
        logger.info("Augmenting user vectors with extra dimension for recommendations")
        extra_dimension = np.zeros((user_vectors.shape[0], 1))
        augmented_user_vectors = np.concatenate((user_vectors, extra_dimension), axis=1)

        augmented_item_vectors = self.augment_vectors(item_vectors)
        vector_index = self.build_vector_index(augmented_item_vectors, n_trees=5)

        n_users = augmented_user_vectors.shape[0]
        recommendations = np.empty([n_users, n_vectors], dtype=np.int32)
        logger.info(f"Finding top {n_vectors} items for each user")
        for user in trange(n_users):
            user_vector = augmented_user_vectors[user]
            recommendations[user] = vector_index.get_nns_by_vector(
                user_vector, n_vectors
            )

        return recommendations

    def explain(self):
        """To be implemented."""
        pass
