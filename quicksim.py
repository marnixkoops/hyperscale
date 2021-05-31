import logging
import sys
from typing import List

import numpy as np
from annoy import AnnoyIndex
from tqdm import trange

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO,)
logger = logging.getLogger("⚡quicksim")


class quicksim:
    def augment_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Augments vectors for fast (aproximate) maximum inner product search.

        This function transforms each row of a matrix by adding one extra dimension
        giving equal norms. Cosine of the augmented vector is now proportional to the
        dot product. As a result, an angular nearest neighbours search will return top
        related items of the inner product.

        This technique was introduced in the paper: "Speeding Up the Xbox Recommender
        System Using a Euclidean Transformation for Inner-Product Spaces"
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

        Args:
            vectors (np.ndarray): (embedding) vectors.

        Returns:
            np.ndarray: Augmented (embedding) vectors.
        """
        logger.info("Augmenting vectors with Euclidean transformation")
        vector_norms = np.linalg.norm(vectors, axis=1)
        max_vector_norm = vector_norms.max()

        extra_dimension = np.sqrt(max_vector_norm ** 2 - vector_norms ** 2)
        augmented_vectors = np.append(
            vectors, extra_dimension.reshape(vector_norms.shape[0], 1), axis=1
        )

        return augmented_vectors

    def build_vector_index(
        self, vectors: np.ndarray, n_trees: int = 5, save_index: bool = False
    ) -> AnnoyIndex:
        """Builds a vector index for fast (approximate) nearest neighbor search.

        More trees is slower but gives higher precision when querying.
        This implementation is powered by https://github.com/spotify/annoy.

        Args:
            vectors (np.ndarray): [description]
            n_trees (int, optional): [description]. Defaults to 5.
            save_index (bool, optional): [description]. Defaults to False.

        Returns:
            AnnoyIndex: Built vector index.
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
        """[summary]

        Args:
            vector_index (AnnoyIndex): [description]
            vector_id (int, optional): [description]. Defaults to None.
            n_vectors (int, optional): [description]. Defaults to 10.

        Returns:
            List: [description]
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

    def recommend(
        self, user_vectors: np.ndarray, item_vectors: np.ndarray, n_vectors: int = 10
    ) -> np.ndarray:
        """[summary]

        Args:
            user_vectors (np.ndarray): [description]
            item_vectors (np.ndarray): [description]
            n_vectors (int, optional): [description]. Defaults to 10.

        Returns:
            np.ndarray: [description]
        """

        logger.info("Augmenting user vectors with extra dimension")
        extra_dimension = np.zeros((user_vectors.shape[0], 1))
        augmented_user_vectors = np.concatenate((user_vectors, extra_dimension), axis=1)

        augmented_item_vectors = self.augment_vectors(item_vectors)
        vector_index = self.build_vector_index(augmented_item_vectors, n_trees=10)

        n_users = augmented_user_vectors.shape[0]
        recommendations = np.empty([n_users, n_vectors], dtype=np.int32)
        logger.info(f"Searching top {n_vectors} items for each user")
        for user in trange(n_users):
            user_vector = augmented_user_vectors[user]
            recommendations[user] = vector_index.get_nns_by_vector(
                user_vector, n_vectors
            )

        return recommendations


vectors = np.random.rand(int(1e6), 16)
user_vectors = np.random.rand(int(1e4), 16)

quicksim = quicksim()
vector_index = quicksim.build_vector_index(vectors)

recommendations = quicksim.recommend(user_vectors, vectors)
most_similar = quicksim.find_most_similar(vector_index)
