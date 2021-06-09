# cd /Users/marnix_koops/projects/c1/customerone/src/
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from customerone.pipelines.retail.recommendation.nodes.matrix_factorization import (
    alternating_least_squares,
    fit,
)
from customerone.pipelines.retail.recommendation.utils import (
    create_mapping_dictionaries,
    create_sparse_interaction_matrix,
    create_train_test_matrix,
    encode_interaction_data,
    map_back_indexes,
)

sns.set_style("darkgrid")
sns.set_palette("crest_r")
sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300, "figure.figsize": (8, 8)})
sns.set_context("notebook", font_scale=0.8)

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO)
################################################################


ratings = pd.read_csv(
    "/Users/marnix_koops/projects/c1/customerone/src/customerone/pipelines/retail/recommendation/data/movielens/ratings.dat",
    delimiter="::",
    names=["userId", "movieId", "rating", "timestamp"],
    engine="python",
)

movies = pd.read_table(
    "/Users/marnix_koops/projects/c1/customerone/src/customerone/pipelines/retail/recommendation/data/movielens/movies.dat",
    delimiter="::",
    names=["movieId", "title", "genres"],
    engine="python",
)

encoded_interaction_data = encode_interaction_data(
    data=ratings,
    user_col="userId",
    item_col="movieId",
    interaction_col="rating",
    interaction_to_binary=False,
)

index_to_user, index_to_item = create_mapping_dictionaries(
    encoded_interaction_data, user_col="userId", item_col="movieId"
)

sparse_interaction_matrix = create_sparse_interaction_matrix(encoded_interaction_data)
train_matrix, test_matrix = create_train_test_matrix(sparse_interaction_matrix)

model = alternating_least_squares(
    num_factors=16, regularization=0.05, num_iterations=48, random_state=2021
)
model = fit(model, sparse_interaction_matrix)

user_vectors = model.user_factors
item_vectors = model.item_factors

item_data = pd.DataFrame.from_dict(
    index_to_item, orient="index", columns=["movieId"]
).reset_index()
item_data.columns = ["item_id", "movieId"]
item_data = item_data.merge(movies)


genre_tags = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

num_factors = 16
median_genre_vectors = np.zeros(shape=[len(genre_tags), num_factors])
for row, tag in enumerate(genre_tags):
    index = item_data[item_data["genres"].str.contains(f"{tag}")]["item_id"].values
    vectors = item_vectors[index]
    median_vector = np.median(vectors, axis=0)
    median_genre_vectors[row] = median_vector


def plot_vector(vector):
    plt.figure(figsize=(8, 8), facecolor="#F9FAFC")
    plt.rcParams["figure.dpi"] = 300
    sns.heatmap(
        vector.reshape(4, 4), annot=False, cmap="flare", square=True, cbar=False,
    )
    plt.axis("off")
    plt.title("genre_tag[x] \n Median Embedding Vector", weight="bold", color="#404040")


plot_vector(median_genre_vectors[11])


def plot_mean_embedding_vectors():
    plt.rcParams["figure.dpi"] = 300
    fig, axs = plt.subplots(figsize=(10, 6), ncols=6, nrows=3, facecolor="#F9FAFC")
    plt.suptitle("Median Embedding Tag Vectors", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[0].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 0],
    ).set_title(f"{genre_tags[0]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[1].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 1],
    ).set_title(f"{genre_tags[1]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[2].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 2],
    ).set_title(f"{genre_tags[2]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[3].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 3],
    ).set_title(f"{genre_tags[3]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[4].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 4],
    ).set_title(f"{genre_tags[4]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[5].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[0, 5],
    ).set_title(f"{genre_tags[5]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[6].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 0],
    ).set_title(f"{genre_tags[6]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[7].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 1],
    ).set_title(f"{genre_tags[7]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[8].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 2],
    ).set_title(f"{genre_tags[8]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[9].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 3],
    ).set_title(f"{genre_tags[9]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[10].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 4],
    ).set_title(f"{genre_tags[10]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[11].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[1, 5],
    ).set_title(f"{genre_tags[11]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[12].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 0],
    ).set_title(f"{genre_tags[12]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[13].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 1],
    ).set_title(f"{genre_tags[13]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[14].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 2],
    ).set_title(f"{genre_tags[14]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[15].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 3],
    ).set_title(f"{genre_tags[15]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[16].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 4],
    ).set_title(f"{genre_tags[16]}", weight="bold", color="#001A4C")
    sns.heatmap(
        median_genre_vectors[17].reshape(6, 6),
        annot=False,
        cmap="flare",
        square=True,
        cbar=False,
        ax=axs[2, 5],
    ).set_title(f"{genre_tags[17]}", weight="bold", color="#001A4C")
    [ax.set_axis_off() for ax in axs.ravel()]
    plt.tight_layout()


plot_mean_embedding_vectors()

item_vectors

item_data[item_data["genres"].str.contains("Animation")][0:50]

item_vectors[0]

plot_vector(item_vectors[0])
