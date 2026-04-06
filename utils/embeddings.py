import numpy as np

def get_embeddings(tokens):
    np.random.seed(42)  # consistent results
    embeddings = {}

    for word in tokens:
        embeddings[word] = np.random.rand(50)

    return embeddings

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings_pca(embeddings):
    words = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure()

    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0], reduced[i, 1], word)

    plt.title("Word Embeddings (PCA 2D)")
    return plt

from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embeddings):
    words = list(embeddings.keys())
    vectors = list(embeddings.values())

    sim_matrix = cosine_similarity(vectors)

    return words, sim_matrix

import seaborn as sns

def plot_similarity(words, sim_matrix):
    plt.figure()
    sns.heatmap(sim_matrix, xticklabels=words, yticklabels=words, annot=True)
    plt.title("Word Similarity Heatmap")
    return plt

from sklearn.cluster import KMeans

def plot_clusters(embeddings):
    words = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))

    kmeans = KMeans(n_clusters=min(3, len(words)), n_init=10)
    labels = kmeans.fit_predict(vectors)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure()

    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0], reduced[i, 1], f"{word} (C{labels[i]})")

    plt.title("Word Clusters")
    return plt