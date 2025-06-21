import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_embeddings(embeddings, labels, method='pca', title=''):
    embeddings = np.array(embeddings)
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=1, max_iter=1000)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(6, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=label)
        plt.annotate(label, (reduced[i, 0], reduced[i, 1] + 0.01))
    plt.title(f"{method.upper()} - {title}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
