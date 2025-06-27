from sklearn.cluster import KMeans, AgglomerativeClustering


class Clusterer:
    """
    Wraps KMeans or AgglomerativeClustering via `method` arg.
    """

    def __init__(self, method="kmeans", n_clusters=5):
        method = method.lower()
        if method == "kmeans":
            self.clust = KMeans(n_clusters=n_clusters, random_state=42)
        elif method in ("agg", "agglomerative"):
            self.clust = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method '{method}'")

    def fit_predict(self, X):
        """Fit the clustering model and return labels."""
        return self.clust.fit_predict(X)
