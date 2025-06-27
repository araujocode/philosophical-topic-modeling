from sklearn.cluster import KMeans, AgglomerativeClustering


class Clusterer:
    def __init__(self, method="kmeans", n_clusters=5):
        if method == "kmeans":
            self.clust = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            self.clust = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_predict(self, X):
        return self.clust.fit_predict(X)
