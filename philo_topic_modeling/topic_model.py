import joblib
from sklearn.decomposition import LatentDirichletAllocation, NMF


class TopicModeler:
    def __init__(self, n_topics=10, method="lda"):
        self.n_topics = n_topics
        if method == "lda":
            self.model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42
            )
        else:
            self.model = NMF(n_components=n_topics, init="nndsvd", random_state=42)

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def get_topics(self, vectorizer, n_top=10):
        terms = vectorizer.get_feature_names_out()
        topics = []
        for comp in self.model.components_:
            top_idxs = comp.argsort()[-n_top:][::-1]
            topics.append([terms[i] for i in top_idxs])
        return topics

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
