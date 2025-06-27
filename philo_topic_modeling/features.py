from philo_topic_modeling.pipeline import FeaturePipeline


class FeatureExtractor:
    """
    Helper that fetches raw docs from your DatabaseManager and
    then vectorizes them via a FeaturePipeline.
    """

    def __init__(self, db, **tfidf_kwargs):
        self.db = db
        self.pipe = FeaturePipeline(**tfidf_kwargs)

    def fit_transform(self):
        """
        Fetch all rows from the DB, extract 'content', vectorize, and
        return the document–term matrix.
        """
        rows = self.db.fetch_all()  # [(id, title, content), …]
        if not rows:
            return None  # caller should handle empty case

        self.doc_ids = [r[0] for r in rows]
        docs = [r[2] for r in rows]
        return self.pipe.fit_transform(docs)

    def transform(self, new_docs):
        """Transform new docs into existing TF–IDF space."""
        return self.pipe.transform(new_docs)

    def get_vectorizer(self):
        """Expose the fitted TfidfVectorizer for extracting feature names."""
        return self.pipe.pipe.named_steps["tfidf"]
