import os
import joblib
from philo_topic_modeling.pipeline import FeaturePipeline
from philo_topic_modeling.config import PROCESSED_DIR

# ensure processed dir exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(PROCESSED_DIR, "tfidf.joblib")


class FeatureExtractor:
    """
    Helper that fetches raw docs from your DatabaseManager and
    then vectorizes them via a FeaturePipeline, with persistence.
    """

    def __init__(self, db, max_df=0.85, min_df=5, ngram_range=(1, 2)):
        self.db = db
        self.pipe = FeaturePipeline(
            max_df=max_df, min_df=min_df, ngram_range=ngram_range
        )

    def fit_transform(self):
        # fetch all docs
        rows = self.db.fetch_all()  # [(id,title,content), …]
        docs = [r[2] for r in rows]
        self.doc_ids = [r[0] for r in rows]

        # fit & persist
        X = self.pipe.fit_transform(docs)
        self.pipe.save(TFIDF_PATH)
        return X

    def load_or_transform(self):
        # try load; if fails, fit_transform
        if os.path.exists(TFIDF_PATH):
            self.pipe.load(TFIDF_PATH)
            rows = self.db.fetch_all()
            docs = [r[2] for r in rows]
            self.doc_ids = [r[0] for r in rows]
            return self.pipe.transform(docs)
        else:
            return self.fit_transform()

    def transform(self, new_docs):
        return self.pipe.transform(new_docs)

    def get_vectorizer(self):
        return self.pipe.pipe.named_steps["tfidf"]
