import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class FeaturePipeline:
    """
    A reusable TF–IDF feature pipeline.
    """

    def __init__(self, max_df=0.85, min_df=5, ngram_range=(1, 2)):
        self.pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_df=max_df,
                        min_df=min_df,
                        ngram_range=ngram_range,
                        stop_words="english",
                        token_pattern=r"(?u)\b\w\w+\b",
                    ),
                )
            ]
        )

    def fit_transform(self, docs):
        """Fit on a list of raw documents and return document–term matrix."""
        return self.pipe.fit_transform(docs)

    def transform(self, docs):
        """Transform new documents into the existing feature space."""
        return self.pipe.transform(docs)

    def save(self, path):
        """Persist the fitted pipeline (including the learned vocabulary)."""
        joblib.dump(self.pipe, path)

    def load(self, path):
        """Restore a previously saved pipeline."""
        self.pipe = joblib.load(path)
