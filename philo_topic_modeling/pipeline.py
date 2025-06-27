import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class FeaturePipeline:
    def __init__(self, max_df=0.85, min_df=5, ngram_range=(1, 2)):
        self.pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_df=max_df, min_df=min_df, ngram_range=ngram_range
                    ),
                )
            ]
        )

    def fit_transform(self, docs):
        return self.pipe.fit_transform(docs)

    def transform(self, docs):
        return self.pipe.transform(docs)

    def save(self, path):
        joblib.dump(self.pipe, path)

    def load(self, path):
        self.pipe = joblib.load(path)
