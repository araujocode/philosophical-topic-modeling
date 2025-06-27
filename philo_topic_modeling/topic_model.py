import joblib
from sklearn.decomposition import LatentDirichletAllocation, NMF


class TopicModeler:
    """
    A wrapper around sklearn's LDA and NMF topic models.

    Parameters
    ----------
    n_topics : int, default=10
        Number of topics to extract.
    method : {'lda', 'nmf'}, default='lda'
        Which topic model to use: 'lda' for LatentDirichletAllocation,
        'nmf' for NMF.
    """

    def __init__(self, n_topics=10, method="lda"):
        self.n_topics = n_topics
        method = method.lower()
        if method == "lda":
            self.model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, learning_method="batch"
            )
        elif method == "nmf":
            self.model = NMF(
                n_components=n_topics, init="nndsvd", random_state=42, max_iter=200
            )
        else:
            raise ValueError(f"Unknown method '{method}'; choose 'lda' or 'nmf'.")

        # Will be populated after fit_transform:
        self.doc_topic_ = None
        self.topic_term_ = None

    def fit_transform(self, X):
        """
        Fit the topic model to the document–term matrix X
        and return the document–topic matrix.

        Parameters
        ----------
        X : array-like, shape (n_documents, n_terms)
            Document–term matrix (e.g. TF-IDF features).

        Returns
        -------
        doc_topic : array, shape (n_documents, n_topics)
            The per-document topic distributions.
        """
        self.doc_topic_ = self.model.fit_transform(X)
        self.topic_term_ = self.model.components_
        return self.doc_topic_

    def get_topics(self, vectorizer, n_top=10):
        """
        Extract the top terms for each topic.

        Parameters
        ----------
        vectorizer : object
            Fitted vectorizer (must implement get_feature_names_out()).
        n_top : int, default=10
            Number of top terms to return per topic.

        Returns
        -------
        topics : list of lists
            topics[i] is the list of the top n_top terms for topic i.
        """
        if self.topic_term_ is None:
            raise RuntimeError("You must call fit_transform before get_topics.")

        terms = vectorizer.get_feature_names_out()
        topics = []
        for comp in self.topic_term_:
            top_idxs = comp.argsort()[::-1][:n_top]
            topics.append([terms[i] for i in top_idxs])
        return topics

    def save(self, path):
        """
        Persist the trained model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. 'lda_model.joblib') where to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path : str
            File path from which to load the model.
        """
        self.model = joblib.load(path)
        # Reset learned attributes (user should re-fit or inspect .components_ if appropriate)
        self.doc_topic_ = None
        self.topic_term_ = getattr(self.model, "components_", None)
