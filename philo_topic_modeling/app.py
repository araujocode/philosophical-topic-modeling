import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.pipeline import FeaturePipeline
from philo_topic_modeling.topic_model import TopicModeler
from philo_topic_modeling.cluster import Clusterer
from philo_topic_modeling.config import DB_PATH, N_TOPICS, N_CLUSTERS


class StreamlitApp:
    def __init__(self):
        st.title("Philosophical Topic Modeling")
        self.db = DatabaseManager(DB_PATH)
        docs = self.db.fetch_all()
        self.ids, self.titles, self.texts = zip(*docs)
        self.feature_pipe = FeaturePipeline()
        self.X = self.feature_pipe.fit_transform(self.texts)

        # Sidebar
        method = st.sidebar.selectbox("Topic Model", ["lda", "nmf"])
        n_topics = st.sidebar.slider("Number of Topics", 5, 20, N_TOPICS)
        clust_method = st.sidebar.selectbox("Clustering", ["kmeans", "agg"])
        n_clusters = st.sidebar.slider("Clusters", 2, 10, N_CLUSTERS)

        # Modeling
        tm = TopicModeler(n_topics=n_topics, method=method)
        theta = tm.fit_transform(self.X)
        topics = tm.get_topics(self.feature_pipe.pipe.named_steps["tfidf"], n_top=8)

        # Clustering
        cl = Clusterer(method=clust_method, n_clusters=n_clusters)
        labels = cl.fit_predict(theta)

        # Display topics
        for i, words in enumerate(topics):
            st.subheader(f"Topic {i}")
            st.write(", ".join(words))

        # 2D projection
        pca = PCA(2)
        coords = pca.fit_transform(theta)
        df = pd.DataFrame(coords, columns=["x", "y"])
        df["title"] = self.titles
        df["cluster"] = labels
        st.write("### Document Clusters")
        st.altair_chart(
            (st.altair_chart(st.altair_chart(st.altair_chart(st.altair_chart(None)))))
        )
        st.map()


if __name__ == "__main__":
    StreamlitApp()
