import os
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA

from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.features import FeatureExtractor
from philo_topic_modeling.topic_model import TopicModeler
from philo_topic_modeling.cluster import Clusterer
from philo_topic_modeling.config import DB_PATH, N_TOPICS, N_CLUSTERS, PROCESSED_DIR

# ensure processed dir exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


class StreamlitApp:
    def __init__(self):
        st.title("Philosophical Topic Modeling")

        # --- Load documents from SQLite ---
        db = DatabaseManager(DB_PATH)
        try:
            rows = db.fetch_all()
            if not rows:
                st.warning("⚠️ No documents found. Run `make scrape` first.")
                return

            self.ids, self.titles, self.texts = zip(*rows)

            # --- Feature extraction (load or fit & save) ---
            feat = FeatureExtractor(db, max_df=0.85, min_df=5, ngram_range=(1, 2))
            X = feat.load_or_transform()
            if X.shape[0] == 0:
                st.warning("⚠️ Feature extraction returned no data.")
                return

            # --- Sidebar controls ---
            method = st.sidebar.selectbox("Topic Model", ["lda", "nmf"])
            n_topics = st.sidebar.slider("Number of Topics", 5, 20, N_TOPICS)
            clust_method = st.sidebar.selectbox("Clustering", ["kmeans", "agg"])
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, N_CLUSTERS)

            # --- Topic modeling (load/save inside) ---
            tm = TopicModeler(n_topics=n_topics, method=method)
            theta = tm.fit_transform(X)

            # get TF–IDF vectorizer to extract terms
            vectorizer = feat.get_vectorizer()
            topics = tm.get_topics(vectorizer, n_top=8)

            # --- Clustering on topic vectors ---
            cl = Clusterer(method=clust_method, n_clusters=n_clusters)
            labels = cl.fit_predict(theta)

            # --- Display topics ---
            st.header("Top Terms per Topic")
            cols = st.columns(2)
            for i, words in enumerate(topics):
                with cols[i % 2]:
                    st.subheader(f"Topic {i+1}")
                    st.markdown(", ".join(f"`{w}`" for w in words))

            # --- 2D projection of documents in topic-space ---
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(theta)
            df = pd.DataFrame(coords, columns=["x", "y"])
            df["title"] = self.titles
            df["cluster"] = labels.astype(str)

            st.header("Document Clusters (PCA projection)")
            chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    color="cluster:N",
                    tooltip=["title:N", "cluster:N"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        finally:
            db.close()


if __name__ == "__main__":
    StreamlitApp()
