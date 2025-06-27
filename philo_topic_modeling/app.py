import os
import streamlit as st
import pandas as pd
import altair as alt
from joblib import load, dump
from sklearn.decomposition import PCA

from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.features import FeatureExtractor
from philo_topic_modeling.topic_model import TopicModeler
from philo_topic_modeling.cluster import Clusterer
from philo_topic_modeling.config import (
    DB_PATH,
    N_TOPICS,
    N_CLUSTERS,
    PROCESSED_DIR,
)

# 1. Ensure processed-artifacts directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Define paths for persisted artifacts
TFIDF_PATH = os.path.join(PROCESSED_DIR, "tfidf_pipeline.joblib")
MODEL_PATH_TMPL = os.path.join(PROCESSED_DIR, "topics_{method}.joblib")


class StreamlitApp:
    def __init__(self):
        st.title("Philosophical Topic Modeling")

        # Sidebar: clear persisted artifacts and restart
        if st.sidebar.button("🔄 Re-fit all models"):
            for p in [TFIDF_PATH] + [
                MODEL_PATH_TMPL.format(method=m) for m in ("lda", "nmf")
            ]:
                if os.path.exists(p):
                    os.remove(p)
            st.sidebar.success("Cleared persisted models. Reload to re-fit.")
            st.experimental_rerun()

        # Load documents
        db = DatabaseManager(DB_PATH)
        try:
            rows = db.fetch_all()
            if not rows:
                st.warning("⚠️ No documents found. Run `make scrape` first.")
                return

            ids, titles, texts = zip(*rows)

            # Feature extraction (load or fit & save)
            feat = FeatureExtractor(db, max_df=0.85, min_df=5, ngram_range=(1, 2))
            X = self._load_or_fit_tfidf(feat, texts)

            # Sidebar controls
            method = st.sidebar.selectbox("Topic Model", ["lda", "nmf"])
            n_topics = st.sidebar.slider("Number of Topics", 5, 20, N_TOPICS)
            clust_method = st.sidebar.selectbox("Clustering", ["kmeans", "agg"])
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, N_CLUSTERS)

            # Topic modeling (load or fit & save)
            theta, topics = self._load_or_fit_topic_model(
                X, feat.get_vectorizer(), n_topics, method
            )

            # Clustering
            cl = Clusterer(method=clust_method, n_clusters=n_clusters)
            labels = cl.fit_predict(theta)

            # Display top‐terms per topic
            st.header("Top Terms per Topic")
            cols = st.columns(2)
            for i, words in enumerate(topics):
                with cols[i % 2]:
                    st.subheader(f"Topic {i+1}")
                    st.markdown(", ".join(f"`{w}`" for w in words))

            # 2D PCA projection of documents in topic‐space
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(theta)
            df = pd.DataFrame(coords, columns=["x", "y"])
            df["title"] = titles
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

    def _load_or_fit_tfidf(self, feat: FeatureExtractor, texts):
        """Load or fit & save the TF–IDF pipeline, with spinner feedback."""
        if os.path.exists(TFIDF_PATH):
            with st.spinner("Loading saved TF–IDF pipeline…"):
                feat.pipe.load(TFIDF_PATH)
                X = feat.transform(texts)
        else:
            with st.spinner("Fitting TF–IDF pipeline…"):
                X = feat.fit_transform()
                feat.pipe.save(TFIDF_PATH)
            st.success(f"✅ Saved TF–IDF pipeline to {TFIDF_PATH}")
        return X

    def _load_or_fit_topic_model(self, X, vectorizer, n_topics, method):
        """
        Load or fit & save the topic model, then return (θ, top_terms),
        all wrapped in a spinner.
        """
        model_path = MODEL_PATH_TMPL.format(method=method)
        tm = TopicModeler(n_topics=n_topics, method=method)

        if os.path.exists(model_path):
            with st.spinner(f"Loading saved {method.upper()} model…"):
                tm.model = load(model_path)
                theta = tm.model.transform(X)
                tm.topic_term_ = tm.model.components_
        else:
            with st.spinner(f"Fitting {method.upper()} model…"):
                theta = tm.fit_transform(X)
                dump(tm.model, model_path)
            st.success(f"✅ Saved {method.upper()} model to {model_path}")

        topics = tm.get_topics(vectorizer, n_top=8)
        return theta, topics


if __name__ == "__main__":
    StreamlitApp()
