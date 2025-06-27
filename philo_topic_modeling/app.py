import os
import streamlit as st
import pandas as pd
import altair as alt
from joblib import load, dump
from wordcloud import WordCloud  # NEW
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

# ────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)
TFIDF_PATH = os.path.join(PROCESSED_DIR, "tfidf_pipeline.joblib")
MODEL_PATH_TMPL = os.path.join(PROCESSED_DIR, "topics_{method}.joblib")

EXP_DIR = "experiments"
LCURVE_PNG = os.path.join(EXP_DIR, "learning_curve.png")
ENSEMBLE_PATH = os.path.join(EXP_DIR, "ensemble_classifier.joblib")
CM_PNG = os.path.join(EXP_DIR, "confusion_matrix.png")  # NEW
REP_CSV = os.path.join(EXP_DIR, "classification_report.csv")  # NEW


class StreamlitApp:
    """
    Streamlit front-end for both unsupervised and supervised parts.
    """

    def __init__(self):
        st.title("Philosophical Topic Modeling")

        # ────────────────────────────────────────────────────────────
        # Sidebar housekeeping
        # ────────────────────────────────────────────────────────────
        if st.sidebar.button("🔄 Re-fit all models"):
            for p in [TFIDF_PATH] + [
                MODEL_PATH_TMPL.format(method=m) for m in ("lda", "nmf")
            ]:
                if os.path.exists(p):
                    os.remove(p)
            st.sidebar.success("Cleared cached models. Reloading …")
            st.experimental_rerun()

        # ────────────────────────────────────────────────────────────
        # Load DB rows
        # ────────────────────────────────────────────────────────────
        db = DatabaseManager(DB_PATH)
        try:
            rows = db.fetch_all()  # (id, title, content, category)
            if not rows:
                st.warning("⚠️ No documents found. Run `make scrape` first.")
                return

            ids, titles, texts, cats = zip(*rows)

            # ────────────────────────────────────────────────────────
            # Feature extraction
            # ────────────────────────────────────────────────────────
            feat = FeatureExtractor(db, max_df=0.85, min_df=5, ngram_range=(1, 2))
            X = self._load_or_fit_tfidf(feat, texts)

            # ────────────────────────────────────────────────────────
            # Sidebar controls
            # ────────────────────────────────────────────────────────
            method = st.sidebar.selectbox("Topic model", ["lda", "nmf"])
            n_topics = st.sidebar.slider("Number of topics", 5, 20, N_TOPICS)
            clust_method = st.sidebar.selectbox("Clustering", ["kmeans", "agg"])
            n_clusters = st.sidebar.slider("Number of clusters", 2, 10, N_CLUSTERS)

            colour_opts = ["cluster", "category"]
            if os.path.exists(ENSEMBLE_PATH):
                colour_opts.append("predicted")
            colour_by = st.sidebar.radio("Colour points by", colour_opts)

            # ────────────────────────────────────────────────────────
            # Topic modeling
            # ────────────────────────────────────────────────────────
            theta, topics, tm = self._load_or_fit_topic_model(
                X, feat.get_vectorizer(), n_topics, method
            )

            # ────────────────────────────────────────────────────────
            # Clustering
            # ────────────────────────────────────────────────────────
            cl = Clusterer(method=clust_method, n_clusters=n_clusters)
            labels = cl.fit_predict(theta)

            # ────────────────────────────────────────────────────────
            # Predicted category (if ensemble available)
            # ────────────────────────────────────────────────────────
            if os.path.exists(ENSEMBLE_PATH):
                ens_bundle = load(ENSEMBLE_PATH)  # {'model':…, 'label_encoder':…}
                y_pred_idx = ens_bundle["model"].predict(X)
                y_pred = ens_bundle["label_encoder"].inverse_transform(y_pred_idx)
                pred_acc = (pd.Series(y_pred) == pd.Series(cats)).mean()
                st.sidebar.markdown(f"**Ensemble acc.: `{pred_acc:.2%}`**")
            else:
                y_pred = ["n/a"] * len(titles)

            # ────────────────────────────────────────────────────────
            # Top terms per topic
            # ────────────────────────────────────────────────────────
            st.header("Top terms per topic")
            cols = st.columns(2)
            for i, words in enumerate(topics):
                with cols[i % 2]:
                    st.subheader(f"Topic {i+1}")
                    st.markdown(", ".join(f"`{w}`" for w in words))

            # ────────────────────────────────────────────────────────
            # Word-clouds (optional)
            # ────────────────────────────────────────────────────────
            if st.checkbox("🎨 Show topic word-clouds"):
                for i, comp in enumerate(tm.topic_term_):
                    top_idx = comp.argsort()[-40:][::-1]
                    freqs = {
                        feat.get_vectorizer().get_feature_names_out()[j]: comp[j]
                        for j in top_idx
                    }
                    wc = WordCloud(
                        width=400, height=250, background_color="white"
                    ).generate_from_frequencies(freqs)
                    st.image(wc.to_array(), caption=f"Topic {i+1}")

            # ────────────────────────────────────────────────────────
            # 2-D PCA scatter
            # ────────────────────────────────────────────────────────
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(theta)
            df = pd.DataFrame(
                {
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "title": titles,
                    "cluster": labels.astype(str),
                    "category": [c or "unknown" for c in cats],
                    "predicted": y_pred,
                }
            )

            st.header("Document map (PCA projection)")
            colour_field = alt.Color(
                f"{colour_by}:N", legend=alt.Legend(title=colour_by.capitalize())
            )
            tooltip_cols = ["title:N", "cluster:N", "category:N"]
            if "predicted" in colour_opts:
                tooltip_cols.append("predicted:N")

            chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    color=colour_field,
                    tooltip=tooltip_cols,
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            # ────────────────────────────────────────────────────────
            # Learning-curve
            # ────────────────────────────────────────────────────────
            if os.path.exists(LCURVE_PNG):
                with st.expander("📈 Training vs. validation curve (SGD)"):
                    st.image(
                        LCURVE_PNG,
                        caption="Accuracy over epochs",
                        use_column_width=True,
                    )

            # ────────────────────────────────────────────────────────
            # Confusion-matrix & per-class metrics
            # ────────────────────────────────────────────────────────
            if os.path.exists(CM_PNG):
                with st.expander("🔍 Confusion matrix & class metrics"):
                    st.image(CM_PNG, use_column_width=True)
                    st.dataframe(pd.read_csv(REP_CSV, index_col=0).round(3))
            else:
                st.info("Run the supervised script to generate confusion-matrix stats.")

        finally:
            db.close()

    # ───────────────────────── helpers ────────────────────────── #
    def _load_or_fit_tfidf(self, feat, texts):
        if os.path.exists(TFIDF_PATH):
            with st.spinner("Loading TF-IDF pipeline…"):
                feat.pipe.load(TFIDF_PATH)
                return feat.transform(texts)
        with st.spinner("Fitting TF-IDF pipeline…"):
            X = feat.fit_transform()
            feat.pipe.save(TFIDF_PATH)
            st.success(f"✅ Saved TF-IDF → {TFIDF_PATH}")
            return X

    def _load_or_fit_topic_model(self, X, vectorizer, n_topics, method):
        path = MODEL_PATH_TMPL.format(method=method)
        tm = TopicModeler(n_topics=n_topics, method=method)

        if os.path.exists(path):
            with st.spinner(f"Loading {method.upper()}…"):
                tm.model = load(path)
                theta = tm.model.transform(X)
                tm.topic_term_ = tm.model.components_
        else:
            with st.spinner(f"Fitting {method.upper()}…"):
                theta = tm.fit_transform(X)
                dump(tm.model, path)
            st.success(f"✅ Saved {method.upper()} → {path}")

        return theta, tm.get_topics(vectorizer, n_top=8), tm


if __name__ == "__main__":
    StreamlitApp()
