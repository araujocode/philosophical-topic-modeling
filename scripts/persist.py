#!/usr/bin/env python3
"""
scripts/persist.py

Fits the TF–IDF pipeline and both LDA/NMF topic models,
prints progress along the way, and saves everything under data/processed/.
"""

import os
import sys
from time import perf_counter

# 1) Ensure our project root is on PYTHONPATH so imports work:
PROJ_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJ_ROOT)

# 2) Now import your package
from philo_topic_modeling.config import DB_PATH, N_TOPICS
from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.features import FeatureExtractor
from philo_topic_modeling.topic_model import TopicModeler


def main():
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    print("\n🔍 1/5 Loading documents from database…")
    start = perf_counter()
    db = DatabaseManager(DB_PATH)
    fe = FeatureExtractor(db)
    rows = db.fetch_all()
    print(f"   → Found {len(rows)} documents (took {perf_counter()-start:.2f}s).")

    print("\n🔢 2/5 Extracting TF–IDF features…")
    start = perf_counter()
    X = fe.fit_transform()
    print(f"   → Feature matrix: {X.shape} (took {perf_counter()-start:.2f}s)")

    print("\n💾 3/5 Saving TF–IDF pipeline…")
    pipeline_path = os.path.join(out_dir, "tfidf_pipeline.joblib")
    fe.pipe.save(pipeline_path)
    print(f"   ✅ Saved TF–IDF pipeline to {pipeline_path}")

    for idx, method in enumerate(("lda", "nmf"), start=4):
        print(f"\n🎯 {idx}/5 Fitting {method.upper()} (n_topics={N_TOPICS})…")
        start = perf_counter()
        tm = TopicModeler(n_topics=N_TOPICS, method=method)
        theta = tm.fit_transform(X)
        print(
            f"   → document–topic matrix: {theta.shape} (took {perf_counter()-start:.2f}s)"
        )

        print(f"💾 {idx+0.5}/5 Saving {method.upper()} model…")
        model_path = os.path.join(out_dir, f"topics_{method}.joblib")
        tm.save(model_path)
        print(f"   ✅ Saved {method.upper()} model to {model_path}")

    db.close()
    print("\n🏁 6/5 Persistence complete! All artifacts are in data/processed/\n")


if __name__ == "__main__":
    main()
