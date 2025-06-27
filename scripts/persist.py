"""
scripts/persist.py

Fits the TF–IDF pipeline and both LDA/NMF topic models, then writes all
artifacts to data/processed/.
"""

import os
import sys
from time import perf_counter

# --------------------------------------------------------------------- #
# Ensure project root is on PYTHONPATH
# --------------------------------------------------------------------- #
PROJ_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJ_ROOT)

from philo_topic_modeling.config import DB_PATH, N_TOPICS
from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.features import FeatureExtractor
from philo_topic_modeling.topic_model import TopicModeler


def main() -> None:
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------- 1/5 ---------------------------------- #
    print("\n🔍 1/5  Loading documents from database …")
    tic = perf_counter()
    db = DatabaseManager(DB_PATH)
    fe = FeatureExtractor(db)
    rows = db.fetch_all()
    print(f"    → {len(rows)} documents   (took {perf_counter()-tic:.2f}s)")

    # --------------------------- 2/5 ---------------------------------- #
    print("\n🔢 2/5  Fitting TF–IDF pipeline …")
    tic = perf_counter()
    X = fe.fit_transform()
    print(f"    → matrix shape {X.shape}   (took {perf_counter()-tic:.2f}s)")

    print("💾     Saving TF–IDF pipeline")
    tfidf_path = os.path.join(out_dir, "tfidf_pipeline.joblib")
    fe.pipe.save(tfidf_path)
    print(f"    ✅ {tfidf_path}")

    # --------------------- 3/5 & 4/5 ---------------------------------- #
    for step, method in enumerate(("lda", "nmf"), start=3):
        print(f"\n🎯 {step}/5  Fitting {method.upper()} (n_topics={N_TOPICS}) …")
        tic = perf_counter()
        tm = TopicModeler(n_topics=N_TOPICS, method=method)
        theta = tm.fit_transform(X)
        print(
            f"    → doc–topic matrix {theta.shape}   "
            f"(took {perf_counter()-tic:.2f}s)"
        )

        print("💾     Saving model")
        model_path = os.path.join(out_dir, f"topics_{method}.joblib")
        tm.save(model_path)
        print(f"    ✅ {model_path}")

    # --------------------------- 5/5 ---------------------------------- #
    db.close()
    print(
        "\n🏁 5/5  Persistence complete — artifacts are in "
        f"{os.path.relpath(out_dir)}\n"
    )


if __name__ == "__main__":
    main()
