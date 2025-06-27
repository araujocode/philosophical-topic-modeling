#!/usr/bin/env python3
"""
experiments_supervised.py
=========================

• grid-search best **alpha** for SGD (log-loss)          → verbose progress
• grid-search best **C** for LogisticRegression          → verbose progress
• online SGD for *epochs*  → learning_curve.png          (tqdm bar)
• soft-voting ensemble (LogReg + ComplementNB)           → ensemble_classifier.joblib
• confusion_matrix.png & classification_report.csv

CLI flags
---------
--epochs       (default 20)     online epochs for SGD
--test-size    (default 0.20)   validation split
--alpha        initial alpha if --no-grid
--no-grid      skip grid-searches
--out-dir      where diagnostics & models are written
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

from philo_topic_modeling.config import DB_PATH
from philo_topic_modeling.db import DatabaseManager
from philo_topic_modeling.features import FeatureExtractor


# ──────────────────────────────────────────────────────────────────
class SupervisedExperiment:
    # ---------------------------------------------------------- #
    def __init__(
        self,
        *,
        db_path: str,
        epochs: int,
        test_size: float,
        alpha: float,
        out_dir: str,
        random_state: int = 42,
        use_grid: bool = True,
    ):
        self.db_path = db_path
        self.epochs = epochs
        self.test_size = test_size
        self.alpha_init = alpha
        self.random_state = random_state
        self.use_grid = use_grid

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- #
    def load_data(self) -> None:
        ids, titles, texts, cats = zip(*DatabaseManager(self.db_path).fetch_all())

        keep = [i for i, c in enumerate(cats) if c]  # only labelled rows
        if not keep:
            raise RuntimeError("No labelled rows in the DB.")

        X_full = FeatureExtractor(DatabaseManager(self.db_path)).load_or_transform()
        X = X_full[keep]
        labels = [cats[i] for i in keep]

        # drop categories with < 3 docs
        self.le = LabelEncoder()
        y_full = self.le.fit_transform(labels)
        overall_counts = Counter(y_full)
        mask = np.array([overall_counts[c] >= 3 for c in y_full])
        X, y = X[mask], y_full[mask]

        if len(np.unique(y)) < 2:
            raise RuntimeError("Need at least 2 categories with ≥3 documents each.")

        # train / validation split
        self.X_tr, self.X_va, self.y_tr, self.y_va = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # CV folds: at least 2, at most 3 (to avoid warnings)
        train_counts = Counter(self.y_tr)
        self.cv_splits = max(2, min(3, min(train_counts.values())))

        print(
            f"✅ data  {self.X_tr.shape[0]} train / {self.X_va.shape[0]} val  "
            f"({len(train_counts)} classes, CV={self.cv_splits}-fold)"
        )

    # ---------------------------------------------------------- #
    def best_alpha(self) -> float:
        if not self.use_grid:
            return self.alpha_init

        print("🔍  Grid-searching alpha for SGD …")
        search = GridSearchCV(
            SGDClassifier(
                loss="log_loss",
                learning_rate="optimal",
                random_state=self.random_state,
            ),
            param_grid={"alpha": [1e-5, 1e-4, 1e-3]},
            cv=self.cv_splits,
            n_jobs=-1,
            verbose=2,  # progress feedback
        )
        search.fit(self.X_tr, self.y_tr)
        alpha = search.best_params_["alpha"]
        print(f"🔧 best alpha = {alpha}")
        return alpha

    # ---------------------------------------------------------- #
    def best_C(self) -> float:
        if not self.use_grid:
            return 1.0

        print("🔍  Grid-searching C for LogisticRegression …")
        search = GridSearchCV(
            LogisticRegression(
                max_iter=1000,
                solver="saga",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            param_grid={"C": [0.1, 1.0, 10.0]},
            cv=self.cv_splits,
            n_jobs=-1,
            verbose=2,  # progress feedback
        )
        search.fit(self.X_tr, self.y_tr)
        C = search.best_params_["C"]
        print(f"🔧 best C     = {C}")
        return C

    # ---------------------------------------------------------- #
    def train(self, alpha: float) -> None:
        print("\n🚀 training SGD (online)…")
        sgd = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            learning_rate="optimal",
            random_state=self.random_state,
        )
        classes = np.unique(self.y_tr)
        self.train_acc, self.val_acc = [], []

        for _ in trange(1, self.epochs + 1, desc="SGD epochs"):
            sgd.partial_fit(self.X_tr, self.y_tr, classes=classes)
            self.train_acc.append(sgd.score(self.X_tr, self.y_tr))
            self.val_acc.append(sgd.score(self.X_va, self.y_va))

        self.clf = sgd
        self._plot_curve()

    # ---------------------------------------------------------- #
    def _plot_curve(self) -> None:
        plt.figure()
        plt.plot(range(1, self.epochs + 1), self.train_acc, label="train")
        plt.plot(range(1, self.epochs + 1), self.val_acc, label="val")
        plt.xlabel("epoch"), plt.ylabel("accuracy")
        plt.title("SGD learning curve")
        plt.legend()
        plt.savefig(self.out_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---------------------------------------------------------- #
    def train_ensemble(self, C: float) -> None:
        print("\n🤝 fitting soft-voting ensemble (LogReg + ComplementNB)…")
        nb = ComplementNB(alpha=0.5)
        lr = LogisticRegression(
            C=C,
            solver="saga",
            max_iter=1000,
            n_jobs=-1,
            random_state=self.random_state,
        )
        ens = VotingClassifier([("lr", lr), ("nb", nb)], voting="soft", n_jobs=-1)
        ens.fit(self.X_tr, self.y_tr)
        print(
            f"📈 ensemble validation accuracy = {ens.score(self.X_va, self.y_va):.3f}"
        )
        self.ensemble = ens

    # ---------------------------------------------------------- #
    def diagnostics(self) -> None:
        preds = self.ensemble.predict(self.X_va)

        cm = confusion_matrix(self.y_va, preds)
        ConfusionMatrixDisplay(cm, display_labels=self.le.classes_).plot(
            cmap="Blues", xticks_rotation=90
        )
        plt.tight_layout()
        plt.savefig(self.out_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        report = classification_report(
            self.y_va,
            preds,
            target_names=self.le.classes_,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            self.out_dir / "classification_report.csv"
        )
        print("📊 saved confusion_matrix.png & classification_report.csv")

    # ---------------------------------------------------------- #
    def save_model(self) -> None:
        joblib.dump(
            {"model": self.clf, "label_encoder": self.le},
            self.out_dir / "sgd_text_classifier.joblib",
        )
        joblib.dump(
            {"model": self.ensemble, "label_encoder": self.le},
            self.out_dir / "ensemble_classifier.joblib",
        )
        json.dump({"epochs": self.epochs}, open(self.out_dir / "params.json", "w"))
        print("💾 models saved")

    # ---------------------------------------------------------- #
    def run(self) -> None:
        self.load_data()
        alpha = self.best_alpha()
        C = self.best_C()
        self.train(alpha)
        self.train_ensemble(C)
        self.diagnostics()
        self.save_model()


# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised SEP category predictor")
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument(
        "--alpha", type=float, default=1e-4, help="initial alpha if --no-grid"
    )
    p.add_argument("--out-dir", default="experiments")
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="skip grid-search (use given alpha & C=1)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SupervisedExperiment(
        db_path=args.db_path,
        epochs=args.epochs,
        test_size=args.test_size,
        alpha=args.alpha,
        out_dir=args.out_dir,
        use_grid=not args.no_grid,
    ).run()
