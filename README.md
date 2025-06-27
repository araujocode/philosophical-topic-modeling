# Philo Topic Modeling

*A fully-reproducible pipeline & Streamlit app for mining the Stanford Encyclopedia of Philosophy.*

Unsupervised **and** supervised:

* **LDA / NMF** topic discovery
* **K-Means / Agglomerative** clustering
* **Word-clouds** for every topic
* **SGD text classifier** *(tuned by grid-search)* that predicts a top-level **category** label
* **Soft-voting Ensemble** (LogReg + ComplementNB)
* Training vs validation curve **and** confusion-matrix heat-map

---

## Summary

This guide walks you through:

1. Cloning the repo
2. Creating & activating a Python virtual environment
3. Installing dependencies (with Windows-specific build tips)
4. Populating the SQLite database with SEP entries **& category labels**
5. Persisting TF–IDF and topic models to disk
6. Training the supervised models & exporting diagnostics
7. Launching the Streamlit app (with lazy loading of models)
8. Running tests
9. Troubleshooting common Windows issues

---

## About This Project

**Philo Topic Modeling** is a lightweight, end-to-end framework for exploring the SEP through modern NLP.

### What it does

1. **Scrapes** every published SEP entry plus its first “Related Entries” link → used as a *category* label.
2. **Stores** the corpus in `data/sep.db` (SQLite).
3. **Vectorizes** text with a reusable TF-IDF pipeline.
4. **Discovers** latent themes via **LDA** or **NMF**.
5. **Clusters** documents on their topic vectors.
6. **Classifies** each article’s category with an online **SGD** model (log-loss), tuned by 3-fold grid-search.
7. **Ensembles** complementary classifiers (LogReg + ComplementNB) for extra accuracy.
8. **Visualizes** top terms, topic word-clouds, 2-D PCA scatter, learning curve, and a confusion matrix in Streamlit.

### Why it exists

* Make large-scale philosophy text analysis turnkey on a laptop.
* Provide a clean OO codebase for teaching (scraper → features → models → UI).
* Demonstrate best practices for scraping, persistence, and interactive visualisation.

---

## 1. Clone & Prepare Environment

```bash
git clone [https://github.com/araujocode/philosophical_topic_modeling.git](https://github.com/araujocode/philosophical_topic_modeling.git)
cd philosophical_topic_modeling
```

### 1.1. Create & Activate Virtual Environment

| Platform                 | Command                                            |
| ------------------------ | -------------------------------------------------- |
| **Windows (PowerShell)** | `python -m venv venv ; .\venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `python3 -m venv venv && source venv/bin/activate` |

---

## 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **Pinned wheels:** `scikit-learn==1.4.2`, `wordcloud==1.9.3`, etc.
> Avoids MSVC builds on Windows + Python 3.12.

---

## 3. (Windows only) Install **make**

```powershell
choco install make
```

Then you can simply run `make scrape`, `make persist`, `make run`, `make test`.

---

## 4. Common Windows Errors & Fixes

| Symptom                          | Fix                                                                 |
| -------------------------------- | ------------------------------------------------------------------- |
| Wheel for scikit-learn not found | `pip install scikit-learn==1.4.2`                                   |
| pip self-upgrade blocked         | `python -m pip install --upgrade pip`                               |
| venv activation refused          | Run *once* as Admin:<br>`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 5. Populate the Database

```bash
make scrape          # full crawl (~30 min)
```

* Crawls **published.html**
* Downloads every entry → extracts *title*, *content*, *category*
* Inserts into **`data/sep.db`**

---

## 6. Persist Unsupervised Artifacts *(optional)*

```bash
make persist
```

Creates:

```
data/processed/
├─ tfidf_pipeline.joblib
├─ topics_lda.joblib
└─ topics_nmf.joblib
```

---

## 7. Train Supervised Models

```bash
python experiments_supervised.py          # 20 epochs by default
# e.g. python experiments_supervised.py --epochs 40 --no-grid
```

Outputs:

```
experiments/
├─ learning_curve.png
├─ confusion_matrix.png
├─ classification_report.csv
├─ sgd_text_classifier.joblib
└─ ensemble_classifier.joblib
```

| Flag          | Purpose                                 |
| ------------- | --------------------------------------- |
| `--epochs`    | SGD epochs (curve length)               |
| `--test-size` | Validation split fraction               |
| `--alpha`     | Initial SGD alpha (if `--no-grid`)      |
| `--no-grid`   | Skip grid-search; use given alpha + C=1 |

---

## 8. Run the Streamlit App

```bash
make run
```

Open **[http://localhost:8501](http://localhost:8501)**

### Sidebar controls

* **Re-fit all models** – clears cached `.joblib` files and retrains
* **Topic model** – LDA / NMF • **# topics** – 5–20
* **Clustering** – KMeans / Agglomerative • **# clusters** – 2–10
* **Colour points by** – `cluster`, `category`, or `predicted`

### Main panel

| Section                              | What you see                                                      |
| ------------------------------------ | ----------------------------------------------------------------- |
| **Top terms per topic** | 8 most salient words for each topic.                              |
| **🎨 Topic word-clouds** (toggle)   | 40-term clouds generated with *wordcloud*.                        |
| **Document map** | 2-D PCA scatter (hover for title, cluster, category, prediction). |
| **SGD learning curve** | Collapsible `learning_curve.png`.                                 |
| **Confusion matrix & class metrics** | Heat-map + precision / recall / F1 table for every category.      |

---

## 9. Run Tests

```bash
make test
```

Covers DB I/O, feature pipeline, topic modeller, clustering, and basic scraper behaviour.

---

## References

* scikit-learn • BeautifulSoup • Streamlit • Altair • WordCloud • SQLite • pytest
