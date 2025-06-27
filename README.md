# Philo Topic Modeling

A Streamlit app for topic modeling of Stanford Encyclopedia of Philosophy entries using LDA/NMF and document clustering.

---

## Summary

This guide walks you through:

1. Cloning the repo  
2. Creating & activating a Python virtual environment  
3. Installing dependencies (with Windows-specific build tips)  
4. Populating the SQLite database with SEP entries  
5. Persisting TF–IDF and topic models to disk  
6. Launching the Streamlit app (with lazy loading of models)  
7. Running tests  
8. Troubleshooting common Windows issues  

---

## About This Project

**Philo Topic Modeling** is a lightweight, end-to-end framework for exploring the Stanford Encyclopedia of Philosophy (SEP) through unsupervised text-analysis techniques:

- **What it does:**  
  1. **Scrapes** full-text SEP articles.  
  2. **Stores** them in `data/sep.db` (SQLite).  
  3. **Vectorizes** content with a TF–IDF pipeline.  
  4. **Discovers** latent themes via LDA or NMF topic models.  
  5. **Groups** documents into clusters on their topic distributions.  
  6. **Visualizes** top terms per topic and a 2D PCA scatter in Streamlit.  

- **Why it exists:**  
  - Make large-scale philosophical text analysis accessible without heavy infra.  
  - Serve as a reusable, OO‐designed codebase for digital humanities/NLP teaching or research.  
  - Demonstrate best practices in scraping, modeling, persistence, and interactive viz.

---

## 1. Clone & Prepare Environment

```bash
git clone https://github.com/araujocode/philosophical_topic_modeling.git
cd philosophical_topic_modeling
````

### 1.1 Create & Activate Virtual Environment

- **Windows (PowerShell)**

  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **macOS/Linux**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

## 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **Note:** We pin `scikit-learn==1.4.2` because earlier versions lack wheels for Python 3.12 on Windows and trigger an MSVC build.

If you see:

```
Microsoft Visual C++ 14.0 or greater is required.
```

install the **Build Tools for Visual Studio** (include “C++ build tools”).

---

## 3. Install GNU make on Windows

If `make` isn’t recognized:

```powershell
choco install make
```

*(Requires Chocolatey: [https://chocolatey.org/install](https://chocolatey.org/install))*

Then you can run:

```bash
make scrape
make persist
make run
make test
```

---

## 4. Common Windows Errors & Fixes

### 4.1 scikit-learn Wheel Unavailable

- **Symptom:**

  ```
  ERROR: Could not find a version that satisfies the requirement scikit-learn==x.x.x
  ```

* **Fix:**

  ```bash
  pip install scikit-learn==1.4.2
  ```

### 4.2 pip Self-Upgrade Restriction

- **Symptom:**

  ```
  To modify pip, run: python.exe -m pip install --upgrade pip
  ```

* **Fix:**

  ```bash
  python -m pip install --upgrade pip
  ```

### 4.3 Virtualenv Activation Blocked

- **Symptom:** PowerShell refuses to run `Activate.ps1`.
- **Fix (run once as Admin):**

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 5. Populate the Database

```bash
make scrape
```

This will:

1. Crawl the SEP index at `https://plato.stanford.edu/entries/`
2. Download each entry, extract title & content via BeautifulSoup
3. Insert into `data/sep.db` (SQLite)

Verify:

```bash
sqlite3 data/sep.db "SELECT COUNT(*) FROM documents;"
```

---

## 6. Persist Processed Artifacts

```bash
make persist
```

This step:

- Fits the TF–IDF pipeline on your scraped articles.
- Fits both LDA and NMF topic models (with your default `N_TOPICS`).
- Saves the objects into `data/processed/` as Joblib files:

```
data/processed/
├── tfidf_pipeline.joblib
├── topics_lda.joblib
└── topics_nmf.joblib
```

On subsequent app launches, these are **loaded** instead of re-fitting, so the UI starts instantly.

---

## 7. Run the Streamlit App

```bash
make run
```

Open your browser to `http://localhost:8501`. In the sidebar you can:

- **Re-fit all models** (clears `data/processed/` and retrains)
- Choose **Topic model** (LDA or NMF)
- Adjust **Number of topics**
- Select **Clustering method** (KMeans or Agglomerative)
- Adjust **Number of clusters**

The main panel shows top‐terms per topic and a 2D PCA‐projected scatter of your documents colored by cluster.

---

## 8. Execute Tests

```bash
make test
```

Runs **pytest** over `tests/` to verify:

- Database insert & fetch
- TF–IDF pipeline and `FeatureExtractor`
- `TopicModeler` shapes & top‐terms
- Clustering consistency

---

## 9. Next Steps & Deployment

- **Deploy** to Streamlit Community Cloud via GitHub integration.
- **Enhancements** you might try:

  - Date-range filtering by `scrape_date`.
  - Full-text search with SQLite FTS.
  - Topic‐evolution timelines.

---

## References

- [scikit-learn Installation Guide](https://scikit-learn.org/stable/install.html)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [SQLite3 Python Docs](https://docs.python.org/3/library/sqlite3.html)
- [Streamlit API Reference](https://docs.streamlit.io/)
- [pytest Documentation](https://docs.pytest.org/)
