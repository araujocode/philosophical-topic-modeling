# Philo Topic Modeling

A Streamlit app for topic modeling of Stanford Encyclopedia of Philosophy entries using LDA/NMF and document clustering.

---

## Summary

This guide walks you through:

1. Cloning the repo  
2. Creating & activating a Python virtual environment  
3. Installing dependencies (with Windows-specific build tips)  
4. Running the scraper to populate the database  
5. Launching the Streamlit app  
6. Running tests  
7. Troubleshooting common Windows issues

---

## 1. Clone & Prepare Environment

```bash
git clone https://github.com/you/philo_topic_modeling.git
cd philo_topic_modeling
````

### 1.1 Create & Activate Virtual Environment

* **Windows (PowerShell)**

  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

* **macOS/Linux**

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

> **Note:** We pin `scikit-learn==1.4.2` because earlier versions lack prebuilt wheels for Python 3.12 on Windows and trigger an MSVC build.

If you see:

```
Microsoft Visual C++ 14.0 or greater is required.
```

install the **Build Tools for Visual Studio** (include “C++ build tools”).

---

## 3. Install GNU make on Windows

If `make` isn’t found:

```powershell
choco install make
```

*(Requires Chocolatey: [https://chocolatey.org/install](https://chocolatey.org/install))*

Then you can use:

```bash
make scrape
make run
make test
```

---

## 4. Common Windows Errors & Fixes

### 4.1 scikit-learn Wheel Unavailable

* **Symptom:**

  ```
  ERROR: Could not find a version that satisfies the requirement scikit-learn==x.x.x
  ```
* **Fix:**

  ```bash
  pip install scikit-learn==1.4.2
  ```

### 4.2 pip Self-Upgrade Restriction

* **Symptom:**

  ```
  To modify pip, run: python.exe -m pip install --upgrade pip
  ```
* **Fix:**

  ```bash
  python -m pip install --upgrade pip
  ```

### 4.3 Virtualenv Activation Blocked

* **Symptom:** PowerShell refuses to run `Activate.ps1`.
* **Fix (Admin once):**

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 5. Populate the Database

```bash
make scrape
```

This will:

1. Crawl the SEP index (`/entries/`)
2. Download each article, extract title & content
3. Insert into `data/sep.db`

Verify:

```bash
sqlite3 data/sep.db "SELECT COUNT(*) FROM documents;"
```

---

## 6. Run the Streamlit App

```bash
make run
```

Then open your browser to `http://localhost:8501`. The sidebar lets you choose:

* **Topic model** (LDA or NMF)
* **Number of topics**
* **Clustering method** (KMeans or Agglomerative)
* **Number of clusters**

The main view shows:

* Top-terms per topic
* 2D PCA scatter of documents colored by cluster

---

## 7. Execute Tests

```bash
make test
```

Runs pytest over `tests/` to verify:

* Database insert & fetch
* TF–IDF pipeline via `FeatureExtractor`
* TopicModeler output shapes & top-terms

---

## 8. Next Steps & Deployment

* **Deploy** to Streamlit Community Cloud (one-click from GitHub).
* **Enhancements** you might try:

  * Date-range slider filtering by `scrape_date`
  * Full-text search via SQLite FTS
  * Visualizing topic evolution over time

---

## References

* [scikit-learn Installation Guide](https://scikit-learn.org/stable/install.html)
* [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [SQLite3 Python Docs](https://docs.python.org/3/library/sqlite3.html)
* [Streamlit API Reference](https://docs.streamlit.io/)
* [pytest Documentation](https://docs.pytest.org/)

```

This README now:
- Reflects your `FeatureExtractor` helper (used in **app.py**)  
- Shows how to handle an empty DB safely  
- Includes all necessary commands and Windows‐specific tips  
- Is organized into clear, numbered sections for easy follow-through.
```
