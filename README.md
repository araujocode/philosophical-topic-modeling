## Summary

This guide shows how to clone the repo, create and activate a Python virtual environment, install dependencies (including resolving scikit-learn build errors on Windows), install GNU make for Windows, run the scraper, launch the Streamlit app, and execute tests. It also covers troubleshooting for common Windows-specific problems (compiler errors, missing wheels, activation commands).

---

## 1. Clone & Prepare Environment

```bash
git clone https://github.com/you/philo_topic_modeling.git
cd philo_topic_modeling
```

### 1.1 Create & Activate Virtual Environment

* **Windows (PowerShell)**:

  ```powershell
  python -m venv venv           # create venv
  .\venv\Scripts\Activate.ps1  # activate
  ```

* **macOS/Linux**:

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

## 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel  # ensure latest installer tools
pip install -r requirements.txt
```

* **Note**: We pin `scikit-learn==1.4.2` because versions ≤ 1.2.2 lack wheels for Python 3.12 on Windows and trigger a source build requiring MSVC.
* If you see:

  ```
  Microsoft Visual C++ 14.0 or greater is required.
  ```

  install the **Build Tools for Visual Studio** (select “C++ build tools”) from Microsoft’s website.

---

## 3. Install GNU make on Windows

If you get `bash: make: command not found`, install `make`:

```powershell
choco install make
```

*(requires Chocolatey: [https://chocolatey.org/install](https://chocolatey.org/install))*

Once installed, you can use:

```bash
make scrape
make run
make test
```

---

## 4. Common Windows Errors & Fixes

### 4.1 scikit-learn Wheel Unavailable

* **Symptom**:

  ```
  ERROR: Could not find a version that satisfies the requirement scikit-learn==1.2.2
  ```

* **Cause**: No binary wheel for Python 3.12.
* **Solution**:

  ```bash
  pip install scikit-learn==1.4.2
  ```

### 4.2 pip Self-Upgrade Restriction

* **Symptom**:

  ```
  ERROR: To modify pip, please run: python.exe -m pip install --upgrade pip
  ```

* **Solution**:

  ```bash
  python -m pip install --upgrade pip
  ```

### 4.3 Virtualenv Activation Blocked

* **Symptom**: PowerShell refuses to run `Activate.ps1`.
* **Solution** (run as Administrator once):

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 5. Populate the Database

```bash
make scrape
```

What it does:

1. Crawls the SEP index (`https://plato.stanford.edu/entries/`)
2. Downloads each article, extracts title & content via BeautifulSoup
3. Inserts into `data/sep.db` using SQLite

**Verify**:

```bash
sqlite3 data/sep.db "SELECT COUNT(*) FROM documents;"
```

---

## 6. Run the Streamlit App

```bash
make run
```

Opens at `http://localhost:8501` showing:

* **Sidebar**: select topic model (LDA/NMF), number of topics, clustering method, number of clusters
* **Main**: top-words per topic, 2D PCA scatter of articles colored by cluster

---

## 7. Execute Tests

```bash
make test
```

Runs **pytest** over `tests/` to verify database operations, TF-IDF pipeline, and topic-modeler functionality.

---

## 8. Deployment & Next Steps

* **Deploy**: Push to GitHub and **Streamlit Community Cloud** (one-click integration).
* **Enhancements**:

  * Add a date-range slider filtering by `scrape_date`.
  * Switch to Altair for interactive visualizations.
  * Incorporate full-text search with SQLite FTS.

---

## References

1. Official scikit-learn install guide (wheels for Win) ([scikit-learn.org][1])
2. BeautifulSoup usage for scraping example ([pypi.org][8])
3. scikit-learn 1.4.2 release notes (supports Python 3.12) ([github.com][3])
4. sqlite3 and sqlite-utils reference for Python DB usage ([stackoverflow.com][9])
5. pip module upgrade method explanation ([scikit-learn.org][6])
6. Microsoft C++ Build Tools requirement for Windows ([scikit-learn.ru][4])
7. Streamlit UI documentation for plotting controls ([medium.com][10])
8. GitHub issue on scikit-learn wheels for Python versions ([github.com][5])
9. PowerShell execution policy guide for venv activation ([ogrisel.github.io][7])
10. pytest integration in Python projects

[1]: https://scikit-learn.org/stable/install.html?utm_source=chatgpt.com "Installing scikit-learn"
[3]: https://github.com/scikit-learn/scikit-learn/releases?utm_source=chatgpt.com "Releases · scikit-learn/scikit-learn - GitHub"
[4]: https://scikit-learn.ru/stable/developers/advanced_installation.html?utm_source=chatgpt.com "Installing the development version of scikit-learn"
[5]: https://github.com/scikit-learn/scikit-learn/issues/29973?utm_source=chatgpt.com "Cannot install sklearn >=1.5 on windows with python 3.13 #29973"
[6]: https://scikit-learn.org/stable/developers/advanced_installation.html?utm_source=chatgpt.com "Installing the development version of scikit-learn"
[7]: https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/install.html?utm_source=chatgpt.com "1. Installing scikit-learn - GitHub Pages"
[8]: https://pypi.org/project/scikit-learn/?utm_source=chatgpt.com "scikit-learn - PyPI"
[9]: https://stackoverflow.com/questions/59974146/installing-an-old-version-of-scikit-learn?utm_source=chatgpt.com "Installing an old version of scikit-learn - Stack Overflow"
[10]: https://medium.com/%406unpnp/install-scikit-learn-d58f1415962d?utm_source=chatgpt.com "Install scikit-learn. Hopefully a happy first step to machine… - Medium"
