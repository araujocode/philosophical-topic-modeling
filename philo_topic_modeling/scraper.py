"""
philo_topic_modeling/scraper.py

Scrapes the SEP “published” index, then downloads each entry and
stores title, URL, full text **and** a simple category label derived
from <div id="related-entries">.

Example snippet on an entry page:

<div id="related-entries">
  <p>
    <a href="../epistemology-bayesian/">epistemology: Bayesian</a> |
    ...
  </p>
</div>

We take the first real <a> inside that div, strip anything after the
colon, lowercase it, and call that the article’s category
(e.g. “epistemology”).
"""

import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from philo_topic_modeling.config import SEP_INDEX, REQUEST_DELAY, DB_PATH
from philo_topic_modeling.db import DatabaseManager


class SEPScraper:
    """
    Orchestrates: index scraping → entry scraping → SQLite insert
    """

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(self, limit=None):
        self.db = DatabaseManager(DB_PATH)
        self.limit = limit

    # ------------------------------------------------------------------ #
    # Index page
    # ------------------------------------------------------------------ #
    def fetch_index(self):
        """
        Download SEP’s ‘published.html’ and extract unique slugs from any
        href that contains '/entries/' (absolute or relative).
        """
        print(f"[fetch_index] GET {SEP_INDEX}")
        resp = requests.get(SEP_INDEX, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.find_all("a", href=re.compile(r"/entries/"))
        print(f"[fetch_index] Total <a> with '/entries/': {len(anchors)}")

        slugs, sample = set(), []
        for a in anchors:
            match = re.search(r"/entries/([^/]+)", a["href"])
            if match:
                slug = match.group(1)
                if slug not in slugs and len(sample) < 10:
                    sample.append(slug)
                slugs.add(slug)

        print(f"[fetch_index] Unique slugs extracted: {len(slugs)}")
        print(f"[fetch_index] Sample slugs: {sample}")
        return list(slugs)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _extract_category(self, soup: BeautifulSoup):
        """
        Return a lowercase category string taken from the first *real*
        article link in <div id="related-entries">, or None if not found.

        A dummy anchor named “Related Entries” often appears first;
        we skip that.
        """
        rel_div = soup.find("div", id="related-entries")
        if not rel_div:
            return None

        for a in rel_div.find_all("a"):
            text = a.get_text(strip=True)
            if not text:
                continue
            # Skip the self-referential “Related Entries” link
            if text.lower().startswith("related entries"):
                continue
            # Keep only the part before any colon, e.g. "ethics: applied"
            return text.split(":")[0].strip().lower()

        return None

    # ------------------------------------------------------------------ #
    # Entry page
    # ------------------------------------------------------------------ #
    def fetch_entry(self, slug):
        """
        Pull <h1> title, join all <p> in <div id="main-text"> as content,
        and grab a category via _extract_category().
        """
        url = f"https://plato.stanford.edu/entries/{slug}/"
        print(f"[fetch_entry] GET {url}")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Title
        h1 = soup.find("h1")
        if not h1:
            raise RuntimeError(f"No <h1> found for slug '{slug}'")
        title = h1.get_text(strip=True)

        # 2. Main text
        main_div = soup.find("div", id="main-text")
        if not main_div:
            raise RuntimeError(f"No <div id='main-text'> for slug '{slug}'")

        paras = [p.get_text(strip=True) for p in main_div.find_all("p")]
        if not paras:
            print(f"[fetch_entry] Warning: no <p> tags under main-text for '{slug}'")

        content = "\n\n".join(paras)

        # 3. Category via related entries
        category = self._extract_category(soup)

        return title, url, content, category

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def run(self):
        slugs = self.fetch_index()
        total = len(slugs)
        print(
            f"[run] Scraping {total} entries"
            + (f" (limit = {self.limit})" if self.limit else "")
        )

        for i, slug in enumerate(slugs):
            if self.limit is not None and i >= self.limit:
                break
            try:
                title, url, content, category = self.fetch_entry(slug)
                self.db.insert_document(
                    title,
                    url,
                    content,
                    category,
                    datetime.utcnow().isoformat(),
                )
                print(
                    f"[{i+1}/{total}] Inserted: {title[:60]:<60} | "
                    f"category = {category or 'n/a'}"
                )
            except Exception as e:
                print(f"[{i+1}/{total}] ERROR on '{slug}': {e}")
            time.sleep(REQUEST_DELAY)

        self.db.close()
        print("[run] Scraping complete.")


if __name__ == "__main__":
    SEPScraper().run()
