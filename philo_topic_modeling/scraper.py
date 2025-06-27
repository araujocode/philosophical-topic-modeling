import time
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup

from philo_topic_modeling.config import SEP_INDEX, REQUEST_DELAY, DB_PATH
from philo_topic_modeling.db import DatabaseManager


class SEPScraper:
    """
    Scrapes SEP’s Published page for all entry slugs, then fetches
    each entry and stores title, URL, content, and scrape timestamp.
    """

    def __init__(self, limit=None):
        self.db = DatabaseManager(DB_PATH)
        self.limit = limit

    def fetch_index(self):
        """
        Download SEP’s published.html and extract unique slugs from any href
        containing '/entries/' (relative or absolute).
        """
        print(f"[fetch_index] GET {SEP_INDEX}")
        resp = requests.get(SEP_INDEX)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Find all <a> tags whose href contains '/entries/'
        anchors = soup.find_all(
            "a", href=re.compile(r"/entries/")
        )  # :contentReference[oaicite:1]{index=1}
        print(f"[fetch_index] Total <a> tags with '/entries/': {len(anchors)}")

        slugs = set()
        sample = []
        for a in anchors:
            href = a["href"]
            # Normalize: strip parameters, trailing slash, and domain
            # e.g. "https://plato.stanford.edu/entries/abduction/" or "/entries/abduction/"
            match = re.search(r"/entries/([^/]+)", href)
            if match:
                slug = match.group(1)
                if slug not in slugs and len(sample) < 10:
                    sample.append(slug)
                slugs.add(slug)

        print(f"[fetch_index] Unique slugs extracted: {len(slugs)}")
        print(f"[fetch_index] Sample slugs: {sample}")
        return list(slugs)

    def fetch_entry(self, slug):
        """
        Fetches a SEP entry by slug, extracts its <h1> title and
        all <p> paragraphs under <div id="main-text"> as content.
        """
        url = f"https://plato.stanford.edu/entries/{slug}/"
        print(f"[fetch_entry] GET {url}")
        resp = requests.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Title from the <h1>
        h1 = soup.find("h1")
        if not h1:
            raise RuntimeError(f"No <h1> found for slug '{slug}'")
        title = h1.get_text(strip=True)

        # 2. Main article text lives in <div id="main-text">
        main_div = soup.find("div", id="main-text") 
        if not main_div:
            raise RuntimeError(f"No <div id='main-text'> for slug '{slug}'")

        # 3. Extract all <p> children of main_div
        paras = [p.get_text(strip=True) for p in main_div.find_all("p")]
        if not paras:
            print(f"[fetch_entry] Warning: no <p> tags found in main-text for '{slug}'")

        # 4. Join paragraphs into one content string
        content = "\n\n".join(paras)

        return title, url, content

    def run(self):
        """
        Orchestrate index scraping, entry fetching, and DB insertion.
        """
        slugs = self.fetch_index()
        total = len(slugs)
        print(
            f"[run] Scraping {total} entries{' (limit='+str(self.limit)+')' if self.limit else ''}"
        )

        for i, slug in enumerate(slugs):
            if self.limit is not None and i >= self.limit:
                break
            try:
                title, url, content = self.fetch_entry(slug)
                self.db.insert_document(
                    title, url, content, datetime.utcnow().isoformat()
                )
                print(f"[{i+1}/{total}] Inserted: {title}")
            except Exception as e:
                print(f"[{i+1}/{total}] ERROR on '{slug}': {e}")
            time.sleep(REQUEST_DELAY)

        self.db.close()
        print("[run] Scraping complete.")


if __name__ == "__main__":
    SEPScraper().run()
