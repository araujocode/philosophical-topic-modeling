import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the SQLite database lives
DB_PATH = os.path.join(BASE_DIR, "..", "data", "sep.db")

# Where we store processed artifacts (tf-idf pipelines, models, etc.)
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# SEP index URL
SEP_INDEX = "https://plato.stanford.edu/published.html"

# Scraping settings
REQUEST_DELAY = 1  # seconds

# Modeling settings
N_TOPICS = 10
N_CLUSTERS = 5
