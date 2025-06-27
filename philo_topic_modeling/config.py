import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "sep.db")
SEP_INDEX = "https://plato.stanford.edu/published.html"

# Scraping settings
REQUEST_DELAY = 1  # seconds

# Modeling settings
N_TOPICS = 10
N_CLUSTERS = 5
