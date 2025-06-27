import sqlite3
from pathlib import Path


class DatabaseManager:
    """
    Thin wrapper around sqlite3 that now also stores an optional
    top-level SEP *category* (logic, ethics, metaphysics, …).
    """

    def __init__(self, db_path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()
        self._ensure_category_column()

    # --------------------------------------------------------------------- #
    # Schema utilities
    # --------------------------------------------------------------------- #
    def _create_table(self):
        """Create the documents table on a brand-new DB."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                title       TEXT,
                url         TEXT UNIQUE,
                content     TEXT,
                category    TEXT,
                scrape_date TEXT
            );
            """
        )
        self.conn.commit()

    def _ensure_category_column(self):
        """
        If the DB was created prior to this upgrade it will lack the new
        “category” column.  Add it on the fly so old databases still work.
        """
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(documents)")}
        if "category" not in cols:
            self.conn.execute("ALTER TABLE documents ADD COLUMN category TEXT;")
            self.conn.commit()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def insert_document(self, title, url, content, category, scrape_date):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO documents
            (title, url, content, category, scrape_date)
            VALUES (?,?,?,?,?)
            """,
            (title, url, content, category, scrape_date),
        )
        self.conn.commit()

    def fetch_all(self):
        """
        Returns a list of tuples:
            (id, title, content, category)
        """
        cur = self.conn.execute("SELECT id, title, content, category FROM documents")
        return cur.fetchall()

    def close(self):
        self.conn.close()
