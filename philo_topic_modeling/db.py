import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            url TEXT UNIQUE,
            content TEXT,
            scrape_date TEXT
        );
        """
        )
        self.conn.commit()

    def insert_document(self, title, url, content, scrape_date):
        self.conn.execute(
            "INSERT OR IGNORE INTO documents (title,url,content,scrape_date) VALUES (?,?,?,?)",
            (title, url, content, scrape_date),
        )
        self.conn.commit()

    def fetch_all(self):
        cursor = self.conn.execute("SELECT id,title,content FROM documents")
        return cursor.fetchall()

    def close(self):
        self.conn.close()
