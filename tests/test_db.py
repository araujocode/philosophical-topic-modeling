import os
import tempfile
from philo_topic_modeling.db import DatabaseManager


def test_insert_and_fetch():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db = DatabaseManager(path)
    db.insert_document("T", "u", "c", "d", "2025-01-01T00:00:00")
    rows = db.fetch_all()
    assert len(rows) == 1
    db.close()
    os.remove(path)
