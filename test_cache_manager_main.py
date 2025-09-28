# tests/test_cache_manager_main.py
import sys
import pytest

def fake_db():
    class FakeDB:
        def __init__(self):
            self.clears = []
        def get_cache_stats(self):
            return {
                "database_path": "/tmp/db.sqlite",
                "total_entries": 1,
                "top_models": [("u/m", 1)],
                "recent_activity": [("2025-09-27", 1)],
            }
        def clear_cache(self, older_than_days=None):
            self.clears.append(older_than_days)
    return FakeDB()

def test_main_stats(monkeypatch, capsys):
    from src.app.database import cache_manager
    db = fake_db()
    monkeypatch.setattr(cache_manager, "get_database", lambda: db)
    monkeypatch.setattr(sys, "argv", ["cache_manager.py", "stats"])
    assert cache_manager.main() == 0
    out = capsys.readouterr().out
    assert "METRICS CACHE STATISTICS" in out

def test_main_clear_all(monkeypatch, capsys):
    from src.app.database import cache_manager
    db = fake_db()
    monkeypatch.setattr(cache_manager, "get_database", lambda: db)
    monkeypatch.setattr(sys, "argv", ["cache_manager.py", "clear"])
    assert cache_manager.main() == 0
    assert db.clears == [None]
    out = capsys.readouterr().out
    assert "Cache cleared" in out

def test_main_clear_days(monkeypatch, capsys):
    from src.app.database import cache_manager
    db = fake_db()
    monkeypatch.setattr(cache_manager, "get_database", lambda: db)
    monkeypatch.setattr(sys, "argv", ["cache_manager.py", "clear", "7"])
    assert cache_manager.main() == 0
    assert db.clears == [7]
    out = capsys.readouterr().out
    assert "older than 7 days" in out

def test_main_bad_days(monkeypatch, capsys):
    from src.app.database import cache_manager
    db = fake_db()
    monkeypatch.setattr(cache_manager, "get_database", lambda: db)
    monkeypatch.setattr(sys, "argv", ["cache_manager.py", "clear", "oops"])
    assert cache_manager.main() == 1
    out = capsys.readouterr().out
    assert "Days must be a number" in out

def test_main_unknown(monkeypatch, capsys):
    from src.app.database import cache_manager
    monkeypatch.setattr(sys, "argv", ["cache_manager.py", "wat"])
    assert cache_manager.main() == 1
    out = capsys.readouterr().out
    assert "Unknown command" in out

def test_main_no_args(monkeypatch, capsys):
    from src.app.database import cache_manager
    monkeypatch.setattr(sys, "argv", ["cache_manager.py"])
    assert cache_manager.main() == 1
    out = capsys.readouterr().out
    assert "Usage:" in out or "Cache" in out  # prints docstring