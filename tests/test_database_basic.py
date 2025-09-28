# tests/test_cache_manager_cli.py
import sys
import types
import pytest

def test_show_stats(monkeypatch, capsys):
    from src.app.database import cache_manager

    # Fake database object returned by get_database
    class FakeDB:
        def get_cache_stats(self):
            return {
                "database_path": "/tmp/fake.sqlite",
                "total_entries": 2,
                "top_models": [("user/model1", 5)],
                "recent_activity": [("2025-09-28", 3)],
            }

    monkeypatch.setattr(cache_manager, "get_database", lambda: FakeDB())

    cache_manager.show_stats()
    out = capsys.readouterr().out
    assert "METRICS CACHE STATISTICS" in out
    assert "model1" in out

def test_clear_cache_all(monkeypatch, capsys):
    from src.app.database import cache_manager

    called = {}
    class FakeDB:
        def clear_cache(self, older_than_days=None):
            called["days"] = older_than_days

    monkeypatch.setattr(cache_manager, "get_database", lambda: FakeDB())
    cache_manager.clear_cache()
    out = capsys.readouterr().out
    assert "Cache cleared" in out
    assert called["days"] is None

def test_clear_cache_days(monkeypatch, capsys):
    from src.app.database import cache_manager

    called = {}
    class FakeDB:
        def clear_cache(self, older_than_days=None):
            called["days"] = older_than_days

    monkeypatch.setattr(cache_manager, "get_database", lambda: FakeDB())
    cache_manager.clear_cache(7)
    out = capsys.readouterr().out
    assert "older than 7 days" in out
    assert called["days"] == 7