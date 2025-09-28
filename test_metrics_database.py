# tests/test_metrics_database.py
import sqlite3
from datetime import datetime, timedelta

def mkdb(tmp_path):
    from src.app.database.database import MetricsDatabase
    return MetricsDatabase(str(tmp_path / "metrics.sqlite"))

def test_cache_miss_and_hit(tmp_path):
    db = mkdb(tmp_path)

    # miss
    assert db.get_cached_result("user/model", [], []) is None

    # hit after write
    db.cache_result("user/model", '{"acc": 0.9}', ["ds/z"], ["https://github.com/acme/x"])
    assert db.get_cached_result("user/model", ["ds/z"], ["https://github.com/acme/x"]) == '{"acc": 0.9}'

def test_overwrite_and_sorting_key(tmp_path):
    db = mkdb(tmp_path)

    # write with unsorted lists
    db.cache_result("u/m", '{"v":1}', ["b","a"], ["c","b","a"])
    # same key even if order differs
    v = db.get_cached_result("u/m", ["a","b"], ["a","b","c"])
    assert v == '{"v":1}'

    # overwrite same key
    db.cache_result("u/m", '{"v":2}', ["a","b"], ["a","b","c"])
    assert db.get_cached_result("u/m", ["b","a"], ["c","b","a"]) == '{"v":2}'

def test_clear_cache_all_and_days(tmp_path):
    db = mkdb(tmp_path)

    # two rows
    db.cache_result("m/old", '{"x":1}')
    db.cache_result("m/new", '{"x":2}')

    # backdate one row's created_at by 10 days
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "UPDATE evaluation_cache SET created_at = datetime('now', '-10 days') WHERE model_url = ?",
            ("m/old",),
        )

    # clear only entries older than 7 days -> removes "m/old", keeps "m/new"
    db.clear_cache(older_than_days=7)
    assert db.get_cached_result("m/old") is None
    assert db.get_cached_result("m/new") == '{"x":2}'

    # full clear
    db.clear_cache()
    assert db.get_cached_result("m/new") is None

def test_get_cache_stats(tmp_path):
    db = mkdb(tmp_path)

    # add entries spread across (backdated) days
    db.cache_result("m/a", '{"a":1}')
    db.cache_result("m/a", '{"a":2}', ["ds1"])
    db.cache_result("m/b", '{"b":1}')

    # backdate 1 entry to yesterday so GROUP BY date(created_at) has at least 2 dates
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "UPDATE evaluation_cache SET created_at = datetime('now', '-1 day') WHERE model_url = ?",
            ("m/b",),
        )

    stats = db.get_cache_stats()
    assert stats["total_entries"] >= 3
    assert stats["database_path"].endswith(".sqlite")
    # top_models returns list of (model_url, count)
    assert any(row[0] == "m/a" and row[1] >= 2 for row in stats["top_models"])
    # recent_activity returns (date, count) tuples
    assert len(stats["recent_activity"]) >= 1

def test_get_database_singleton(monkeypatch, tmp_path):
    # Exercise get_database() path without writing to project root DB
    import src.app.database.database as dbmod

    made = {}
    real_cls = dbmod.MetricsDatabase

    class Stub(real_cls):
        def __init__(self):
            super().__init__(str(tmp_path / "singleton.sqlite"))
            made["constructed"] = made.get("constructed", 0) + 1

    monkeypatch.setattr(dbmod, "MetricsDatabase", Stub)

    # first call constructs
    inst1 = dbmod.get_database()
    # second call returns same instance
    inst2 = dbmod.get_database()
    assert inst1 is inst2
    assert made["constructed"] == 1