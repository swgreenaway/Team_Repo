# import pytest
# from pathlib import Path

# # ---------- BaseMetric + stable_01 ----------
# def test_stable_01_deterministic():
#     from src.app.metrics.base_metric import stable_01
#     a = stable_01("abc")
#     b = stable_01("abc")
#     c = stable_01("xyz")
#     assert a == b and 0 < a <= 1
#     assert a != c

# def test_base_metric_error_path(monkeypatch):
#     from src.app.metrics.base_metric import BaseMetric
#     from src.app.metrics.base import ResourceBundle

#     class BadMetric(BaseMetric):
#         name = "bad"
#         def _compute_score(self, resource): raise RuntimeError("fail")
#     m = BadMetric()
#     rb = ResourceBundle(model_url="m", dataset_urls=[], code_urls=[])
#     res = m.compute(rb)
#     assert "placeholder" in res.notes.lower()
#     assert 0 <= res.score <= 1

# # ---------- BusFactor ----------
# def test_bus_factor_scoring(monkeypatch):
#     from src.app.metrics.bus_factor import BusFactorMetric
#     from src.app.metrics.base import ResourceBundle
#     data = {
#         "author": "a",
#         "organization": "org",
#         "siblings": [{"rfilename": "https://github.com/u/r.git"}],
#         "cardData": {}
#     }
#     # fake responses
#     class DummyResp:
#         def __init__(self, js): self._js = js
#         def json(self): return self._js
#         def raise_for_status(self): pass
#     def fake_get(url, **k):
#         if "contributors" in url: return DummyResp([{"login": "a"}])
#         if "commits" in url: return DummyResp([{"commit":{"author":{"date":"2025-09-01T00:00:00Z"}}}])
#         return DummyResp(data)
#     import requests; monkeypatch.setattr(requests, "get", fake_get)
#     m = BusFactorMetric()
#     rb = ResourceBundle("url", [], [], model_id="m")
#     res = m.compute(rb)
#     assert 0 <= res.score <= 1

# # ---------- CodeQuality ----------
# def test_code_quality_local_repo(tmp_path, monkeypatch):
#     from src.app.metrics.code_quality import CodeQualityMetric
#     from src.app.metrics.base import ResourceBundle
#     repo = tmp_path / "repo"; repo.mkdir()
#     (repo / "README.md").write_text("hi")
#     (repo / "LICENSE").write_text("x")
#     (repo / "test_a.py").write_text("def f(x:int)->int:\n    return x\n")
#     (repo / "main.py").write_text("print(1)")
#     # monkeypatch git clone to just mkdir
#     import subprocess
#     monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"returncode":0,"stdout":"","stderr":""})())
#     monkeypatch.setattr(CodeQualityMetric, "_analyze_repository", lambda self,u:0.8)
#     m = CodeQualityMetric()
#     rb = ResourceBundle("m", [], ["https://github.com/u/r"])
#     res = m.compute(rb)
#     assert 0 <= res.score <= 1
#     assert "Analyzed" in m._get_computation_notes(rb)

# # ---------- DatasetAndCode ----------
# def test_dataset_and_code(monkeypatch):
#     from src.app.metrics.dataset_and_code import DatasetAndCodeScoreMetric
#     from src.app.metrics.base import ResourceBundle
#     data = {"cardData":{"README":"this dataset uses code at github.com"}}
#     class DummyResp: 
#         def json(self): return data
#         def raise_for_status(self): pass
#     import requests; monkeypatch.setattr(requests,"get",lambda *a,**k:DummyResp())
#     m = DatasetAndCodeScoreMetric()
#     rb = ResourceBundle("m", ["ds"], ["c"], model_id="m")
#     res = m.compute(rb)
#     assert res.score > 0.2
#     assert "dataset and code" in m._get_computation_notes(rb).lower()

# # ---------- DatasetQuality ----------
# def test_dataset_quality_stub(monkeypatch):
#     from src.app.metrics.dataset_quality import DatasetQualityMetric
#     from src.app.metrics.base import ResourceBundle
#     m = DatasetQualityMetric()
#     # stub hf_client methods
#     class FakeHF:
#         def get_dataset_info(self, ds): return {"dataset_info": {}}
#         def get_dataset_card_data(self, ds): return {"downloads":1000,"likes":50,"task_categories":["a"],"language":["en"],"size_categories":["100K<n<1M"]}
#         def get_dataset_readme(self, ds): return "dataset description usage citation license"
#     monkeypatch.setattr("src.app.metrics.dataset_quality.hf_client", FakeHF())
#     rb = ResourceBundle("m", ["https://huggingface.co/datasets/x"], [])
#     res = m.compute(rb)
#     assert 0 <= res.score <= 1
#     notes = m._get_computation_notes(rb)
#     assert "analyzed" in notes.lower()

# # ---------- License ----------
# def test_license_metric(monkeypatch):
#     import src.app.metrics.license_metric as lm
#     from src.app.metrics.base import ResourceBundle

#     monkeypatch.setattr(lm, "HF_API_MODEL", "https://example.com/{repo_id}")

#     class DummyResp:
#         def json(self): return {"cardData": {"license": "apache-2.0", "README": ""}}
#         def raise_for_status(self): pass

#     monkeypatch.setattr(lm, "requests", type("R", (), {"get": lambda *a, **k: DummyResp()}))

#     res = lm.LicenseMetric().compute(ResourceBundle("m", [], [], model_id="m"))
#     assert res.score == 0.7  # apache-2.0 → 0.7


# def test_perf_claims(monkeypatch):
#     import src.app.metrics.performance_claims as pm
#     from src.app.metrics.base import ResourceBundle

#     monkeypatch.setattr(pm, "HF_API_MODEL", "https://example.com/{repo_id}")

#     class DummyResp:
#         def json(self): return {"cardData": {"README": "accuracy | f1 arxiv.org"}}
#         def raise_for_status(self): pass

#     monkeypatch.setattr(pm, "requests", type("R", (), {"get": lambda *a, **k: DummyResp()}))
#     res = pm.PerformanceClaimsMetric().compute(ResourceBundle("m", [], [], model_id="m"))
#     assert res.score >= 0.4


# def test_ramp_up_time(monkeypatch):
#     import src.app.metrics.ramp_up_time as rm
#     from src.app.metrics.base import ResourceBundle

#     monkeypatch.setattr(rm, "HF_API_MODEL", "https://example.com/{repo_id}")

#     class DummyResp:
#         def json(self): return {"cardData": {"README": "usage ``` example"}}
#         def raise_for_status(self): pass

#     monkeypatch.setattr(rm, "requests", type("R", (), {"get": lambda *a, **k: DummyResp()}))
#     res = rm.RampUpTimeMetric().compute(ResourceBundle("m", [], [], model_id="m"))
#     assert res.score >= 0.6
# # ---------- Engine ----------
# def test_engine_assemble_and_run(monkeypatch):
#     from src.app.metrics.engine import assemble_ndjson_row, compute_net_score, run_bundle
#     from src.app.metrics.base import MetricResult, ResourceBundle, Metric

#     rb = ResourceBundle("https://huggingface.co/u/m", [], [])
#     results = {
#         "ramp_up_time": MetricResult(0.6,1),
#         "bus_factor": MetricResult(0.5,2),
#         "performance_claims": MetricResult(0.3,3),
#         "license": MetricResult(0.7,4),
#         "size_score": MetricResult(0.8,5),
#         "dataset_and_code_score": MetricResult(0.4,6),
#         "dataset_quality": MetricResult(0.9,7),
#         "code_quality": MetricResult(0.5,8),
#     }
#     net = compute_net_score(results,{k:1 for k in results})
#     row = assemble_ndjson_row(rb, results, net, 123)
#     assert '"net_score":' in row
#     # run_bundle with fake metric
#     class DummyMetric:
#         name="x"
#         def compute(self, rb): return MetricResult(0.5,1,"ok")
#     out = run_bundle(rb, [DummyMetric()], {"x":1})
#     assert "net_score" in out

# # ---------- registry ----------
# def test_registry_register_and_all():
#     from src.app.metrics import registry

#     @registry.register("dummy_metric")
#     class Dummy:
#         name = "dummy_metric"
#         def compute(self, rb): return "ok"

#     all_m = registry.all_metrics()
#     assert "dummy_metric" in all_m
#     inst = all_m["dummy_metric"]()  # call factory to get instance
#     assert isinstance(inst, Dummy)

# # ---------- size.py helpers ----------
# def test_is_weight_file_and_extract_id():
#     from src.app.metrics.size import _is_weight_file, _extract_model_id_from_url
#     assert _is_weight_file("model.safetensors")
#     assert not _is_weight_file("readme.md")
#     assert _extract_model_id_from_url("https://huggingface.co/user/model") == "user/model"
#     assert _extract_model_id_from_url("https://huggingface.co/models/bert-base-uncased") == "bert-base-uncased"
#     assert _extract_model_id_from_url("nonsense") is None

# # ---------- SizeScoreMetric ----------
# def test_size_score_metric_with_stub(monkeypatch):
#     from src.app.metrics.size import SizeScoreMetric
#     from src.app.metrics.base import ResourceBundle

#     # stub hf_client available
#     monkeypatch.setattr("src.app.metrics.size._HF_CLIENT_AVAILABLE", True)
#     # stub fetch size
#     monkeypatch.setattr(SizeScoreMetric, "_fetch_model_size", lambda self,u: 150.0)

#     rb = ResourceBundle("https://huggingface.co/user/model", [], [])
#     m = SizeScoreMetric()
#     res = m.compute(rb)
#     assert 0 <= res.score <= 1
#     notes = m._get_computation_notes(rb)
#     assert "150.0MB" in notes

# def test_size_score_metric_no_client(monkeypatch):
#     from src.app.metrics.size import SizeScoreMetric
#     from src.app.metrics.base import ResourceBundle
#     monkeypatch.setattr("src.app.metrics.size._HF_CLIENT_AVAILABLE", False)
#     m = SizeScoreMetric()
#     rb = ResourceBundle("u/m", [], [])
#     assert m.compute(rb) == 0.0
#     assert "not available" in m._get_computation_notes(rb).lower()

# def test_size_score_metric_fetch_none(monkeypatch):
#     from src.app.metrics.size import SizeScoreMetric
#     from src.app.metrics.base import ResourceBundle
#     monkeypatch.setattr("src.app.metrics.size._HF_CLIENT_AVAILABLE", True)
#     monkeypatch.setattr(SizeScoreMetric, "_fetch_model_size", lambda self,u: None)
#     rb = ResourceBundle("m", [], [])
#     m = SizeScoreMetric()
#     assert m.compute(rb) == 0.0
#     assert "could not fetch" in m._get_computation_notes(rb).lower()

# def test_size_score_device_scores_boundaries():
#     from src.app.metrics.size import SizeScoreMetric
#     m = SizeScoreMetric()
#     # very small
#     s = m._compute_device_scores(10)
#     assert all(v == 1.0 for v in s.values())
#     # very large
#     s2 = m._compute_device_scores(999999)
#     assert all(v == 0.0 for v in s2.values())