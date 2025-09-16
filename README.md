## Group Members
- Simon Greenaway
- Luke Miller  
- Ayddan Hartle
- Luke Meyer


---
## Class Structure
run -> Orchestrator.py  -> Installer.py
                        -> Tester.py
                        -> Url_Parser.py    

## Metrics System Implementation

```
src/app/metrics/
├── implementations/          # Individual metric implementations
│   ├── license.py           # License compatibility (LGPLv2.1)
│   ├── documentation.py     # Ramp-up time / docs quality  
│   ├── maintainer.py        # Bus factor / contributor health
│   ├── performance.py       # Performance claims evidence
│   ├── size.py              # Device compatibility scoring
│   ├── dataset.py           # Dataset quality metrics
│   └── code_quality.py      # Code maintainability analysis
├── base.py                  # Core data structures (ResourceBundle, MetricResult)
├── base_metric.py           # Abstract base class
├── registry.py              # Auto-discovery system (@register decorator)
└── engine.py                # Parallel orchestration + NDJSON output
```

### Metrics Implemented

| Metric | Purpose | Scoring Focus |
|--------|---------|---------------|
| `license` | LGPLv2.1 compatibility analysis | License clarity, commercial use permissions |
| `ramp_up_time` | Documentation quality assessment | README completeness, examples, tutorials |
| `bus_factor` | Maintainer health evaluation | Contributor diversity, project sustainability |
| `performance_claims` | Benchmark evidence validation | Performance metrics credibility |
| `size_score` | Device compatibility analysis | Model size vs deployment targets |
| `dataset_and_code_score` | Ecosystem completeness | Linked datasets and code repositories |
| `dataset_quality` | Dataset reliability metrics | Data quality, popularity, maintenance |
| `code_quality` | Repository maintainability | Code style, testing, CI/CD practices |

### Test Coverage for test_metrics.py
- All 8 metrics properly registered
- Score validation (0-1 range)
- Timing measurements
- NDJSON structure compliance
- NetScore calculation accuracy
---