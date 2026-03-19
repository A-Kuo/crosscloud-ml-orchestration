# Skill: Testing Guide

## When to Use

Use this skill when writing new tests, running the test suite, or understanding the testing strategy for this project.

## Test Architecture

The test suite has two tiers:

### Fast Tests (default, no model download)
- Test pure math functions (`entropy_from_probs`, `entropy_from_logits`, `_compute_layer_head_entropies`)
- Test `ThresholdCalibrator` with synthetic data
- Test `InferenceRouter` with a **mock probe** (returns fixed H_route values)
- Run in ~40s, no network required

### Slow Tests (`--runslow`, downloads DistilBERT ~250MB)
- Test `AttentionEntropyProbe.compute()` end-to-end on real text
- AUROC benchmark: entropy should separate "easy" vs "hard" synthetic texts
- Marked with `@pytest.mark.slow`, skipped by default via `conftest.py`

## Running Tests

```bash
# Fast tests only (CI default)
pytest

# All tests including model tests
pytest --runslow

# Single test file
pytest tests/test_entropy.py -v

# Single test class
pytest tests/test_threshold.py::TestThresholdCalibrator -v

# With coverage
pytest --cov=router --cov-report=html
```

## Writing New Tests

### For Pure Functions (router/ modules)

Place in the appropriate `tests/test_*.py` file. Use synthetic data, not real model outputs:

```python
def test_my_new_function():
    result = my_function(input_data)
    assert result == expected
```

### For Router Behaviour

Use the `make_mock_probe` helper from `tests/test_router.py`:

```python
from tests.test_router import make_mock_probe
from router.router import InferenceRouter
from router.threshold import ThresholdCalibrator

def test_custom_routing_scenario():
    calibrator = ThresholdCalibrator()
    calibrator.update_tau(1.5)
    probe = make_mock_probe(h_route=1.8)
    router = InferenceRouter(probe=probe, calibrator=calibrator)
    decision = router.route("test input")
    assert decision.destination.value == "aws_sagemaker"
```

### For Model Integration (slow)

Mark with `@pytest.mark.slow` and use a class-scoped fixture to avoid reloading the model per test:

```python
import pytest
from router.entropy import AttentionEntropyProbe

@pytest.mark.slow
class TestMyModelFeature:
    @pytest.fixture(scope="class")
    def probe(self):
        return AttentionEntropyProbe()

    def test_something(self, probe):
        result = probe.compute("test text")
        assert result.h_route > 0
```

### For the Airflow DAG

The DAG uses `MOCK_CLOUD=true` by default. Test DAG task callables directly:

```python
from airflow.dags.crosscloud_orchestration import entropy_audit
# Create a mock context with XCom support
```

### For the FastAPI Server

Use `httpx.AsyncClient` with the ASGI transport:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from inference.server import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
```

## Key Test Properties to Preserve

1. **Entropy monotonicity:** Uniform attention must produce maximum entropy (`log(T)`).
2. **Threshold within range:** Calibrated tau must fall within the observed entropy range.
3. **Isotonic monotonicity:** `predict_error_prob` must be monotone non-decreasing.
4. **Routing boundary:** `H >= tau` always routes to AWS, `H < tau` always routes to GCP.
5. **Serialisation round-trip:** `CalibrationResult.save()` then `.load()` must preserve all fields.
6. **AUROC above chance:** On correlated synthetic data, isotonic AUROC must exceed 0.55.
