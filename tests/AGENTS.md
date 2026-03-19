# tests/ — Agent Context

## Purpose

Pytest test suite. Two tiers: fast (mocked, ~40s) and slow (model-loading, `--runslow`).

## Test Files

| File | Tests | Tier |
|------|-------|------|
| `test_entropy.py` | `entropy_from_probs`, `entropy_from_logits`, `_compute_layer_head_entropies` (synthetic tensors) + `AttentionEntropyProbe` (real DistilBERT) | Fast + Slow |
| `test_threshold.py` | `ThresholdCalibrator.fit`, monotonicity, AUROC, save/load, edge cases | Fast |
| `test_router.py` | `InferenceRouter.route` with mock probes — routing direction, boundary, metadata, serialisation | Fast |

## Configuration

- `conftest.py` — adds `--runslow` CLI flag; skips `@pytest.mark.slow` tests by default.
- `pytest.ini` — sets `testpaths = tests`, `addopts = -v --tb=short`.

## Mock Probe Pattern

Router tests avoid downloading models by using `make_mock_probe(h_route)`:

```python
def make_mock_probe(h_route: float) -> MagicMock:
    probe = MagicMock()
    probe.compute.return_value = EntropyResult(
        h_route=h_route,
        per_head_entropies=np.array([[h_route]]),
        input_tokens=12,
        model_name="mock-probe",
    )
    return probe
```

Use this pattern for any new test that needs routing decisions without model inference.

## Invariants to Always Test

1. Uniform attention → entropy = `log(T)` (maximum).
2. Deterministic attention → entropy near 0.
3. `H >= tau` → AWS SageMaker. `H < tau` → GCP Cloud Run.
4. Isotonic predictions are monotone non-decreasing.
5. `CalibrationResult` JSON round-trip preserves all fields.
6. AUROC > 0.55 on correlated synthetic data.

## When Adding Tests

- Fast tests go in existing `test_*.py` files under the appropriate class.
- New slow tests must be decorated `@pytest.mark.slow` and use `scope="class"` fixtures for model loading.
- Server endpoint tests should use `httpx.AsyncClient` with `ASGITransport`.
