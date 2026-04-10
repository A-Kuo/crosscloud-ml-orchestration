# router/ — Agent Context

## Purpose

Core routing logic. Three modules with a clean dependency chain:

```
entropy.py (standalone) ← threshold.py (standalone) ← router.py (combines both)
```

## Module Responsibilities

### entropy.py
- `AttentionEntropyProbe`: loads a HuggingFace transformer, runs forward pass with `output_attentions=True`, computes H_route.
- Pure helpers: `entropy_from_probs`, `entropy_from_logits`, `_compute_layer_head_entropies` — usable without a model instance.
- All entropy computations clip to `[1e-12, 1.0]` before `log()`. Do not remove this floor.

### threshold.py
- `ThresholdCalibrator`: wraps `sklearn.isotonic.IsotonicRegression`. Fits on `(entropy, error)` pairs, exposes `tau` and `predict_error_prob()`.
- `CalibrationResult`: dataclass with `.save()` / `.load()` JSON round-trip.
- `_safe_auroc`: handles single-class edge case.

### router.py
- `InferenceRouter`: combines probe + calibrator. `route(text)` returns `RoutingDecision`.
- `RoutingDecision`: the telemetry record. Must stay JSON-serialisable via `.to_dict()`.
- `RoutingDestination`: enum with `gcp_cloud_run` and `aws_sagemaker`.

## Invariants

- `H >= tau` **always** routes to `AWS_SAGEMAKER`. `H < tau` **always** routes to `GCP_CLOUD_RUN`. This boundary logic is tested in `test_router.py` — do not change the comparison direction.
- `ThresholdCalibrator.predict_error_prob()` is **monotone non-decreasing** (guaranteed by isotonic regression). Tests verify this.
- `AttentionEntropyProbe` is always used with `model.eval()` and `@torch.no_grad()`.

## When Modifying

- If you change `EntropyResult`, `RoutingDecision`, or `CalibrationResult` fields, update the corresponding `.to_dict()` method and the `telemetry/bigquery_schema.json`.
- If you change the entropy aggregation formula (e.g., weighting heads differently), update the docstring in `entropy.py` and the math section in `CLAUDE.md`.
- Run `pytest tests/test_entropy.py tests/test_threshold.py tests/test_router.py -v` after any change.
