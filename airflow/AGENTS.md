# airflow/ — Agent Context

## Purpose

Apache Airflow orchestration layer. A single DAG ties GCP and AWS together with four nightly tasks.

## DAG: `crosscloud_ml_orchestration`

Schedule: `0 2 * * *` (02:00 UTC daily)

### Task Graph

```
entropy_audit ──┬──► threshold_recalibrate ──┐
                │                             ├──► health_check
                └──► retrain_trigger ────────┘
```

### Task Details

| Task | Trigger | Reads XCom From | Cloud APIs |
|------|---------|----------------|------------|
| `entropy_audit` | Always | — | BigQuery (read entropy scores) |
| `threshold_recalibrate` | `audit.should_recalibrate` | `entropy_audit` | BigQuery (error labels), SSM/Secret Manager (write tau) |
| `retrain_trigger` | `audit.should_retrain` | `entropy_audit` | Vertex AI Pipelines |
| `health_check` | Always | — | HTTP to Cloud Run + SageMaker |

### Mock Mode

All cloud SDK calls are gated behind `MOCK_CLOUD=true` (the default). Mock functions:
- `_mock_entropy_distributions()` — returns synthetic Gaussian samples with slight distributional shift
- `_mock_error_labels()` — sigmoid-based synthetic errors correlated with entropy
- `_mock_vertex_pipeline_trigger()` — returns a fake job ID
- `_push_tau_to_parameter_store()` — writes to `/tmp/current_tau.json`
- `_send_alert()` — logs a warning

### XCom Contract

`entropy_audit` pushes `audit_result` dict to XCom with these keys:
- `kl_divergence: float`
- `should_recalibrate: bool`
- `should_retrain: bool`
- `n_current_samples: int`
- `n_baseline_samples: int`
- `current_mean_entropy: float`
- `baseline_mean_entropy: float`
- `current_entropies: list[float]`

Downstream tasks pull from `task_ids="entropy_audit", key="audit_result"`.

## When Modifying

- New tasks: define callable inside the `with DAG` block, create `PythonOperator`, wire into dependency graph.
- To gate real cloud calls: add `if MOCK_CLOUD: return mock_result` at the top of the callable, then add the real implementation in the `else` branch.
- The DAG imports `router.threshold` via `sys.path.insert`. If project structure changes, update that path.
- All configuration is via env vars with sensible defaults. Do not hardcode cloud-specific values.
