# telemetry/ — Agent Context

## Purpose

BigQuery schema definition and synthetic data generation for the routing telemetry pipeline.

## Files

### bigquery_schema.json
JSON array of BigQuery column definitions for the `routing_events` table. This is the single source of truth for what fields are logged per inference request.

Key columns:
- `h_route` (FLOAT64) — the routing signal
- `tau` (FLOAT64) — threshold at decision time
- `destination` (STRING) — `gcp_cloud_run` or `aws_sagemaker`
- `is_error` (BOOL, NULLABLE) — ground-truth label, populated asynchronously
- `error_probability` (FLOAT64) — calibrator's P(error|H) at decision time

If you add fields to `RoutingDecision.to_dict()` in `router/router.py`, add the corresponding column here.

### sample_telemetry.py
CLI script that generates realistic JSONL rows:
- Bimodal entropy distribution (70% low-entropy, 30% high-entropy)
- Sigmoid-based synthetic error labels
- Realistic latency distributions per destination

Usage: `python telemetry/sample_telemetry.py --rows 2000 --out telemetry/sample_rows.jsonl`

Generated `.jsonl` files are gitignored. Do not commit them.

## When Modifying

- Schema changes must stay in sync with `RoutingDecision.to_dict()` in `router/router.py`.
- If you change the entropy distribution parameters in `sample_telemetry.py`, update the corresponding mock distributions in the Airflow DAG (`_mock_entropy_distributions`).
