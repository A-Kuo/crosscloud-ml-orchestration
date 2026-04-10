"""
BigQuery telemetry writer with async-safe batching and retries.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BigQueryTelemetryWriter:
    """
    Batches rows and flushes to BigQuery using insert_rows_json.

    This class is safe to call from async request handlers because network
    insertion is executed in a worker thread via asyncio.to_thread.
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        table: str,
        batch_size: int = 100,
        flush_interval_s: float = 2.0,
        max_retries: int = 3,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.table = table
        self.batch_size = max(1, batch_size)
        self.flush_interval_s = max(0.1, flush_interval_s)
        self.max_retries = max(1, max_retries)
        self.enabled = enabled
        self._rows: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[Any] = None
        self._table_ref = f"{project}.{dataset}.{table}"

    def start(self) -> None:
        if not self.enabled:
            logger.info("BigQuery telemetry disabled")
            return
        from google.cloud import bigquery

        self._client = bigquery.Client(project=self.project)
        self._task = asyncio.create_task(self._periodic_flush(), name="bq-telemetry-flush")
        logger.info("BigQueryTelemetryWriter started", extra={"table": self._table_ref})

    async def stop(self) -> None:
        if not self.enabled:
            return
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("BigQueryTelemetryWriter stopped")

    async def write(self, row: Any) -> None:
        if not self.enabled:
            return
        payload = _to_row_dict(row)
        async with self._lock:
            self._rows.append(payload)
            should_flush = len(self._rows) >= self.batch_size
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        if not self.enabled:
            return
        async with self._lock:
            if not self._rows:
                return
            batch = self._rows
            self._rows = []
        await self._insert_with_retries(batch)

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval_s)
            try:
                await self.flush()
            except Exception:
                logger.exception("Periodic telemetry flush failed")

    async def _insert_with_retries(self, rows: list[dict[str, Any]]) -> None:
        if self._client is None:
            return
        delay = 0.25
        for attempt in range(1, self.max_retries + 1):
            try:
                errors = await asyncio.to_thread(
                    self._client.insert_rows_json, self._table_ref, rows
                )
                if errors:
                    logger.error("BigQuery insert errors", extra={"errors": errors})
                return
            except Exception:
                logger.exception(
                    "BigQuery insert failed",
                    extra={"attempt": attempt, "table": self._table_ref},
                )
                if attempt == self.max_retries:
                    return
                await asyncio.sleep(delay)
                delay *= 2


def from_env() -> BigQueryTelemetryWriter:
    enabled = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"
    return BigQueryTelemetryWriter(
        project=os.getenv("BQ_PROJECT", "crosscloud-demo"),
        dataset=os.getenv("BQ_DATASET", "ml_telemetry"),
        table=os.getenv("BQ_TABLE", "routing_events"),
        batch_size=int(os.getenv("BQ_BATCH_SIZE", "100")),
        flush_interval_s=float(os.getenv("BQ_FLUSH_INTERVAL_S", "2.0")),
        max_retries=int(os.getenv("BQ_MAX_RETRIES", "3")),
        enabled=enabled,
    )


def _to_row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if is_dataclass(row):
        return asdict(row)
    raise TypeError(f"Unsupported telemetry row type: {type(row)}")
