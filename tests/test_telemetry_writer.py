"""Unit tests for BigQueryTelemetryWriter (mocked BigQuery client)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from telemetry.writer import BigQueryTelemetryWriter, _to_row_dict


class TestToRowDict:
    def test_dict_passthrough(self):
        assert _to_row_dict({"a": 1}) == {"a": 1}

    def test_dataclass_like_to_dict(self):
        class Row:
            def to_dict(self):
                return {"k": "v"}

        assert _to_row_dict(Row()) == {"k": "v"}


@pytest.mark.asyncio
class TestBigQueryTelemetryWriter:
    async def test_disabled_skips_client_and_insert(self):
        w = BigQueryTelemetryWriter(
            project="p",
            dataset="d",
            table="t",
            enabled=False,
        )
        w.start()
        await w.write({"x": 1})
        await w.flush()
        await w.stop()
        assert w._client is None

    async def test_flush_on_batch_size(self):
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []

        w = BigQueryTelemetryWriter(
            project="proj",
            dataset="ds",
            table="tbl",
            batch_size=2,
            flush_interval_s=3600.0,
            enabled=True,
        )
        w._client = mock_client
        await w.write({"row": 1})
        mock_client.insert_rows_json.assert_not_called()
        await w.write({"row": 2})
        mock_client.insert_rows_json.assert_called_once()
        args, _ = mock_client.insert_rows_json.call_args
        assert args[0] == "proj.ds.tbl"
        assert args[1] == [{"row": 1}, {"row": 2}]

        await w.stop()

    async def test_stop_flushes_remaining(self):
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []

        w = BigQueryTelemetryWriter(
            project="proj",
            dataset="ds",
            table="tbl",
            batch_size=10,
            flush_interval_s=3600.0,
            enabled=True,
        )
        w._client = mock_client
        await w.write({"tail": 1})
        await w.stop()
        mock_client.insert_rows_json.assert_called_once()
        call_args = mock_client.insert_rows_json.call_args[0]
        assert call_args[0] == "proj.ds.tbl"
        assert call_args[1] == [{"tail": 1}]
