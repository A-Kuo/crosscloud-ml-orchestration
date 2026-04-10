"""Tests for telemetry.tracing helpers."""

from __future__ import annotations

from telemetry.tracing import new_trace, trace_headers


def test_new_trace_generates_ids_when_none():
    ctx = new_trace()
    assert ctx.trace_id
    assert ctx.span_id
    assert ctx.trace_id != ctx.span_id


def test_new_trace_respects_provided_trace_id():
    ctx = new_trace(trace_id="fixed-id")
    assert ctx.trace_id == "fixed-id"
    assert ctx.span_id


def test_trace_headers_shape():
    ctx = new_trace(trace_id="tid")
    h = trace_headers(ctx)
    assert h["X-Trace-ID"] == "tid"
    assert h["X-Span-ID"] == ctx.span_id
