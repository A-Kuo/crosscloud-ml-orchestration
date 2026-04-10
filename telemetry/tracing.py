"""
Minimal tracing helpers for request correlation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    started_at_utc: float


def new_trace(trace_id: Optional[str] = None, parent_span_id: Optional[str] = None) -> TraceContext:
    return TraceContext(
        trace_id=trace_id or str(uuid.uuid4()),
        span_id=str(uuid.uuid4()),
        parent_span_id=parent_span_id,
        started_at_utc=time.time(),
    )


def trace_headers(ctx: TraceContext) -> dict[str, str]:
    return {
        "X-Trace-ID": ctx.trace_id,
        "X-Span-ID": ctx.span_id,
    }
