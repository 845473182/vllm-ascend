# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scoped activation-memory peak diagnostics for Ascend profile runs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from vllm.logger import logger

_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True)
class ActivationPeakRecord:
    category: str
    label: str
    layer_idx: int | None
    num_tokens: int | None
    baseline_bytes: int
    peak_bytes: int
    allocated_after_bytes: int
    peak_growth_bytes: int
    retained_bytes: int
    elapsed_ms: float


@dataclass
class _ActiveScope:
    category: str
    label: str
    layer_idx: int | None
    num_tokens: int | None
    baseline_bytes: int
    started_at: float
    peak_bytes: int = 0


@dataclass
class ActivationPeakProfiler:
    """Collect nested peak measurements without losing the global profile peak."""

    rank: int
    records: list[ActivationPeakRecord] = field(default_factory=list)
    overall_peak_bytes: int = 0
    _scopes: list[_ActiveScope] = field(default_factory=list)

    def _synchronize(self) -> None:
        torch.npu.synchronize()

    def _checkpoint(self) -> int:
        self._synchronize()
        peak_bytes = int(torch.npu.max_memory_allocated())
        self.overall_peak_bytes = max(self.overall_peak_bytes, peak_bytes)
        for scope in self._scopes:
            scope.peak_bytes = max(scope.peak_bytes, peak_bytes)
        return peak_bytes

    @contextmanager
    def record(
        self,
        category: str,
        label: str,
        *,
        layer_idx: int | None = None,
        num_tokens: int | None = None,
    ) -> Iterator[None]:
        # Preserve parent work performed before resetting the peak for this
        # child. Nested scopes propagate their peak back to every parent.
        self._checkpoint()
        baseline_bytes = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        scope = _ActiveScope(
            category=category,
            label=label,
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            baseline_bytes=baseline_bytes,
            started_at=time.perf_counter(),
            peak_bytes=baseline_bytes,
        )
        self._scopes.append(scope)
        try:
            yield
        finally:
            self._checkpoint()
            allocated_after_bytes = int(torch.npu.memory_allocated())
            self._scopes.pop()
            peak_bytes = max(scope.peak_bytes, allocated_after_bytes)
            if self._scopes:
                self._scopes[-1].peak_bytes = max(self._scopes[-1].peak_bytes, peak_bytes)
            record = ActivationPeakRecord(
                category=category,
                label=label,
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                baseline_bytes=baseline_bytes,
                peak_bytes=peak_bytes,
                allocated_after_bytes=allocated_after_bytes,
                peak_growth_bytes=max(0, peak_bytes - baseline_bytes),
                retained_bytes=allocated_after_bytes - baseline_bytes,
                elapsed_ms=(time.perf_counter() - scope.started_at) * 1000,
            )
            self.records.append(record)
            logger.warning(
                "[ACTIVATION_PEAK][STAGE] rank=%s category=%s label=%s "
                "layer=%s tokens=%s baseline=%.3f GiB peak=%.3f GiB "
                "growth=%.3f MiB retained=%+.3f MiB elapsed=%.3f ms",
                self.rank,
                category.upper(),
                label,
                "-" if layer_idx is None else layer_idx,
                "-" if num_tokens is None else num_tokens,
                baseline_bytes / _GIB,
                peak_bytes / _GIB,
                record.peak_growth_bytes / _MIB,
                record.retained_bytes / _MIB,
                record.elapsed_ms,
            )

    def log_summary(self) -> None:
        if not self.records:
            logger.warning("[ACTIVATION_PEAK][SUMMARY] rank=%s no measured stages", self.rank)
            return

        category_winners: dict[str, ActivationPeakRecord] = {}
        for record in self.records:
            previous = category_winners.get(record.category)
            if previous is None or record.peak_growth_bytes > previous.peak_growth_bytes:
                category_winners[record.category] = record

        for category, record in sorted(category_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s max_growth=%.3f MiB "
                "absolute_peak=%.3f GiB label=%s layer=%s tokens=%s",
                self.rank,
                category.upper(),
                record.peak_growth_bytes / _MIB,
                record.peak_bytes / _GIB,
                record.label,
                "-" if record.layer_idx is None else record.layer_idx,
                "-" if record.num_tokens is None else record.num_tokens,
            )

        top_level = [
            record
            for category in ("target_model", "mtp_draft")
            if (record := category_winners.get(category)) is not None
        ]
        components = [
            record for category, record in category_winners.items() if category not in {"target_model", "mtp_draft"}
        ]
        dominant_top = max(top_level, key=lambda item: item.peak_growth_bytes, default=None)
        dominant_component = max(components, key=lambda item: item.peak_growth_bytes, default=None)
        logger.warning(
            "[ACTIVATION_PEAK][VERDICT] rank=%s dominant_top_level=%s "
            "top_level_growth=%s dominant_component=%s component_growth=%s "
            "profile_absolute_peak=%.3f GiB",
            self.rank,
            dominant_top.category.upper() if dominant_top else "UNKNOWN",
            f"{dominant_top.peak_growth_bytes / _MIB:.3f} MiB" if dominant_top else "unknown",
            dominant_component.category.upper() if dominant_component else "UNKNOWN",
            f"{dominant_component.peak_growth_bytes / _MIB:.3f} MiB" if dominant_component else "unknown",
            self.overall_peak_bytes / _GIB,
        )


_ACTIVE_PROFILER: ContextVar[ActivationPeakProfiler | None] = ContextVar(
    "ascend_activation_peak_profiler",
    default=None,
)


@contextmanager
def activation_peak_profile_session(
    *,
    enabled: bool,
    rank: int,
) -> Iterator[ActivationPeakProfiler | None]:
    if not enabled:
        yield None
        return

    profiler = ActivationPeakProfiler(rank=rank)
    profiler._synchronize()
    profiler.overall_peak_bytes = int(torch.npu.max_memory_allocated())
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    logger.warning(
        "[ACTIVATION_PEAK] rank=%s profiling enabled; NPU synchronization is active and intended only for debugging.",
        rank,
    )
    try:
        yield profiler
    finally:
        profiler._checkpoint()
        _ACTIVE_PROFILER.reset(token)
        profiler.log_summary()


@contextmanager
def record_activation_peak(
    category: str,
    label: str,
    *,
    layer_idx: int | None = None,
    num_tokens: int | None = None,
) -> Iterator[None]:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.record(
        category,
        label,
        layer_idx=layer_idx,
        num_tokens=num_tokens,
    ):
        yield
