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


def _is_moe_stage(category: str) -> bool:
    """Whether a category is an MoE/FFN stage.

    MoE stages run once per layer, so their per-stage logs are far too noisy;
    they are aggregated into the summary instead of printed individually.
    """
    return "moe" in category or "ffn" in category


def _is_attention_stage(category: str) -> bool:
    """Whether a category is an attention stage (whole scope or component)."""
    return "_attn_" in category or category.endswith("_attention")


def _attn_stage_name(category: str) -> str:
    """Human readable component name, e.g. `target_attn_q_proj` -> `q_proj`."""
    return category.split("_attn_", 1)[1] if "_attn_" in category else category


def _fmt_layer(layer_idx: int | None) -> str:
    return "-" if layer_idx is None else str(layer_idx)


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
    # Category of the innermost scope active when this scope's peak was set,
    # i.e. the "driver" of the peak. None if the peak happened outside any
    # nested measured scope.
    peak_component: str | None = None


@dataclass(frozen=True)
class ActivationBufferRecord:
    category: str
    label: str
    size_bytes: int


@dataclass
class _ActiveScope:
    category: str
    label: str
    layer_idx: int | None
    num_tokens: int | None
    baseline_bytes: int
    started_at: float
    peak_bytes: int = 0
    # Innermost measured scope active when peak_bytes was recorded (the
    # component whose execution drove the peak); None if no nested scope was
    # active at that moment.
    peak_component: str | None = None


@dataclass
class ActivationPeakProfiler:
    """Collect nested peak measurements without losing the global profile peak."""

    rank: int
    records: list[ActivationPeakRecord] = field(default_factory=list)
    buffers: list[ActivationBufferRecord] = field(default_factory=list)
    overall_peak_bytes: int = 0
    verbose_stages: bool = False
    # Suppress per-layer/per-chunk logs; only emit the end-of-step summary.
    quiet_stages: bool = False
    _scopes: list[_ActiveScope] = field(default_factory=list)

    def _synchronize(self) -> None:
        torch.npu.synchronize()

    def _checkpoint(self) -> int:
        self._synchronize()
        peak_bytes = int(torch.npu.max_memory_allocated())
        prev_overall = self.overall_peak_bytes
        self.overall_peak_bytes = max(self.overall_peak_bytes, peak_bytes)
        if peak_bytes > prev_overall and peak_bytes > 0:
            # This checkpoint set a new all-time peak: remember which nested
            # measured scope was executing when it happened, so the summary
            # can attribute the peak to the component that drove it.
            driver = self._scopes[-1].category if len(self._scopes) > 1 else None
            for scope in self._scopes:
                if scope.peak_bytes < peak_bytes:
                    scope.peak_bytes = peak_bytes
                    scope.peak_component = driver
        else:
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
                parent = self._scopes[-1]
                if peak_bytes > parent.peak_bytes:
                    parent.peak_bytes = peak_bytes
                    parent.peak_component = scope.peak_component
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
                peak_component=scope.peak_component,
            )
            self.records.append(record)
            if self.verbose_stages and not self.quiet_stages and not _is_moe_stage(category):
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

    def record_buffer(self, category: str, label: str, size_bytes: int) -> None:
        if self.quiet_stages:
            return
        record = ActivationBufferRecord(
            category=category,
            label=label,
            size_bytes=size_bytes,
        )
        self.buffers.append(record)
        logger.warning(
            "[ACTIVATION_PEAK][BUFFER] rank=%s category=%s label=%s persistent=%.3f MiB (%.3f GiB)",
            self.rank,
            category.upper(),
            label,
            size_bytes / _MIB,
            size_bytes / _GIB,
        )

    def log_request_summary(self) -> None:
        """Log the focused per-request summary.

        Prints:
        - the overall NPU memory peak of the step,
        - the attention peak and the layer it occurred on,
        - the MoE peak and the layer it occurred on,
        - the per-component attention memory breakdown at the attention peak layer.
        """
        if not self.records:
            logger.warning("[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s no measured stages", self.rank)
            return

        attention_records = [
            record for record in self.records if _is_attention_stage(record.category)
        ]
        # The attention peak is the whole-attention scope (`*_attention`);
        # `*_attn_*` records are its components and must not be selected here,
        # otherwise the peak line points at a component and the component
        # breakdown lists the whole scope as one of its own parts.
        # Select by absolute peak_bytes (not growth) so the attention breakdown
        # reflects the layer where the all-time peak actually occurred, matching
        # the total_peak line. Selecting by growth would pick the layer with the
        # largest *increment* (often layer 0, whose low baseline inflates growth)
        # and misattribute the peak to a different layer than total_peak.
        whole_attention_records = [
            record for record in attention_records if record.category.endswith("_attention")
        ]
        attn_peak = max(
            whole_attention_records, key=lambda item: item.peak_bytes, default=None
        )
        overall_peak_record = max(self.records, key=lambda item: item.peak_bytes, default=None)

        logger.warning(
            "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s total_peak=%.3f GiB "
            "at layer=%s label=%s",
            self.rank,
            self.overall_peak_bytes / _GIB,
            _fmt_layer(overall_peak_record.layer_idx) if overall_peak_record else "-",
            overall_peak_record.label if overall_peak_record else "-",
        )
        if attn_peak is not None:
            peak_layer_components = [
                record
                for record in attention_records
                if record.layer_idx == attn_peak.layer_idx
                and record.category != attn_peak.category
            ]
            # Determine which component the attention peak occurred in.
            # Prefer the checkpoint-time attribution (`peak_component`), which
            # records the innermost scope active the instant a new all-time peak
            # was first observed. That attribution is timing-sensitive: if the
            # new peak first shows up on a component's *entry* checkpoint (the
            # component scope has not been pushed yet, so the scope stack holds
            # only the whole-attention scope), `peak_component` is None even
            # though the peak belongs to the component that is about to run --
            # its later exit checkpoint observes the same peak value.
            #
            # Fall back to matching by absolute peak: components execute
            # sequentially, so the component whose `peak_bytes` equals the
            # attention peak is the one executing when the peak was set.
            driver = next(
                (
                    record
                    for record in peak_layer_components
                    if attn_peak.peak_component == record.category
                ),
                None,
            )
            if driver is None:
                driver = next(
                    (
                        record
                        for record in peak_layer_components
                        if record.peak_bytes == attn_peak.peak_bytes
                    ),
                    None,
                )
            # Annotate which component (q_proj, core, o_proj, hc_post, ...) was
            # executing when the attention peak was recorded, so the peak
            # location is visible at a glance instead of only in the breakdown.
            peak_component_name = (
                _attn_stage_name(driver.category) if driver is not None else "unknown"
            )
            logger.warning(
                "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn_peak=%.3f GiB "
                "(growth=%.3f MiB) at layer=%s label=%s component=%s",
                self.rank,
                attn_peak.peak_bytes / _GIB,
                attn_peak.peak_growth_bytes / _MIB,
                _fmt_layer(attn_peak.layer_idx),
                attn_peak.label,
                peak_component_name,
            )
            for record in sorted(
                peak_layer_components,
                key=lambda item: item.peak_growth_bytes,
                reverse=True,
            ):
                logger.warning(
                    "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn component=%s "
                    "peak=%.3f GiB growth=%.3f MiB",
                    self.rank,
                    _attn_stage_name(record.category),
                    record.peak_bytes / _GIB,
                    record.peak_growth_bytes / _MIB,
                )
            if peak_layer_components:
                # The components run sequentially, so their growths do not
                # coexist and must not be summed. The attention peak
                # decomposes instead as: the growth of the component that was
                # executing when the peak was recorded, plus the baseline
                # memory already resident when that component started.
                if driver is not None:
                    carried_bytes = attn_peak.peak_bytes - driver.peak_growth_bytes
                    logger.warning(
                        "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn peak breakdown: "
                        "%s growth=%.3f MiB + carried=%.3f MiB (non-component residual "
                        "at component entry) = %.3f MiB total",
                        self.rank,
                        _attn_stage_name(driver.category),
                        driver.peak_growth_bytes / _MIB,
                        carried_bytes / _MIB,
                        attn_peak.peak_bytes / _MIB,
                    )
                else:
                    logger.warning(
                        "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn peak breakdown: "
                        "peak=%.3f MiB occurred outside measured attn components "
                        "(between components; growth values above do not sum)",
                        self.rank,
                        attn_peak.peak_bytes / _MIB,
                    )

        moe_records = [
            record
            for record in self.records
            if record.category.endswith("_moe") or "moe" in record.category or "ffn" in record.category
        ]
        # Select by absolute peak_bytes to match total_peak attribution (see
        # attn_peak selection rationale above).
        moe_peak = max(moe_records, key=lambda item: item.peak_bytes, default=None)
        if moe_peak is not None:
            logger.warning(
                "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s moe_peak=%.3f GiB "
                "(growth=%.3f MiB) at layer=%s label=%s",
                self.rank,
                moe_peak.peak_bytes / _GIB,
                moe_peak.peak_growth_bytes / _MIB,
                _fmt_layer(moe_peak.layer_idx),
                moe_peak.label,
            )

    def log_summary(self) -> None:
        if not self.records and not self.buffers:
            logger.warning("[ACTIVATION_PEAK][SUMMARY] rank=%s no measured stages", self.rank)
            return

        category_winners: dict[str, ActivationPeakRecord] = {}
        for record in self.records:
            previous = category_winners.get(record.category)
            if previous is None or record.peak_growth_bytes > previous.peak_growth_bytes:
                category_winners[record.category] = record

        for category, record in sorted(category_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s max_growth=%.3f MiB (%.3f GiB) "
                "absolute_peak=%.3f GiB label=%s layer=%s tokens=%s",
                self.rank,
                category.upper(),
                record.peak_growth_bytes / _MIB,
                record.peak_growth_bytes / _GIB,
                record.peak_bytes / _GIB,
                record.label,
                "-" if record.layer_idx is None else record.layer_idx,
                "-" if record.num_tokens is None else record.num_tokens,
            )

        buffer_winners: dict[str, ActivationBufferRecord] = {}
        for record in self.buffers:
            previous = buffer_winners.get(record.category)
            if previous is None or record.size_bytes > previous.size_bytes:
                buffer_winners[record.category] = record
        for category, record in sorted(buffer_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s persistent=%.3f MiB (%.3f GiB) label=%s",
                self.rank,
                category.upper(),
                record.size_bytes / _MIB,
                record.size_bytes / _GIB,
                record.label,
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
        dominant_activation = max(components, key=lambda item: item.peak_growth_bytes, default=None)
        dominant_buffer = max(buffer_winners.values(), key=lambda item: item.size_bytes, default=None)
        if dominant_buffer is not None and (
            dominant_activation is None or dominant_buffer.size_bytes > dominant_activation.peak_growth_bytes
        ):
            dominant_component_category = dominant_buffer.category.upper()
            dominant_component_kind = "PERSISTENT_BUFFER"
            dominant_component_bytes = dominant_buffer.size_bytes
        elif dominant_activation is not None:
            dominant_component_category = dominant_activation.category.upper()
            dominant_component_kind = "PEAK_GROWTH"
            dominant_component_bytes = dominant_activation.peak_growth_bytes
        else:
            dominant_component_category = "UNKNOWN"
            dominant_component_kind = "UNKNOWN"
            dominant_component_bytes = 0
        logger.warning(
            "[ACTIVATION_PEAK][VERDICT] rank=%s dominant_top_level=%s "
            "top_level_growth=%s dominant_component=%s component_kind=%s "
            "component_bytes=%.3f MiB profile_absolute_peak=%.3f GiB",
            self.rank,
            dominant_top.category.upper() if dominant_top else "UNKNOWN",
            f"{dominant_top.peak_growth_bytes / _MIB:.3f} MiB" if dominant_top else "unknown",
            dominant_component_category,
            dominant_component_kind,
            dominant_component_bytes / _MIB,
            self.overall_peak_bytes / _GIB,
        )


_ACTIVE_PROFILER: ContextVar[ActivationPeakProfiler | None] = ContextVar(
    "ascend_activation_peak_profiler",
    default=None,
)


def is_activation_peak_profiling() -> bool:
    return _ACTIVE_PROFILER.get() is not None


@contextmanager
def activation_peak_profile_session(
    *,
    enabled: bool,
    rank: int,
    verbose_stages: bool = True,
) -> Iterator[ActivationPeakProfiler | None]:
    if not enabled:
        yield None
        return

    profiler = ActivationPeakProfiler(rank=rank, verbose_stages=verbose_stages)
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
def request_activation_peak_session(
    *,
    enabled: bool,
    rank: int,
    num_tokens: int,
) -> Iterator[None]:
    """Per-forward activation peak session for real (served) requests.

    Unlike `activation_peak_profile_session` (used by the startup dummy
    runs), this wraps one real scheduler step and prints the per-category
    summary exactly once per invocation, without the per-stage detail log.
    """
    if not enabled:
        yield
        return

    profiler = ActivationPeakProfiler(rank=rank, quiet_stages=True)
    profiler._synchronize()
    profiler.overall_peak_bytes = int(torch.npu.max_memory_allocated())
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    logger.warning(
        "[ACTIVATION_PEAK] rank=%s serving-step profiling started (num_tokens=%s); "
        "NPU synchronization is active and intended only for debugging.",
        rank,
        num_tokens,
    )
    try:
        yield
    finally:
        profiler._checkpoint()
        _ACTIVE_PROFILER.reset(token)
        logger.warning(
            "[ACTIVATION_PEAK][REQUEST] rank=%s num_tokens=%s absolute_peak=%.3f GiB",
            rank,
            num_tokens,
            profiler.overall_peak_bytes / _GIB,
        )
        profiler.log_request_summary()


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


def record_activation_buffer(category: str, label: str, size_bytes: int) -> None:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.record_buffer(category, label, size_bytes)


def log_ffn_chunk_decision(
    *,
    comm_method: str,
    path: str,
    enabled: bool,
    raw_tokens: int,
    dispatched_tokens: int | None,
    chunk_size: int,
    num_chunks: int,
    applied: bool,
    reason: str,
) -> None:
    """Log one MoE chunking decision inside an activation profile scope."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None or profiler.quiet_stages:
        return

    scope = next(
        (scope for scope in reversed(profiler._scopes) if scope.category in {"target_moe", "mtp_moe"}),
        profiler._scopes[-1] if profiler._scopes else None,
    )
    logger.warning(
        "[ACTIVATION_PEAK][FFN_CHUNK] rank=%s category=%s label=%s "
        "layer=%s comm=%s path=%s enabled=%s raw_tokens=%s "
        "dispatched_tokens=%s chunk_size=%s chunks=%s applied=%s reason=%s",
        profiler.rank,
        scope.category.upper() if scope is not None else "UNKNOWN",
        scope.label if scope is not None else "-",
        "-" if scope is None or scope.layer_idx is None else scope.layer_idx,
        comm_method,
        path,
        enabled,
        raw_tokens,
        "-" if dispatched_tokens is None else dispatched_tokens,
        chunk_size,
        num_chunks,
        applied,
        reason,
    )


def _current_mtp_moe_scope(profiler: ActivationPeakProfiler) -> _ActiveScope | None:
    for scope in reversed(profiler._scopes):
        if scope.category.startswith("mtp_") and ("moe" in scope.category or "ffn" in scope.category):
            return scope
    return None


@contextmanager
def record_mtp_moe_substage(
    component: str,
    label: str,
    *,
    num_tokens: int | None = None,
) -> Iterator[None]:
    """Record a detailed MTP MoE substage when inside an MTP MoE scope."""
    profiler = _ACTIVE_PROFILER.get()
    scope = _current_mtp_moe_scope(profiler) if profiler is not None else None
    if profiler is None or scope is None:
        yield
        return

    with profiler.record(
        f"mtp_moe_{component}",
        f"mtp.layer.{scope.layer_idx}.{label}",
        layer_idx=scope.layer_idx,
        num_tokens=num_tokens,
    ):
        yield


def log_mtp_moe_chunk_detail(
    *,
    chunk_idx: int,
    num_chunks: int,
    raw_start: int,
    raw_end: int,
    dispatched_tokens: int | None,
) -> None:
    """Log raw and dispatched token counts for one MTP MoE chunk."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None or profiler.quiet_stages:
        return
    scope = _current_mtp_moe_scope(profiler)
    if scope is None:
        return

    logger.warning(
        "[ACTIVATION_PEAK][FFN_CHUNK_DETAIL] rank=%s category=MTP_MOE "
        "layer=%s chunk=%s/%s raw_range=%s:%s raw_tokens=%s dispatched_tokens=%s",
        profiler.rank,
        "-" if scope.layer_idx is None else scope.layer_idx,
        chunk_idx + 1,
        num_chunks,
        raw_start,
        raw_end,
        raw_end - raw_start,
        "-" if dispatched_tokens is None else dispatched_tokens,
    )
