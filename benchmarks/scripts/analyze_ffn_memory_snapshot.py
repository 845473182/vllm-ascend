#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analyze torch_npu allocator snapshots captured around the routed MoE FFN.

The script has no torch/vllm dependency. It reconstructs allocated, active and
reserved memory timelines from the final allocator state and the recorded
events, reports allocations live at the peak, aggregates allocation call sites,
and compares matching FFN chunk OFF/ON snapshots.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SNAPSHOT_NAME_RE = re.compile(
    r"^moe_ffn_chunk_(?P<state>on|off)_rank(?P<rank>\d+)_tokens(?P<tokens>\d+)_"
    r"chunks(?P<chunks>\d+)_(?P<timestamp>\d+)\.pickle$"
)
_MEMORY_VIZ_URL = "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"


class _PrimitiveSnapshotUnpickler(pickle.Unpickler):
    """Load the primitive-only snapshot format without importing globals."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"snapshot contains forbidden global {module}.{name}")


@dataclass
class SnapshotMetadata:
    state: str | None = None
    rank: int | None = None
    routed_tokens: int | None = None
    chunks: int | None = None
    timestamp_ns: int | None = None


@dataclass
class TimelineSummary:
    initial_bytes: int
    peak_bytes: int
    final_bytes: int
    peak_growth_bytes: int
    minimum_bytes: int
    peak_event_index: int
    peak_event_action: str


def format_bytes(value: int | float) -> str:
    sign = "-" if value < 0 else ""
    size = abs(float(value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{sign}{size:.3f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _parse_metadata(path: Path) -> SnapshotMetadata:
    match = _SNAPSHOT_NAME_RE.match(path.name)
    if match is None:
        return SnapshotMetadata()
    values = match.groupdict()
    return SnapshotMetadata(
        state=values["state"],
        rank=int(values["rank"]),
        routed_tokens=int(values["tokens"]),
        chunks=int(values["chunks"]),
        timestamp_ns=int(values["timestamp"]),
    )


def _load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as snapshot_file:
        snapshot = _PrimitiveSnapshotUnpickler(snapshot_file).load()
    if not isinstance(snapshot, dict):
        raise ValueError(f"{path}: snapshot root must be a dictionary")
    if not isinstance(snapshot.get("segments"), list):
        raise ValueError(f"{path}: snapshot has no segments list")
    if not isinstance(snapshot.get("device_traces"), list):
        raise ValueError(f"{path}: snapshot has no device_traces list")
    return snapshot


def _frame_text(frames: Any, *, depth: int = 4) -> str:
    if not isinstance(frames, list) or not frames:
        return "<no Python stack>"
    rendered = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        filename = str(frame.get("filename", "?"))
        line = frame.get("line", "?")
        name = frame.get("name", "?")
        rendered.append(f"{Path(filename).name}:{line}:{name}")
        if len(rendered) == depth:
            break
    return " <- ".join(rendered) if rendered else "<no Python stack>"


def _event_delta(event: dict[str, Any], metric: str) -> int:
    action = event.get("action")
    size = int(event.get("size", 0))
    if metric == "allocated":
        if action == "alloc":
            return size
        if action == "free_requested":
            return -size
    elif metric == "active":
        if action == "alloc":
            return size
        if action == "free_completed":
            return -size
    elif metric == "reserved":
        if action == "segment_alloc":
            return size
        if action == "segment_free":
            return -size
    else:
        raise ValueError(f"unknown timeline metric: {metric}")
    return 0


def _reconstruct_timeline(
    events: list[dict[str, Any]],
    *,
    final_bytes: int,
    metric: str,
) -> TimelineSummary:
    net_change = sum(_event_delta(event, metric) for event in events)
    initial = final_bytes - net_change
    current = initial
    peak = current
    minimum = current
    peak_index = -1
    peak_action = "initial"
    for event_index, event in enumerate(events):
        current += _event_delta(event, metric)
        minimum = min(minimum, current)
        if current > peak:
            peak = current
            peak_index = event_index
            peak_action = str(event.get("action", "unknown"))
    return TimelineSummary(
        initial_bytes=initial,
        peak_bytes=peak,
        final_bytes=current,
        peak_growth_bytes=peak - initial,
        minimum_bytes=minimum,
        peak_event_index=peak_index,
        peak_event_action=peak_action,
    )


def _segment_device(segment: dict[str, Any], default_device: int) -> int:
    device = segment.get("device", default_device)
    return int(device) if isinstance(device, int) else default_device


def _segments_for_device(
    snapshot: dict[str, Any],
    device: int,
    default_device: int,
) -> list[dict[str, Any]]:
    return [
        segment
        for segment in snapshot["segments"]
        if isinstance(segment, dict) and _segment_device(segment, default_device) == device
    ]


def _allocator_end_state(segments: list[dict[str, Any]]) -> dict[str, int | float]:
    reserved = sum(int(segment.get("total_size", 0)) for segment in segments)
    allocated = sum(
        int(
            segment.get(
                "allocated_size",
                sum(
                    int(block.get("size", 0))
                    for block in segment.get("blocks", [])
                    if block.get("state") == "active_allocated"
                ),
            )
        )
        for segment in segments
    )
    active = sum(
        int(
            segment.get(
                "active_size",
                sum(
                    int(block.get("size", 0))
                    for block in segment.get("blocks", [])
                    if block.get("state") in {"active_allocated", "active_awaiting_free"}
                ),
            )
        )
        for segment in segments
    )
    inactive = 0
    internal_fragmentation = 0
    largest_inactive = 0
    block_count = 0
    for segment in segments:
        for block in segment.get("blocks", []):
            if not isinstance(block, dict):
                continue
            block_count += 1
            size = int(block.get("size", 0))
            if block.get("state") == "inactive":
                inactive += size
                largest_inactive = max(largest_inactive, size)
            elif block.get("state") in {"active_allocated", "active_awaiting_free"}:
                internal_fragmentation += max(0, size - int(block.get("requested_size", size)))
    return {
        "segment_count": len(segments),
        "block_count": block_count,
        "reserved_bytes": reserved,
        "allocated_bytes": allocated,
        "active_bytes": active,
        "awaiting_free_bytes": max(0, active - allocated),
        "inactive_bytes": inactive,
        "internal_fragmentation_bytes": internal_fragmentation,
        "largest_inactive_block_bytes": largest_inactive,
        "inactive_ratio": inactive / reserved if reserved else 0.0,
        "internal_fragmentation_ratio": internal_fragmentation / reserved if reserved else 0.0,
    }


def _final_live_allocations(segments: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    live: dict[int, dict[str, Any]] = {}
    for segment in segments:
        next_address = int(segment.get("address", 0))
        for block in segment.get("blocks", []):
            if not isinstance(block, dict):
                continue
            address = int(block.get("address", next_address))
            size = int(block.get("size", 0))
            if block.get("state") == "active_allocated":
                live[address] = {
                    "address": address,
                    "size_bytes": size,
                    "requested_bytes": int(block.get("requested_size", size)),
                    "stream": int(segment.get("stream", 0)),
                    "site": _frame_text(block.get("frames", [])),
                    "origin": "preexisting_or_persistent",
                }
            next_address = address + size
    return live


def _allocations_at_allocated_peak(
    events: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    peak_event_index: int,
) -> list[dict[str, Any]]:
    live = _final_live_allocations(segments)

    # Reverse the retained trace from the final allocator state to reconstruct
    # allocations that were live when history recording began.
    for event in reversed(events):
        action = event.get("action")
        address = event.get("addr")
        if not isinstance(address, int):
            continue
        if action == "alloc":
            live.pop(address, None)
        elif action == "free_requested":
            size = int(event.get("size", 0))
            live[address] = {
                "address": address,
                "size_bytes": size,
                "requested_bytes": size,
                "stream": int(event.get("stream", 0)),
                "site": _frame_text(event.get("frames", [])),
                "origin": "preexisting",
            }

    if peak_event_index < 0:
        return list(live.values())

    for event_index, event in enumerate(events):
        action = event.get("action")
        address = event.get("addr")
        if isinstance(address, int):
            if action == "alloc":
                size = int(event.get("size", 0))
                live[address] = {
                    "address": address,
                    "size_bytes": size,
                    "requested_bytes": size,
                    "stream": int(event.get("stream", 0)),
                    "site": _frame_text(event.get("frames", [])),
                    "origin": "allocated_during_capture",
                }
            elif action == "free_requested":
                live.pop(address, None)
        if event_index == peak_event_index:
            break
    return list(live.values())


def _allocation_hotspots(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sites: dict[str, dict[str, int | str]] = {}
    for event in events:
        if event.get("action") != "alloc":
            continue
        site = _frame_text(event.get("frames", []))
        size = int(event.get("size", 0))
        if site not in sites:
            sites[site] = {
                "site": site,
                "allocation_count": 0,
                "total_allocated_bytes": 0,
                "largest_allocation_bytes": 0,
            }
        row = sites[site]
        row["allocation_count"] = int(row["allocation_count"]) + 1
        row["total_allocated_bytes"] = int(row["total_allocated_bytes"]) + size
        row["largest_allocation_bytes"] = max(int(row["largest_allocation_bytes"]), size)
    return sorted(sites.values(), key=lambda row: int(row["total_allocated_bytes"]), reverse=True)


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    bytes_by_action: Counter[str] = Counter()
    streams: Counter[int] = Counter()
    oom_requests = []
    for event in events:
        action = str(event.get("action", "unknown"))
        size = int(event.get("size", 0))
        counts[action] += 1
        bytes_by_action[action] += size
        if isinstance(event.get("stream"), int):
            streams[int(event["stream"])] += 1
        if action == "oom":
            oom_requests.append(
                {
                    "requested_bytes": size,
                    "device_free_bytes": int(event.get("device_free", 0)),
                    "site": _frame_text(event.get("frames", [])),
                }
            )
    return {
        "event_count": len(events),
        "counts": dict(sorted(counts.items())),
        "bytes_by_action": dict(sorted(bytes_by_action.items())),
        "stream_event_counts": {str(stream): count for stream, count in streams.most_common()},
        "oom_requests": oom_requests,
    }


def _device_ids(snapshot: dict[str, Any]) -> list[int]:
    traces = snapshot["device_traces"]
    ids = {index for index, trace in enumerate(traces) if trace}
    default_device = max(ids, key=lambda index: len(traces[index])) if ids else 0
    ids.update(
        _segment_device(segment, default_device)
        for segment in snapshot["segments"]
        if isinstance(segment, dict)
    )
    return sorted(ids or {0})


def _select_devices(snapshot: dict[str, Any], device_arg: str) -> tuple[list[int], int]:
    available = _device_ids(snapshot)
    traces = snapshot["device_traces"]
    primary = max(available, key=lambda device: len(traces[device]) if device < len(traces) else 0)
    if device_arg == "all":
        return available, primary
    if device_arg == "auto":
        return [primary], primary
    device = int(device_arg)
    if device not in available:
        raise ValueError(f"device {device} is absent; available devices: {available}")
    return [device], device


def analyze_snapshot(path: Path, *, device_arg: str = "auto", top: int = 10) -> dict[str, Any]:
    snapshot = _load_snapshot(path)
    selected_devices, primary_device = _select_devices(snapshot, device_arg)
    all_traces = snapshot["device_traces"]
    default_device = max(
        range(len(all_traces)),
        key=lambda index: len(all_traces[index]),
        default=0,
    )
    device_reports = []
    warnings = []

    for device in selected_devices:
        raw_events = all_traces[device] if device < len(all_traces) else []
        events = [event for event in raw_events if isinstance(event, dict)]
        segments = _segments_for_device(snapshot, device, default_device)
        end_state = _allocator_end_state(segments)
        allocated = _reconstruct_timeline(
            events,
            final_bytes=int(end_state["allocated_bytes"]),
            metric="allocated",
        )
        active = _reconstruct_timeline(
            events,
            final_bytes=int(end_state["active_bytes"]),
            metric="active",
        )
        reserved = _reconstruct_timeline(
            events,
            final_bytes=int(end_state["reserved_bytes"]),
            metric="reserved",
        )
        for metric_name, timeline in (
            ("allocated", allocated),
            ("active", active),
            ("reserved", reserved),
        ):
            if timeline.minimum_bytes < 0:
                warnings.append(
                    f"device {device}: reconstructed {metric_name} timeline became negative; "
                    "the retained history may be incomplete"
                )

        peak_allocations = _allocations_at_allocated_peak(events, segments, allocated.peak_event_index)
        peak_allocations.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
        trace_peak_allocations = [
            row for row in peak_allocations if row["origin"] == "allocated_during_capture"
        ]
        device_reports.append(
            {
                "device": device,
                "allocated_timeline": asdict(allocated),
                "active_timeline": asdict(active),
                "reserved_timeline": asdict(reserved),
                "allocator_end_state": end_state,
                "event_summary": _event_summary(events),
                "largest_live_allocations_at_peak": peak_allocations[:top],
                "largest_capture_allocations_live_at_peak": trace_peak_allocations[:top],
                "allocation_hotspots": _allocation_hotspots(events)[:top],
            }
        )

    metadata = _parse_metadata(path)
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "metadata": asdict(metadata),
        "primary_device": primary_device,
        "devices": device_reports,
        "warnings": warnings,
    }


def _primary_device_report(report: dict[str, Any]) -> dict[str, Any]:
    primary_device = report["primary_device"]
    for device_report in report["devices"]:
        if device_report["device"] == primary_device:
            return device_report
    return report["devices"][0]


def _saving(off_value: int, on_value: int) -> tuple[int, float | None]:
    saved = off_value - on_value
    return saved, saved / off_value * 100 if off_value else None


def build_comparisons(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"off": [], "on": []}
    )
    warnings = []
    for report in reports:
        metadata = report["metadata"]
        state = metadata["state"]
        rank = metadata["rank"]
        tokens = metadata["routed_tokens"]
        if state in {"off", "on"} and rank is not None and tokens is not None:
            grouped[(rank, tokens)][state].append(report)

    comparisons = []
    for (rank, tokens), states in sorted(grouped.items()):
        if not states["off"] or not states["on"]:
            warnings.append(f"rank={rank}, routed_tokens={tokens}: missing chunk OFF or ON snapshot")
            continue

        def newest(items: list[dict[str, Any]]) -> dict[str, Any]:
            return max(
                items,
                key=lambda report: (
                    report["metadata"]["timestamp_ns"] or 0,
                    Path(report["path"]).stat().st_mtime_ns,
                ),
            )

        off_report = newest(states["off"])
        on_report = newest(states["on"])
        if len(states["off"]) > 1 or len(states["on"]) > 1:
            warnings.append(
                f"rank={rank}, routed_tokens={tokens}: multiple snapshots found; newest OFF and ON files were used"
            )
        off_device = _primary_device_report(off_report)
        on_device = _primary_device_report(on_report)
        off_allocated = off_device["allocated_timeline"]
        on_allocated = on_device["allocated_timeline"]
        off_active = off_device["active_timeline"]
        on_active = on_device["active_timeline"]
        off_reserved = off_device["reserved_timeline"]
        on_reserved = on_device["reserved_timeline"]

        peak_saved, peak_pct = _saving(off_allocated["peak_bytes"], on_allocated["peak_bytes"])
        growth_saved, growth_pct = _saving(
            off_allocated["peak_growth_bytes"],
            on_allocated["peak_growth_bytes"],
        )
        active_saved, active_pct = _saving(off_active["peak_bytes"], on_active["peak_bytes"])
        reserved_saved, reserved_pct = _saving(off_reserved["peak_bytes"], on_reserved["peak_bytes"])
        comparisons.append(
            {
                "rank": rank,
                "routed_tokens": tokens,
                "off_chunks": off_report["metadata"]["chunks"],
                "on_chunks": on_report["metadata"]["chunks"],
                "off_path": off_report["path"],
                "on_path": on_report["path"],
                "off_peak_allocated_bytes": off_allocated["peak_bytes"],
                "on_peak_allocated_bytes": on_allocated["peak_bytes"],
                "peak_allocated_saved_bytes": peak_saved,
                "peak_allocated_saved_percent": peak_pct,
                "off_peak_growth_bytes": off_allocated["peak_growth_bytes"],
                "on_peak_growth_bytes": on_allocated["peak_growth_bytes"],
                "peak_growth_saved_bytes": growth_saved,
                "peak_growth_saved_percent": growth_pct,
                "off_peak_active_bytes": off_active["peak_bytes"],
                "on_peak_active_bytes": on_active["peak_bytes"],
                "peak_active_saved_bytes": active_saved,
                "peak_active_saved_percent": active_pct,
                "off_peak_reserved_bytes": off_reserved["peak_bytes"],
                "on_peak_reserved_bytes": on_reserved["peak_bytes"],
                "peak_reserved_saved_bytes": reserved_saved,
                "peak_reserved_saved_percent": reserved_pct,
            }
        )
    return comparisons, warnings


def build_system_summary(comparisons: list[dict[str, Any]]) -> dict[str, int | float | None]:
    if not comparisons:
        return {}
    off_peak = max(int(row["off_peak_allocated_bytes"]) for row in comparisons)
    on_peak = max(int(row["on_peak_allocated_bytes"]) for row in comparisons)
    off_growth = max(int(row["off_peak_growth_bytes"]) for row in comparisons)
    on_growth = max(int(row["on_peak_growth_bytes"]) for row in comparisons)
    peak_saved, peak_pct = _saving(off_peak, on_peak)
    growth_saved, growth_pct = _saving(off_growth, on_growth)
    return {
        "compared_rank_token_pairs": len(comparisons),
        "max_rank_off_peak_allocated_bytes": off_peak,
        "max_rank_on_peak_allocated_bytes": on_peak,
        "max_rank_peak_allocated_saved_bytes": peak_saved,
        "max_rank_peak_allocated_saved_percent": peak_pct,
        "max_rank_off_peak_growth_bytes": off_growth,
        "max_rank_on_peak_growth_bytes": on_growth,
        "max_rank_peak_growth_saved_bytes": growth_saved,
        "max_rank_peak_growth_saved_percent": growth_pct,
    }


def _print_timeline(label: str, timeline: dict[str, Any]) -> None:
    print(
        f"  {label:<10} initial={format_bytes(timeline['initial_bytes'])}, "
        f"peak={format_bytes(timeline['peak_bytes'])}, "
        f"growth={format_bytes(timeline['peak_growth_bytes'])}, "
        f"final={format_bytes(timeline['final_bytes'])}, "
        f"peak_event={timeline['peak_event_index']}:{timeline['peak_event_action']}"
    )


def _print_rows(title: str, rows: list[dict[str, Any]], top: int) -> None:
    print(f"\n  {title}:")
    if not rows:
        print("    <none>")
        return
    for index, row in enumerate(rows[:top], start=1):
        if "size_bytes" in row:
            print(
                f"    {index:>2}. {format_bytes(row['size_bytes']):>14} "
                f"origin={row['origin']}, stream={row['stream']}, {row['site']}"
            )
        else:
            print(
                f"    {index:>2}. total={format_bytes(row['total_allocated_bytes']):>14}, "
                f"max={format_bytes(row['largest_allocation_bytes']):>14}, "
                f"count={row['allocation_count']:>5}, {row['site']}"
            )


def print_report(report: dict[str, Any], *, top: int) -> None:
    metadata = report["metadata"]
    print("=" * 100)
    print(f"Snapshot: {report['path']} ({format_bytes(report['file_size_bytes'])})")
    print(
        "Metadata: "
        f"chunk={str(metadata['state']).upper()}, rank={metadata['rank']}, "
        f"routed_tokens={metadata['routed_tokens']}, chunks={metadata['chunks']}"
    )
    for device_report in report["devices"]:
        print(f"\nDevice {device_report['device']}:")
        _print_timeline("allocated", device_report["allocated_timeline"])
        _print_timeline("active", device_report["active_timeline"])
        _print_timeline("reserved", device_report["reserved_timeline"])
        end = device_report["allocator_end_state"]
        print(
            "  end state  "
            f"segments={end['segment_count']}, blocks={end['block_count']}, "
            f"inactive={format_bytes(end['inactive_bytes'])} ({end['inactive_ratio'] * 100:.2f}%), "
            f"internal_frag={format_bytes(end['internal_fragmentation_bytes'])} "
            f"({end['internal_fragmentation_ratio'] * 100:.2f}%), "
            f"largest_inactive={format_bytes(end['largest_inactive_block_bytes'])}"
        )
        events = device_report["event_summary"]
        print(
            f"  events     total={events['event_count']}, counts={events['counts']}, "
            f"streams={events['stream_event_counts']}, OOMs={len(events['oom_requests'])}"
        )
        _print_rows(
            "largest allocations created during capture and live at allocated peak",
            device_report["largest_capture_allocations_live_at_peak"],
            top,
        )
        _print_rows("allocation hotspots during capture", device_report["allocation_hotspots"], top)
    for warning in report["warnings"]:
        print(f"\nWARNING: {warning}")


def _percent_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def print_comparisons(
    comparisons: list[dict[str, Any]],
    system_summary: dict[str, Any],
    warnings: list[str],
) -> None:
    print("\n" + "#" * 100)
    print("FFN CHUNK OFF/ON COMPARISON")
    if not comparisons:
        print("No exact OFF/ON pair was found. File names must have the same rank and routed_tokens.")
    for row in comparisons:
        print(
            f"rank={row['rank']}, routed_tokens={row['routed_tokens']}, "
            f"chunks={row['off_chunks']}->{row['on_chunks']}"
        )
        print(
            "  absolute allocated peak: "
            f"{format_bytes(row['off_peak_allocated_bytes'])} -> "
            f"{format_bytes(row['on_peak_allocated_bytes'])}, "
            f"saved={format_bytes(row['peak_allocated_saved_bytes'])} "
            f"({_percent_text(row['peak_allocated_saved_percent'])})"
        )
        print(
            "  FFN peak growth:          "
            f"{format_bytes(row['off_peak_growth_bytes'])} -> "
            f"{format_bytes(row['on_peak_growth_bytes'])}, "
            f"saved={format_bytes(row['peak_growth_saved_bytes'])} "
            f"({_percent_text(row['peak_growth_saved_percent'])})"
        )
        print(
            "  allocator active peak:    "
            f"{format_bytes(row['off_peak_active_bytes'])} -> "
            f"{format_bytes(row['on_peak_active_bytes'])}, "
            f"saved={format_bytes(row['peak_active_saved_bytes'])} "
            f"({_percent_text(row['peak_active_saved_percent'])})"
        )
        print(
            "  allocator reserved peak:  "
            f"{format_bytes(row['off_peak_reserved_bytes'])} -> "
            f"{format_bytes(row['on_peak_reserved_bytes'])}, "
            f"saved={format_bytes(row['peak_reserved_saved_bytes'])} "
            f"({_percent_text(row['peak_reserved_saved_percent'])})"
        )

    if system_summary:
        print("\nMax-rank summary (the relevant capacity bottleneck for TP/DP serving):")
        print(
            "  allocated peak: "
            f"{format_bytes(system_summary['max_rank_off_peak_allocated_bytes'])} -> "
            f"{format_bytes(system_summary['max_rank_on_peak_allocated_bytes'])}, "
            f"saved={format_bytes(system_summary['max_rank_peak_allocated_saved_bytes'])} "
            f"({_percent_text(system_summary['max_rank_peak_allocated_saved_percent'])})"
        )
        print(
            "  FFN peak growth: "
            f"{format_bytes(system_summary['max_rank_off_peak_growth_bytes'])} -> "
            f"{format_bytes(system_summary['max_rank_on_peak_growth_bytes'])}, "
            f"saved={format_bytes(system_summary['max_rank_peak_growth_saved_bytes'])} "
            f"({_percent_text(system_summary['max_rank_peak_growth_saved_percent'])})"
        )
    for warning in warnings:
        print(f"WARNING: {warning}")


def _resolve_paths(inputs: list[str], *, recursive: bool) -> list[Path]:
    paths = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if path.is_dir():
            paths.extend(path.rglob("*.pickle") if recursive else path.glob("*.pickle"))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"snapshot path does not exist: {path}")
    return sorted({path.resolve() for path in paths})


def _write_json(
    path: Path,
    reports: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    system_summary: dict[str, Any],
    warnings: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            {
                "snapshots": reports,
                "comparisons": comparisons,
                "system_summary": system_summary,
                "comparison_warnings": warnings,
            },
            output,
            indent=2,
            ensure_ascii=False,
        )


def _write_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    if not comparisons:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def _write_html(snapshot_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(snapshot_path.read_bytes()).decode("ascii")
    files_json = json.dumps([{"name": snapshot_path.name, "base64": encoded}])
    outputs = []
    for kind, label in (
        ("Active Memory Timeline", "active_memory"),
        ("Allocator State History", "allocator_state"),
    ):
        output_path = output_dir / f"{snapshot_path.stem}_{label}.html"
        output_path.write_text(
            f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{kind}: {snapshot_path.name}</title></head>
<body><script type="module">
import {{add_local_files}} from "{_MEMORY_VIZ_URL}";
add_local_files({files_json}, {json.dumps(kind)});
</script></body></html>
""",
            encoding="utf-8",
        )
        outputs.append(output_path)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Analyze torch_npu memory snapshot pickle files captured around the routed MoE FFN, "
            "and compare matching FFN chunk OFF/ON runs."
        ),
        epilog=(
            "The loader accepts primitive-only snapshot pickles and rejects Python globals. "
            "OFF/ON comparison requires file names generated by the FFN snapshot logger."
        ),
    )
    parser.add_argument("paths", nargs="+", help="Snapshot pickle files or directories")
    parser.add_argument("--device", default="auto", help="'auto', 'all', or an allocator device index")
    parser.add_argument("--top", type=int, default=10, help="Number of allocations/hotspots to print")
    parser.add_argument("--recursive", action="store_true", help="Search snapshot directories recursively")
    parser.add_argument("--brief", action="store_true", help="Print only the OFF/ON comparison")
    parser.add_argument("--json", type=Path, help="Write the complete machine-readable report")
    parser.add_argument("--csv", type=Path, help="Write OFF/ON comparison rows")
    parser.add_argument(
        "--html-dir",
        type=Path,
        help="Generate Active Memory Timeline and Allocator State History HTML files",
    )
    args = parser.parse_args(argv)
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.device not in {"auto", "all"}:
        try:
            device = int(args.device)
        except ValueError:
            parser.error("--device must be 'auto', 'all', or a non-negative integer")
        if device < 0:
            parser.error("--device must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot_paths = _resolve_paths(args.paths, recursive=args.recursive)
    except (FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not snapshot_paths:
        print("error: no .pickle snapshots found", file=sys.stderr)
        return 2

    reports = []
    failed = 0
    for snapshot_path in snapshot_paths:
        try:
            report = analyze_snapshot(snapshot_path, device_arg=args.device, top=args.top)
            reports.append(report)
            if not args.brief:
                print_report(report, top=args.top)
            if args.html_dir is not None:
                for html_path in _write_html(snapshot_path, args.html_dir):
                    print(f"HTML written: {html_path}")
        except (OSError, ValueError, pickle.UnpicklingError) as error:
            failed += 1
            print(f"error: failed to analyze {snapshot_path}: {error}", file=sys.stderr)

    comparisons, comparison_warnings = build_comparisons(reports)
    system_summary = build_system_summary(comparisons)
    print_comparisons(comparisons, system_summary, comparison_warnings)

    if args.json is not None:
        _write_json(args.json, reports, comparisons, system_summary, comparison_warnings)
        print(f"JSON written: {args.json}")
    if args.csv is not None:
        _write_csv(args.csv, comparisons)
        if comparisons:
            print(f"CSV written: {args.csv}")
        else:
            print("CSV was not written because no OFF/ON comparison pair was found.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
