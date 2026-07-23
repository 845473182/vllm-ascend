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

import importlib.util
import pickle
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[3] / "benchmarks" / "scripts" / "analyze_ffn_memory_snapshot.py"
SPEC = importlib.util.spec_from_file_location("analyze_ffn_memory_snapshot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(name: str) -> list[dict[str, object]]:
    return [{"filename": "/workspace/vllm_ascend/ops/fused_moe/moe_mlp.py", "line": 100, "name": name}]


def _snapshot(events: list[dict[str, object]], *, final_allocated: int, final_reserved: int) -> dict:
    blocks = [
        {
            "address": 1000,
            "size": 100,
            "requested_size": 100,
            "state": "active_allocated",
            "frames": _frame("persistent"),
        }
    ]
    if final_allocated > 100:
        blocks.append(
            {
                "address": 2000,
                "size": final_allocated - 100,
                "requested_size": final_allocated - 100,
                "state": "active_allocated",
                "frames": _frame("remaining_output"),
            }
        )
    blocks.append(
        {
            "address": 3000,
            "size": final_reserved - final_allocated,
            "requested_size": 0,
            "state": "inactive",
            "frames": [],
        }
    )
    return {
        "segments": [
            {
                "device": 0,
                "address": 1000,
                "total_size": final_reserved,
                "allocated_size": final_allocated,
                "active_size": final_allocated,
                "stream": 0,
                "blocks": blocks,
            }
        ],
        "device_traces": [events],
    }


def _write_snapshot(path: Path, snapshot: dict) -> None:
    with path.open("wb") as output:
        pickle.dump(snapshot, output)


def test_reconstructs_peak_and_allocation_hotspots(tmp_path):
    events = [
        {"action": "alloc", "addr": 2000, "size": 60, "stream": 0, "frames": _frame("gmm1")},
        {"action": "alloc", "addr": 3000, "size": 40, "stream": 0, "frames": _frame("swiglu")},
        {"action": "free_requested", "addr": 2000, "size": 60, "stream": 0, "frames": []},
        {"action": "free_completed", "addr": 2000, "size": 60, "stream": 0, "frames": []},
    ]
    path = tmp_path / "moe_ffn_chunk_off_rank0_tokens65536_chunks1_1.pickle"
    _write_snapshot(path, _snapshot(events, final_allocated=140, final_reserved=384))

    report = MODULE.analyze_snapshot(path, top=10)
    device = report["devices"][0]

    assert device["allocated_timeline"] == {
        "initial_bytes": 100,
        "peak_bytes": 200,
        "final_bytes": 140,
        "peak_growth_bytes": 100,
        "minimum_bytes": 100,
        "peak_event_index": 1,
        "peak_event_action": "alloc",
    }
    assert device["event_summary"]["counts"] == {
        "alloc": 2,
        "free_completed": 1,
        "free_requested": 1,
    }
    assert device["allocation_hotspots"][0]["site"].endswith("gmm1")
    assert device["largest_capture_allocations_live_at_peak"][0]["size_bytes"] == 60


def test_compares_matching_chunk_off_and_on_snapshots(tmp_path):
    off_events = [
        {"action": "alloc", "addr": 2000, "size": 100, "stream": 0, "frames": _frame("off_gmm1")},
        {"action": "free_requested", "addr": 2000, "size": 100, "stream": 0, "frames": []},
        {"action": "free_completed", "addr": 2000, "size": 100, "stream": 0, "frames": []},
    ]
    on_events = [
        {"action": "alloc", "addr": 2000, "size": 30, "stream": 0, "frames": _frame("on_gmm1")},
        {"action": "free_requested", "addr": 2000, "size": 30, "stream": 0, "frames": []},
        {"action": "free_completed", "addr": 2000, "size": 30, "stream": 0, "frames": []},
    ]
    off_path = tmp_path / "moe_ffn_chunk_off_rank0_tokens65536_chunks1_1.pickle"
    on_path = tmp_path / "moe_ffn_chunk_on_rank0_tokens65536_chunks2_2.pickle"
    _write_snapshot(off_path, _snapshot(off_events, final_allocated=100, final_reserved=256))
    _write_snapshot(on_path, _snapshot(on_events, final_allocated=100, final_reserved=256))

    reports = [MODULE.analyze_snapshot(off_path), MODULE.analyze_snapshot(on_path)]
    comparisons, warnings = MODULE.build_comparisons(reports)
    system = MODULE.build_system_summary(comparisons)

    assert warnings == []
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row["off_peak_growth_bytes"] == 100
    assert row["on_peak_growth_bytes"] == 30
    assert row["peak_growth_saved_bytes"] == 70
    assert row["peak_growth_saved_percent"] == 70
    assert system["max_rank_peak_growth_saved_bytes"] == 70


def test_restricted_loader_rejects_non_primitive_pickle(tmp_path):
    path = tmp_path / "unsafe.pickle"
    with path.open("wb") as output:
        pickle.dump(Path("/tmp/not-a-snapshot"), output)

    try:
        MODULE.analyze_snapshot(path)
    except pickle.UnpicklingError as error:
        assert "forbidden global" in str(error)
    else:
        raise AssertionError("non-primitive pickle was accepted")
