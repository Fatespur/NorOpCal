"""PB8 G — incremental actuator cost benchmark and deployment facts."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT
DET = "dinov2_vits14"
N_LOOPS = 100_000


def _cases() -> list[tuple[np.ndarray, np.ndarray, float]]:
    fw = P8.Firewall()
    cases: list[tuple[np.ndarray, np.ndarray, float]] = []
    for cat, cond, det in P8.k8_eligible_entities():
        if det != DET:
            continue
        src = P8.load_src(cat, det, fw)
        cnd = P8.load_cnd(cat, cond, det, fw)[:8]
        cases.append((src, cnd, P8.source_threshold(src)))
    return cases


def _calibrate(case: tuple[np.ndarray, np.ndarray, float]) -> float:
    src, cnd, tau_source = case
    score = P8.correctability_score(src, cnd, tau_source)
    if not P8.gate_adapt(score, P8.PRINCIPAL_CUTOFF):
        return tau_source
    return P8.bounded_tau(P8.PRINCIPAL_FAMILY, src, cnd, tau_source, P8.PRINCIPAL_LAMBDA)


def main() -> None:
    cases = _cases()
    for case in cases:
        _calibrate(case)
    samples = np.empty(N_LOOPS, dtype=np.int64)
    tracemalloc.start()
    start = time.perf_counter_ns()
    for index in range(N_LOOPS):
        begin = time.perf_counter_ns()
        _calibrate(cases[index % len(cases)])
        samples[index] = time.perf_counter_ns() - begin
    total_ns = time.perf_counter_ns() - start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rows: list[dict[str, Any]] = [
        {
            "component": "Base DINOv2 Memory inference",
            "measurement_status": "NOT_VERIFIED_IN_PB8",
            "runtime_p50_ms_per_entity": "",
            "runtime_p95_ms_per_entity": "",
            "runtime_mean_ms_per_entity": "",
            "additional_cpu_memory_bytes": "",
            "additional_gpu_memory_bytes": "",
            "target_normal_samples": "n/a",
            "cpu_only_feasible": "implementation-dependent",
            "retraining_required": False,
            "gradient_required": False,
            "target_anomaly_required": False,
            "extra_model_parameters": "n/a",
            "storage_overhead_bytes": "NOT_VERIFIED",
            "evidence": "Historical Phase-5 planning estimate is ~30 s per ~100-sample cell; no accepted per-image PB8 runtime artifact.",
        },
        {
            "component": "PB7/PB8 threshold actuator incremental",
            "measurement_status": "VERIFIED_CPU_MICROBENCHMARK",
            "runtime_p50_ms_per_entity": float(np.percentile(samples, 50) / 1e6),
            "runtime_p95_ms_per_entity": float(np.percentile(samples, 95) / 1e6),
            "runtime_mean_ms_per_entity": float(total_ns / N_LOOPS / 1e6),
            "additional_cpu_memory_bytes": int(peak_bytes),
            "additional_gpu_memory_bytes": 0,
            "target_normal_samples": 8,
            "cpu_only_feasible": True,
            "retraining_required": False,
            "gradient_required": False,
            "target_anomaly_required": False,
            "extra_model_parameters": 0,
            "storage_overhead_bytes": 8 * 8 + 4 * 8,
            "evidence": f"{N_LOOPS} loops over {len(cases)} DINOv2 K8-eligible entities; median/Q50 gate/scalar bounded update only.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "G_COMPUTATIONAL_OVERHEAD.csv", index=False)
    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "timer": "time.perf_counter_ns",
        "loops": N_LOOPS,
        "n_cases": len(cases),
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
    }
    (OUT / "G_OVERHEAD_ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    gate = {
        "schema_version": "1.0",
        "phase": "G",
        "incremental_actuator_measured": True,
        "base_detector_cost_separated": True,
        "base_detector_pb8_measurement": "NOT_VERIFIED",
        "complexity_statement": [
            "No detector retraining",
            "No target anomaly labels",
            "K=8 target-normal samples",
            "Scalar median computation",
            "Single lambda=0.25 bounded threshold update",
            "Frozen Q50 abstention",
        ],
        "verdict": "PASS_WITH_BASE_DETECTOR_COST_NOT_VERIFIED",
        "terminal": "PROCEED_TO_H",
    }
    (OUT / "G_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(
        "G done: incremental p50="
        f"{np.percentile(samples, 50) / 1e6:.6f} ms, p95={np.percentile(samples, 95) / 1e6:.6f} ms"
    )


if __name__ == "__main__":
    main()
