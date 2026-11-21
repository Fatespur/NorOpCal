"""Quickstart demo for NorOpCal.

Demonstrates how NorOpCal selectively calibrates an anomaly detector's
operating-point (decision threshold) under distribution shift using only
normal observations from the target domain.
"""

from __future__ import annotations

import numpy as np
from noropcal import NorOpCal


def main() -> None:
    print("=" * 60)
    print(" NorOpCal: Selective Operating-Point Calibration Demo")
    print("=" * 60)

    # 1. Initialize NorOpCal policy with frozen defaults (lambda=0.25, Q50 gate)
    policy = NorOpCal()
    print(f"Policy configuration: family={policy.family}, lambda={policy.lam}, gate_cutoff={policy.cutoff:.5f}")

    # 2. Simulate source domain normal score distribution (calibrated at tau_s = 0.5)
    np.random.seed(42)
    source_normals = np.random.normal(loc=0.30, scale=0.08, size=100)
    tau_source = 0.50
    print(f"\n[Source Domain] Median Normal Score: {np.median(source_normals):.4f}, Source Threshold: {tau_source:.4f}")

    # Case A: Substantial acquisition shift (scores shift upward -> potential false alarms)
    shifted_normals = np.random.normal(loc=0.42, scale=0.08, size=8)  # K=8 observations
    score_a = policy.correctability_score(source_normals, shifted_normals)
    tau_a = policy.threshold(source_normals, shifted_normals, tau_source)
    print("\n--- Case A: Acquisition Shift Detected (Median Shift >= Q50) ---")
    print(f"Target Normal Median (K=8): {np.median(shifted_normals):.4f}")
    print(f"Median Shift Signal (Delta_m): {score_a:.4f} (Gate threshold: {policy.cutoff:.4f})")
    print(f"Gate Decision: {'PASS -> Adapt' if policy.gate(score_a) else 'ABSTAIN'}")
    print(f"Calibrated Threshold: {tau_source:.4f} -> {tau_a:.4f} (bounded contraction applied)")

    # Case B: Negligible shift (fluctuation within normal range -> abstain to avoid harm)
    mild_normals = np.random.normal(loc=0.295, scale=0.08, size=8)
    score_b = policy.correctability_score(source_normals, mild_normals)
    tau_b = policy.threshold(source_normals, mild_normals, tau_source)
    print("\n--- Case B: Negligible Shift (Median Shift < Q50) ---")
    print(f"Target Normal Median (K=8): {np.median(mild_normals):.4f}")
    print(f"Median Shift Signal (Delta_m): {score_b:.4f}")
    print(f"Gate Decision: {'PASS -> Adapt' if policy.gate(score_b) else 'ABSTAIN -> Keep Source Threshold'}")
    print(f"Calibrated Threshold: {tau_b:.4f} (abstention preserves safety: tau_t == tau_s)")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
