# Phase 4.5 Unified Baseline Comparison Configurations

## Overview

This directory contains the formal, version-controlled protocol and template
configurations for the Phase 4.5 unified baseline comparison between
DINOv2 Memory and PatchCore on five MVTec AD categories.

## Files

| File | Purpose |
|------|---------|
| `protocol.yaml` | Formal protocol definition (categories, seeds, quantile, methods) |
| `base_dino_memory.yaml` | DINOv2 Memory config template with `env:` path injection |
| `base_patchcore.yaml` | PatchCore config template with `env:` path injection |
| `generated/` | Auto-generated individual configs (gitignored, contains resolved paths) |

## Data Path Injection

Formal configs use `env:SHIFTAD_MVTEC_ROOT` for the dataset root, keeping
them portable and free of local absolute paths.

Set the environment variable before running experiments:

```powershell
$env:SHIFTAD_MVTEC_ROOT="/path/to/mvtec_ad"
```

## Generating Experiment Configs

```bash
python tools/generate_phase45_configs.py
```

This produces 30 YAML files in `generated/`:

```
generated/{category}_{method}_seed{seed}.yaml
```

## Running Experiments

```bash
# Set the dataset root
export SHIFTAD_MVTEC_ROOT=/path/to/mvtec_ad

# Build memory and evaluate (example: bottle, dino_memory, seed 0)
python tools/build_memory.py --config configs/experiments/phase45/generated/bottle_dino_memory_seed0.yaml
python tools/evaluate.py \
  --config configs/experiments/phase45/generated/bottle_dino_memory_seed0.yaml \
  --model-dir outputs/phase45_baseline_comparison/bottle_dino_memory_seed0/model
```

## Protocol Audit

```bash
python tools/verify_phase45_protocol.py --output-base outputs/phase45_baseline_comparison
```

## Result Aggregation

```bash
python tools/aggregate_phase45_results.py --output-base outputs/phase45_baseline_comparison
```

## Protocol Parameters

| Parameter | Value |
|-----------|-------|
| Categories | bottle, cable, hazelnut, metal_nut, transistor |
| Split seeds | 0, 1, 2 |
| Calibration ratio | 0.20 |
| Threshold quantile | 0.95 |
| Methods | dino_memory (vits14), patchcore (WideResNet50_2) |
| Total runs | 30 (5 x 3 x 2) |

Both methods share the same memory/calibration split for each
category x seed combination. The split is deterministic given
`total_samples`, `calibration_ratio`, and `seed`.
