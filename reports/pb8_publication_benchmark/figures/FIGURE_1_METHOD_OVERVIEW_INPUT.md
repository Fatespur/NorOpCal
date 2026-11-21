# ShiftAD PB8: conservative normal-only operating-point adaptation

## Data foundation

- Frozen source-domain threshold and a fixed anomaly-score model.
- Eight known-normal target support samples per eligible acquisition-shift entity.
- No target anomalies, masks, final-test labels, gradient updates, or retraining.

## Main method: frozen conservative threshold actuation

### Component A: estimate an outcome-free shift signal

- Compute the raw median score offset between the target normal support and the source reference.
- Compare the median shift with the frozen source-calibrated Q50 gate.

### Component B: selective abstention branch

- Gate rejected: abstain and preserve the source threshold (NoAdapt).
- Gate accepted: pass the raw median offset to conservative actuation.

### Component C: bounded threshold update branch

- Shrink the accepted raw offset with the frozen factor lambda = 0.25.
- Apply the bounded scalar update to obtain the target operating threshold.

### Merge and inference

- Merge the abstention and accepted-update branches into one deployment threshold.
- Keep DINOv2 features and anomaly scores fixed.
- Apply the selected threshold to produce binary normal/anomalous decisions.

## Validation and reporting branches

- Utility branch: balanced-accuracy gain and false-positive-rate change versus NoAdapt.
- Safety branch: false-negative-rate increase, harm rate, and catastrophic-harm rate.
- Selectivity branch: coverage, benefit precision, and benefit recall.
- Scope branch: frozen TEST K=8 eligibility, detector-specific interpretation, and residual-risk localization.

## Final output

- A DINOv2-specific, normal-only, test-time operating-point adaptation claim under the frozen K=8 acquisition-shift protocol.
- Explicit non-claims: no representation improvement, no AUROC/AUPR improvement, no detector-agnostic generalization, and no end-to-end runtime superiority.
