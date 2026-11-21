# PB8 Frozen Figure Captions

## Figure 1 — Method overview

Conservative normal-only operating-point adaptation. The deployment path uses a frozen source threshold and K=8 known-normal target support to compute a median score offset. A frozen Q50 gate selects abstention or a λ=0.25 bounded threshold update. No target anomalies, final-test labels, gradients, or detector retraining enter the path. Bottom boxes state the frozen utility, safety, and evidence-scope boundaries.

## Figure 2 — Safety–utility curve

DINOv2 K=8 safety–utility path over the already frozen λ grid at the Q50 gate. The preregistered mean ΔFNR limit is +2 percentage points. λ=0.25 is the frozen principal operating point; larger actuation increases safety cost and does not monotonically improve utility.

## Figure 3 — Detector correctability decomposition

Frozen K=8 policy decomposed across DINOv2, PatchCore (WideResNet50-2 backbone), and EfficientAD. The panels show safe-correctable headroom, achieved ΔBA, harm, and adaptation coverage. DINOv2 supports useful/safe/selectable actuation; PatchCore is actuator-mismatched; EfficientAD has no safe headroom.

## Figure 4 — Few-normal robustness

K=4, K=8, and FULL results for the DINOv2 principal analysis and pooled secondary analysis. K=4 is unsafe, K=8 is the principal safe setting, and DINOv2 FULL exceeds the mean ΔFNR budget, demonstrating that additional support is not monotonically safer.

## Figure 5 — Residual harm map

Entity-level mean harm over 30 sealed K=8 support repeats for the DINOv2 frozen policy. The only non-zero category-condition cell is walnuts/shift 1, making the localized residual-risk concentration explicit. The map covers only the two frozen TEST K=8-eligible categories.
