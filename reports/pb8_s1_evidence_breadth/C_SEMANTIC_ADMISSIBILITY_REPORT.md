# PB8-S1 — Stage C: Semantic Pooling Admissibility Report

Verdict: **C-STOP** — C-STOP

## The question

Does AD2's data and acquisition semantics permit more categories to share a legal 8-sample target-normal calibration pool, without changing the frozen method and without choosing pooling rules from anomaly outcomes?

## Finding

No. Every TEST category that is K8-ineligible (rice=7, sheet_metal=4, vial=5 per acquisition domain) has a true per-domain shortage. The only resources that could mechanically reach 8 are (a) cross-acquisition-condition pooling, (b) source-domain supplementation, (c) unlabeled/private supplementation, or (d) relabeling n<8 as K8. All four are semantically inadmissible under S1-S7.

## Candidates scored against S1-S7

| ID | Pooling rule | S2 (same target-domain) | S3 (deployment) | Admissible |
|----|--------------|-------------------------|-----------------|-----------|
| C1 | Pool all public target-normal images across the acquisition conditions of one ... | FAIL | FAIL | False |
| C2 | Pad a short target pool with source-domain normal images (public good regular ... | FAIL | FAIL | False |
| C3 | Supplement a short public pool with unlabeled normals from test_private / test... | FAIL | FAIL | False |
| C4 | Run with (n<8) support and still report it as fixed K=8.... | FAIL | FAIL | False |

## Detailed rationale

### C1 — Same object category is a shared prior; mechanically reaches

- Rationale: Same object category is a shared prior; mechanically reaches >=8 unique images.
- Deployment plausibility: NO: overexposed and underexposed are opposite acquisition corruptions and shift_1/shift_2/shift_3 are distinct regimes. A user facing one concrete target shift would never calibrate the threshold with normal images from a different acquisition regime; a pooled set describes no single deployed setting.
- Semantic risk: CROSS_DOMAIN: mixes distinct acquisition regimes into a pool that matches no actual target domain; a threshold calibrated on the mixture has no defined deployment meaning and violates S2/S3.
- Verdict: `INADMISSIBLE` — Fails S2 (no single target-domain scientific meaning) and S3 (deployment implausibility).

### C2 — More normal images exist if regular-domain normals are inclu

- Rationale: More normal images exist if regular-domain normals are included.
- Deployment plausibility: NO: source-domain (regular) normals are the reference domain being adapted AWAY from; using them as target calibration support is explicitly 'source-normal filling target-normal', which the stage forbids.
- Semantic risk: SOURCE_FILLING: source-domain images do not represent the target acquisition shift, nullifying the point of adaptation.
- Verdict: `INADMISSIBLE` — Fails S2/S3 and is an explicitly forbidden operation (source-normal to fill target-normal).

### C3 — Additional normal-looking images exist in held-out private s

- Rationale: Additional normal-looking images exist in held-out private splits.
- Deployment plausibility: NO: the private splits carry no acquisition-condition label (are mixed), so their target-domain identity under acquisition shift cannot be established; they cannot be shown to belong to the same target regime, and their status is excluded upstream.
- Semantic risk: UNLABELED_DOMAIN: cannot define the target-domain semantics without condition labels; risks including images outside the target regime.
- Verdict: `INADMISSIBLE` — Fails S2 (no verifiable target-domain identity) and S4 (membership not defined by condition metadata); excluded split, OUT_OF_SCOPE.

### C4 — None (label-substitution is not a pooling protocol).

- Rationale: None (label-substitution is not a pooling protocol).
- Deployment plausibility: NO: claims an 8-sample estimate with <8 real target-normal observations.
- Semantic risk: PSEUDO_EXPANSION: explicitly forbidden ('K=min(8,n) but still labelled K8').
- Verdict: `INADMISSIBLE` — Fails S6 and is an explicitly forbidden pseudo-expansion; not a real pooling protocol.

## Conclusion

Because no pooling candidate passes S1-S7, the supplementary K8 support protocol cannot be frozen. Per the C gate, the stage enters **C-STOP** with terminal `PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION`. The original PB8 evidence breadth (EB2, fabric + walnuts) is preserved unchanged. No outcome evaluation (FREEZE/D/E/F) is run.


_This report was generated from metadata, image identity and target-normal availability only; no anomaly outcome was read during Stage C._
