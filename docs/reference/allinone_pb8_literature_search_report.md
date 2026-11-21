# PB8 Closest-Literature Search Report

Per-source hits: semantic_scholar=0, open_alex=40, arxiv=50, openreview=0, crossref=50, dblp=0 · 106 unique (34 duplicates merged) · 86 filtered as clearly irrelevant after title-and-abstract review.

The unfiltered, abstract-bearing recovery record is `reports/pb8_publication_benchmark/A_LITERATURE_SEARCH_RAW.json`.

| # | Title | Date | Venue | Citations | Score | Sources |
|---:|---|---:|---|---:|---:|---|
| 1 | [Calibrated few-shot industrial anomaly detection with frozen visual features](https://doi.org/10.2139/ssrn.7230717) | unknown | not indexed | 0 | 11 | Crossref |
| 3 | [CANDI: Curated Test-Time Adaptation for Multivariate Time-Series Anomaly Detection Under Distribution Shift](https://doi.org/10.1609/aaai.v40i17.38524) | 2026-03-14 | Proceedings of the AAAI Conference on Artificial Intelligence | 1 | 9 | Crossref |
| 6 | [Selective Test-Time Adaptation for Unsupervised Anomaly Detection using Neural Implicit Representations](http://arxiv.org/abs/2410.03306v2) | 2024-10-04 | arXiv | 0 | 8 | arXiv |
| 7 | [When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection](https://doi.org/10.1609/aaai.v38i12.29210) | 2024-03-24 | Proceedings of the AAAI Conference on Artificial Intelligence | 34 | 7 | OpenAlex, Crossref |
| 9 | [Addressing Representation and Aggregation Biases in Streaming Anomaly Detection through Domain Adaptation and Domain Shift Detection](https://doi.org/10.2139/ssrn.7070753) | unknown | not indexed | 0 | 7 | Crossref |
| 13 | [One Class Process Anomaly Detection Using Kernel Density Estimation Methods](https://doi.org/10.1109/tsm.2022.3181468) | 2022-06-08 | IEEE Transactions on Semiconductor Manufacturing | 37 | 6 | OpenAlex |
| 16 | [Add-On Anomaly Threshold Technique for Improving Unsupervised Intrusion Detection on SCADA Data](https://doi.org/10.3390/electronics9061017) | 2020-06-18 | Electronics | 13 | 6 | OpenAlex |
| 23 | [SAFE-TTA: Safety-Aware Test-Time Adaptation for Real-Time Object Detection under Adverse Weather](https://doi.org/10.14445/23488549/ijece-v13i7p107) | 2026-08-11 | International Journal of Electronics and Communication Engineering | 0 | 6 | Crossref |
| 48 | [A loss framework for calibrated anomaly detection](https://openalex.org/W7067709062) | 2018-01-01 | ANU Open Research (Australian National University) | 6 | 4 | OpenAlex |
| 50 | [An Interactive Threshold-Setting Procedure for Improved Multivariate Anomaly Detection in Time Series](https://doi.org/10.1109/access.2023.3310653) | 2023 | IEEE Access | 4 | 4 | Crossref |
| 54 | [Multi -directional decision fusion for black-box source-free anomaly detection](https://doi.org/10.1016/j.patcog.2026.113038) | 2026-01-09 | Pattern Recognition | 0 | 4 | OpenAlex |
| 59 | [DNPR: Zero-shot industrial anomaly detection via dynamic normal prototype refinement](https://doi.org/10.2139/ssrn.5786608) | unknown | not indexed | 0 | 4 | Crossref |
| 60 | [The Threshold Paradox: Why Calibrating on the Test Set Introduces Bias in Anomaly Detection](https://doi.org/10.5220/0014392600004084) | 2026 | Proceedings of the 21st International Conference on Computer Vision Theory and Applications | 0 | 4 | Crossref |
| 61 | [ViTranCoC: Vision Transformers and Conformal Calibration for Industrial Anomaly Detection](https://doi.org/10.2139/ssrn.5495023) | unknown | not indexed | 0 | 4 | Crossref |
| 66 | [Few-Shot Source-Free Domain Adaptation for Surface Defect Detection](https://doi.org/10.1109/icip61757.2026.11630359) | 2026-09 | 2026 IEEE International Conference on Image Processing (ICIP) | 0 | 4 | Crossref |
| 72 | [Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection](https://doi.org/10.48550/arxiv.2210.10487) | 2022-10-19 | arXiv (Cornell University) | 9 | 3 | OpenAlex |
| 84 | [Automatic Threshold Setting for Anomaly Detection](https://doi.org/10.2139/ssrn.4888242) | unknown | not indexed | 0 | 3 | Crossref |
| 88 | [Evaluating calibration of deep fault diagnostic models under distribution shift](https://doi.org/10.1016/j.compind.2025.104334) | 2025-07-04 | Computers in Industry | 48 | 2 | OpenAlex |
| 104 | [[survey] A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts](https://doi.org/10.1007/s11263-024-02181-w) | 2024-07-18 | International Journal of Computer Vision | 220 | 4 | OpenAlex, arXiv |
| 105 | [[survey] Trustworthy Anomaly Detection: A Survey](http://arxiv.org/abs/2202.07787v1) | 2022-02-15 | arXiv | 0 | 2 | arXiv |

## Model Knowledge (1 paper, live-verified)

| # | Title | Year | Venue | Notes |
|---:|---|---:|---|---|
| 1 | [Online Adaptive Anomaly Thresholding with Confidence Sequences](https://proceedings.mlr.press/v235/sun24h.html) | 2024 | ICML (PMLR 235) | Live-verified primary publication; closest threshold-adaptation work, but it assumes an online unlabeled score stream/cold-start protocol rather than K known target-normal acquisition-shift support. |

## Overview

Queries covered industrial visual anomaly detection under acquisition/domain shift, test-time threshold calibration, few-normal calibration, source-free operating-point adaptation, and safe selective adaptation from 2018–2026. The APIs returned 106 unique records; 20 remained plausibly relevant after semantic review, and one high-signal ICML paper missed by the failed/rate-limited sources was added and live-verified.

Source diagnostics: Semantic Scholar returned HTTP 429 for all five queries; OpenReview was unavailable because `openreview-py` is not installed; OpenAlex returned HTTP 504 on some queries; DBLP reported one proxy disconnect. These failures are preserved rather than silently treated as zero evidence.

## Trends

The closest work clusters after 2022 around test-time adaptation, source-free adaptation, and calibration. Most methods update representations, prompts, prototypes, or pseudo-label models; genuinely threshold-only work is mostly generic/online rather than industrial visual acquisition-shift calibration.

## Key themes

- Threshold adaptation and calibration: records 1, 16, 48, 50, 60, 61, 72, 84, plus the live-verified ICML paper.
- Test-time adaptation under shift: records 3, 6, 7, 9, 23, 54, 66.
- Normal-only or few-normal industrial AD: records 1, 13, 59, 61, 66.
- Safety, trust, and abstention: records 6, 23, 60, 88, 105 and the ICML paper.
- Broad positioning surveys: records 104 and 105.

## Keywords frequency

| Keyword | Count |
|---|---:|
| anomaly | 16 |
| adaptation | 7 |
| threshold | 4 |
| industrial | 3 |
| calibration | 2 |

## Most cited by accepted paper

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts | 2024 | 220 |
| 2 | Evaluating calibration of deep fault diagnostic models under distribution shift | 2025 | 48 |
| 3 | One Class Process Anomaly Detection Using Kernel Density Estimation Methods | 2022 | 37 |
| 4 | When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection | 2024 | 34 |
| 5 | Add-On Anomaly Threshold Technique for Improving Unsupervised Intrusion Detection on SCADA Data | 2020 | 13 |

## Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Jian Liang | 1 | 220 |
| 2 | Yiming Xiao | 1 | 48 |
| 3 | Christopher I. Lang | 1 | 37 |
| 4 | Dongmin Kim | 1 | 34 |
| 5 | Abdulmohsen Almalawi | 1 | 13 |

## Recommendations for reading

1. **Online Adaptive Anomaly Thresholding with Confidence Sequences (ICML 2024)** — strongest threshold-adaptation anchor and the closest deployment-neighbor, although not protocol-matched.
2. **A loss framework for calibrated anomaly detection (2018)** — foundational calibration/quantile perspective for anomaly scores.
3. **When Model Meets New Normals (AAAI 2024)** — representative anomaly-specific TTA under evolving normality, but for time series and model adaptation.
4. **Selective Test-Time Adaptation for Unsupervised Anomaly Detection (2024)** — closest selective-adaptation mechanism, but medical and representation/output adapting.
5. **A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts (IJCV 2024)** — taxonomy for positioning why PB8 is operating-point adaptation rather than representation TTA.

## Comparator conclusion

`NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS`

The live-verified ICML threshold method is the closest conceptual neighbor, but it requires an ordered unlabeled score stream/cold-start design and has no identified official public implementation for a faithful K=8 batch-normal replay. Running it here would require new sequence and hyperparameter choices. All closer industrial visual records either alter the detector/representation, use different evidence semantics, or lack verifiable publication/implementation evidence. It is therefore cited/positioned, not inserted as a fabricated B8 run.
