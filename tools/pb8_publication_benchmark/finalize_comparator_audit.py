"""Finalize PB8 Stage A from the saved multi-source literature search.

This stage performs no method development. It records semantic relevance and
the five frozen comparator-eligibility criteria, then emits the complete
paper-search report required by the active skill plus the PB8 audit CSV/gate.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
RAW = OUT / "A_LITERATURE_SEARCH_RAW.json"

# One-based ranks retained after title+abstract review of all 106 unique hits.
# Entries without an abstract were retained when their title remained plausibly
# relevant; clearly unrelated domains/tasks were removed.
RELEVANT_RANKS = [1, 3, 6, 7, 9, 13, 16, 23, 48, 50, 54, 59, 60, 61, 66, 72, 84, 88, 104, 105]

PMLR_THRESHOLD = {
    "title": "Online Adaptive Anomaly Thresholding with Confidence Sequences",
    "authors": [
        "Sophia Huiwen Sun",
        "Abishek Sankararaman",
        "Balakrishnan Murali Narayanaswamy",
    ],
    "year": 2024,
    "venue": "ICML (PMLR 235)",
    "url": "https://proceedings.mlr.press/v235/sun24h.html",
    "notes": (
        "Live-verified primary publication; closest threshold-adaptation work, "
        "but it assumes an online unlabeled score stream/cold-start protocol "
        "rather than K known target-normal acquisition-shift support."
    ),
}


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _source_abbrev(sources: list[str]) -> str:
    mapping = {
        "semantic_scholar": "SS",
        "open_alex": "OpenAlex",
        "arxiv": "arXiv",
        "openreview": "OpenReview",
        "crossref": "Crossref",
        "dblp": "DBLP",
    }
    return ", ".join(mapping.get(source, source) for source in sources)


def _relevant_report(payload: dict[str, Any]) -> str:
    ranked = payload["ranked_unique"]
    kept = [ranked[index - 1] for index in RELEVANT_RANKS]
    hit_line = ", ".join(
        f"{source}={count}" for source, count in payload["per_source_hits"].items()
    )
    lines = [
        "# PB8 Closest-Literature Search Report",
        "",
        (
            f"Per-source hits: {hit_line} · {payload['n_unique']} unique "
            f"({payload['n_duplicates_merged']} duplicates merged) · "
            f"{payload['n_unique'] - len(kept)} filtered as clearly irrelevant "
            "after title-and-abstract review."
        ),
        "",
        (
            "The unfiltered, abstract-bearing recovery record is "
            "`reports/pb8_publication_benchmark/A_LITERATURE_SEARCH_RAW.json`."
        ),
        "",
        "| # | Title | Date | Venue | Citations | Score | Sources |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    for index, paper in zip(RELEVANT_RANKS, kept):
        survey = "[survey] " if paper.get("is_survey") else ""
        title = f"[{survey}{_cell(paper.get('title'))}]({_cell(paper.get('url'))})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    title,
                    _cell(paper.get("publication_date") or paper.get("year") or "unknown"),
                    _cell(paper.get("venue") or "not indexed"),
                    str(paper.get("citation_count") or 0),
                    str(paper.get("relevance_score") or 0),
                    _source_abbrev(paper.get("found_in") or []),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Model Knowledge (1 paper, live-verified)",
            "",
            "| # | Title | Year | Venue | Notes |",
            "|---:|---|---:|---|---|",
            (
                f"| 1 | [{PMLR_THRESHOLD['title']}]({PMLR_THRESHOLD['url']}) | "
                f"{PMLR_THRESHOLD['year']} | {PMLR_THRESHOLD['venue']} | "
                f"{PMLR_THRESHOLD['notes']} |"
            ),
            "",
            "## Overview",
            "",
            (
                "Queries covered industrial visual anomaly detection under acquisition/domain "
                "shift, test-time threshold calibration, few-normal calibration, source-free "
                "operating-point adaptation, and safe selective adaptation from 2018–2026. "
                f"The APIs returned {payload['n_unique']} unique records; 20 remained plausibly "
                "relevant after semantic review, and one high-signal ICML paper missed by the "
                "failed/rate-limited sources was added and live-verified."
            ),
            "",
            "Source diagnostics: Semantic Scholar returned HTTP 429 for all five queries; "
            "OpenReview was unavailable because `openreview-py` is not installed; OpenAlex "
            "returned HTTP 504 on some queries; DBLP reported one proxy disconnect. These "
            "failures are preserved rather than silently treated as zero evidence.",
            "",
            "## Trends",
            "",
            (
                "The closest work clusters after 2022 around test-time adaptation, source-free "
                "adaptation, and calibration. Most methods update representations, prompts, "
                "prototypes, or pseudo-label models; genuinely threshold-only work is mostly "
                "generic/online rather than industrial visual acquisition-shift calibration."
            ),
            "",
            "## Key themes",
            "",
            "- Threshold adaptation and calibration: records 1, 16, 48, 50, 60, 61, 72, 84, plus the live-verified ICML paper.",
            "- Test-time adaptation under shift: records 3, 6, 7, 9, 23, 54, 66.",
            "- Normal-only or few-normal industrial AD: records 1, 13, 59, 61, 66.",
            "- Safety, trust, and abstention: records 6, 23, 60, 88, 105 and the ICML paper.",
            "- Broad positioning surveys: records 104 and 105.",
            "",
            "## Keywords frequency",
            "",
        ]
    )

    tokens = Counter()
    terms = ["anomaly", "adaptation", "threshold", "industrial", "calibration"]
    titles = " ".join(str(paper.get("title") or "").lower() for paper in kept)
    for term in terms:
        tokens[term] = titles.count(term)
    lines.extend(["| Keyword | Count |", "|---|---:|"])
    for term, count in tokens.most_common(5):
        lines.append(f"| {term} | {count} |")

    cited = sorted(kept, key=lambda paper: int(paper.get("citation_count") or 0), reverse=True)[:5]
    lines.extend(
        [
            "",
            "## Most cited by accepted paper",
            "",
            "| Rank | Title | Year | Citations |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, paper in enumerate(cited, 1):
        lines.append(
            f"| {rank} | {_cell(paper.get('title'))} | "
            f"{_cell(paper.get('year') or 'unknown')} | {int(paper.get('citation_count') or 0)} |"
        )

    author_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"papers": 0, "citations": 0})
    for paper in kept:
        authors = paper.get("authors") or []
        if not authors:
            continue
        first = str(authors[0])
        author_stats[first]["papers"] += 1
        author_stats[first]["citations"] += int(paper.get("citation_count") or 0)
    top_authors = sorted(
        author_stats.items(),
        key=lambda item: (item[1]["citations"], item[1]["papers"]),
        reverse=True,
    )[:5]
    lines.extend(
        [
            "",
            "## Most cited by first author",
            "",
            "| Rank | Author | Papers in set | Total citations |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, (author, stats) in enumerate(top_authors, 1):
        lines.append(f"| {rank} | {_cell(author)} | {stats['papers']} | {stats['citations']} |")

    lines.extend(
        [
            "",
            "## Recommendations for reading",
            "",
            "1. **Online Adaptive Anomaly Thresholding with Confidence Sequences (ICML 2024)** — strongest threshold-adaptation anchor and the closest deployment-neighbor, although not protocol-matched.",
            "2. **A loss framework for calibrated anomaly detection (2018)** — foundational calibration/quantile perspective for anomaly scores.",
            "3. **When Model Meets New Normals (AAAI 2024)** — representative anomaly-specific TTA under evolving normality, but for time series and model adaptation.",
            "4. **Selective Test-Time Adaptation for Unsupervised Anomaly Detection (2024)** — closest selective-adaptation mechanism, but medical and representation/output adapting.",
            "5. **A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts (IJCV 2024)** — taxonomy for positioning why PB8 is operating-point adaptation rather than representation TTA.",
            "",
            "## Comparator conclusion",
            "",
            "`NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS`",
            "",
            (
                "The live-verified ICML threshold method is the closest conceptual neighbor, "
                "but it requires an ordered unlabeled score stream/cold-start design and has "
                "no identified official public implementation for a faithful K=8 batch-normal "
                "replay. Running it here would require new sequence and hyperparameter choices. "
                "All closer industrial visual records either alter the detector/representation, "
                "use different evidence semantics, or lack verifiable publication/implementation "
                "evidence. It is therefore cited/positioned, not inserted as a fabricated B8 run."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _audit_rows() -> list[dict[str, str]]:
    return [
        {
            "method": PMLR_THRESHOLD["title"],
            "publication": "Sun et al., ICML 2024, PMLR 235:47105-47132",
            "source_url": PMLR_THRESHOLD["url"],
            "target_normal_required": "no (ordered unlabeled score stream; optional offline warm start)",
            "target_anomaly_required": "no labels; the online stream may contain anomalies",
            "retraining_required": "no detector retraining",
            "modifies_representation": "no",
            "threshold_only": "yes, with online abstention",
            "same_deployment_semantics": "NO: ordered stream/cold-start, not K known target-normal batch support",
            "reproducible_from_official_public": "NO: paper/algorithms public; no official code identified",
            "eligible_comparator": "NO",
            "reason": "Closest conceptual comparator, but faithful replay would require new stream order and tuning choices not frozen by PB8.",
        },
        {
            "method": "Selective TTA with Neural Implicit Representations",
            "publication": "Ambekar et al., arXiv:2410.03306, 2024",
            "source_url": "https://arxiv.org/abs/2410.03306",
            "target_normal_required": "no; per-test-image zero-shot adaptation",
            "target_anomaly_required": "no labels",
            "retraining_required": "yes, lightweight MLP/output adaptation",
            "modifies_representation": "yes",
            "threshold_only": "no",
            "same_deployment_semantics": "NO: medical reconstruction AD and output adaptation",
            "reproducible_from_official_public": "paper verified; matched implementation not verified",
            "eligible_comparator": "NO",
            "reason": "Selective TTA is relevant positioning, but it is not a frozen-score operating-point correction.",
        },
        {
            "method": "When Model Meets New Normals",
            "publication": "Kim et al., AAAI 2024, DOI 10.1609/aaai.v38i12.29210",
            "source_url": "https://doi.org/10.1609/aaai.v38i12.29210",
            "target_normal_required": "unlabeled evolving test sequence",
            "target_anomaly_required": "no labels",
            "retraining_required": "yes, self-supervised test-time adaptation",
            "modifies_representation": "yes",
            "threshold_only": "no",
            "same_deployment_semantics": "NO: time-series new-normal adaptation",
            "reproducible_from_official_public": "publication verified; matched visual implementation not applicable",
            "eligible_comparator": "NO",
            "reason": "Different modality and capability semantics.",
        },
        {
            "method": "AnoPLe",
            "publication": "Lee et al., arXiv:2408.13516, 2024",
            "source_url": "https://arxiv.org/abs/2408.13516",
            "target_normal_required": "yes, few normal images",
            "target_anomaly_required": "no true anomalies",
            "retraining_required": "yes, prompt/decoder learning",
            "modifies_representation": "yes",
            "threshold_only": "no",
            "same_deployment_semantics": "NO: constructs a few-shot detector",
            "reproducible_from_official_public": "YES: https://github.com/YoojLee/AnoPLe",
            "eligible_comparator": "NO",
            "reason": "Few-normal visual AD but not fair on the same frozen detector score.",
        },
        {
            "method": "Multi-directional decision fusion for black-box source-free AD",
            "publication": "Gao et al., Pattern Recognition 175:113038, 2026",
            "source_url": "https://www.sciencedirect.com/science/article/pii/S0031320326000014",
            "target_normal_required": "unlabeled target data",
            "target_anomaly_required": "no labels; pseudo-labels inferred",
            "retraining_required": "yes, target model/self-learning",
            "modifies_representation": "yes",
            "threshold_only": "no",
            "same_deployment_semantics": "NO: rotating-machinery/source-free model adaptation",
            "reproducible_from_official_public": "publication verified; official code not identified",
            "eligible_comparator": "NO",
            "reason": "Industrial and source-free, but different modality and model-adaptation semantics.",
        },
        {
            "method": "Estimating the Contamination Factor's Distribution",
            "publication": "Perini et al., arXiv:2210.10487, 2022",
            "source_url": "https://arxiv.org/abs/2210.10487",
            "target_normal_required": "no",
            "target_anomaly_required": "no labels; consumes mixed unlabeled test scores",
            "retraining_required": "no detector retraining",
            "modifies_representation": "no",
            "threshold_only": "yes",
            "same_deployment_semantics": "NO: estimates mixed-test contamination, not target-normal shift correction",
            "reproducible_from_official_public": "public code stated in paper; matched integration not verified",
            "eligible_comparator": "NO",
            "reason": "Would consume a different evidence role and violate the K8 known-normal estimand.",
        },
        {
            "method": "Calibrated few-shot industrial AD with frozen visual features",
            "publication": "Crossref record DOI 10.2139/ssrn.7230717; publication status NOT_VERIFIED",
            "source_url": "https://doi.org/10.2139/ssrn.7230717",
            "target_normal_required": "yes",
            "target_anomaly_required": "claimed no",
            "retraining_required": "claimed no",
            "modifies_representation": "fuses features/scores",
            "threshold_only": "no, detector/scorer also differs",
            "same_deployment_semantics": "NO: constructs a calibrated few-shot detector",
            "reproducible_from_official_public": "NOT_VERIFIED: exact-title/DOI search found no official page or code",
            "eligible_comparator": "NO",
            "reason": "Crossref-only record cannot satisfy the verifiable-public-implementation gate.",
        },
        {
            "method": "ViTranCoC",
            "publication": "Crossref record DOI 10.2139/ssrn.5495023; publication status NOT_VERIFIED",
            "source_url": "https://doi.org/10.2139/ssrn.5495023",
            "target_normal_required": "not verified",
            "target_anomaly_required": "not verified",
            "retraining_required": "not verified",
            "modifies_representation": "not verified",
            "threshold_only": "not verified",
            "same_deployment_semantics": "NOT_VERIFIED",
            "reproducible_from_official_public": "NOT_VERIFIED: no abstract/code found",
            "eligible_comparator": "NO",
            "reason": "Insufficient inspectable evidence; missing evidence cannot be replaced by title inference.",
        },
        {
            "method": "The Threshold Paradox",
            "publication": "Cools et al., VISAPP 2026, DOI 10.5220/0014392600004084",
            "source_url": "https://doi.org/10.5220/0014392600004084",
            "target_normal_required": "n/a",
            "target_anomaly_required": "n/a",
            "retraining_required": "n/a",
            "modifies_representation": "n/a",
            "threshold_only": "n/a",
            "same_deployment_semantics": "NO: methodological critique",
            "reproducible_from_official_public": "publication metadata verified",
            "eligible_comparator": "NO",
            "reason": "Positioning/reference only; not a deployable method.",
        },
    ]


def main() -> None:
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    report = _relevant_report(payload)
    (ROOT / "allinone.md").write_text(report, encoding="utf-8")
    (OUT / "A_LITERATURE_SEARCH_REPORT.md").write_text(report, encoding="utf-8")

    rows = _audit_rows()
    fieldnames = list(rows[0])
    with (OUT / "A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gate = {
        "schema_version": "1.1",
        "stage": "PB8_PUBLICATION_BENCHMARK_ABLATION_CLAIM_FREEZE",
        "phase": "A",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "search": {
            "n_source_records": sum(payload["per_source_hits"].values()),
            "n_unique": payload["n_unique"],
            "n_semantically_retained": len(RELEVANT_RANKS),
            "n_live_verified_additions": 1,
            "raw_artifact": "A_LITERATURE_SEARCH_RAW.json",
            "report_artifact": "A_LITERATURE_SEARCH_REPORT.md",
            "source_failures_disclosed": True,
        },
        "n_candidates_audited": len(rows),
        "n_eligible": 0,
        "criteria": [
            "no_target_anomaly_label",
            "problem_semantics_close",
            "fair_on_same_detector_score",
            "no_test_outcome_tuning",
            "verifiable_implementation",
        ],
        "verdict": "NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS",
        "benchmark_rule": (
            "Use strongest protocol-compatible frozen baselines; do not insert a "
            "literature B8 run that requires new stream/hyperparameter choices."
        ),
        "terminal": "PROCEED_TO_B",
    }
    (OUT / "A_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"A finalized: {len(rows)} candidates audited, 0 eligible")


if __name__ == "__main__":
    main()
