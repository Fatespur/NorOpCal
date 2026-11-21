"""Run the auditable PB8 closest-literature search.

The installed ``paper-search`` CLI does not expose the ``--json`` option
documented by its skill.  This wrapper uses the documented programmatic API,
preserves every source result (including abstracts), and also stores the
deduplicated lexical ranking used for manual semantic eligibility review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "paper-search" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from postprocess import dedup, rank  # type: ignore[import-not-found]  # noqa: E402
from search_papers import search_papers  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "reports" / "pb8_publication_benchmark"
QUERIES = [
    "industrial visual anomaly detection acquisition shift normal-only adaptation",
    "industrial anomaly detection test-time threshold calibration",
    "few-normal anomaly threshold calibration",
    "source-free operating-point adaptation anomaly detection",
    "safe selective test-time adaptation anomaly detection",
]
SOURCES = [
    "semantic_scholar",
    "open_alex",
    "arxiv",
    "openreview",
    "crossref",
    "dblp",
]


def main() -> None:
    results = search_papers(
        query=QUERIES,
        start_year=2018,
        end_year=2026,
        max_results=10,
        sources=SOURCES,
        parallel=True,
    )
    merged = dedup(results)
    ranked, n_dropped = rank(merged, QUERIES, min_score=None)
    payload = {
        "schema_version": "1.0",
        "queries": QUERIES,
        "year_range": [2018, 2026],
        "max_results_per_query_per_source": 10,
        "sources": SOURCES,
        "per_source_hits": {source: len(papers) for source, papers in results.items()},
        "n_unique": len(merged),
        "n_duplicates_merged": sum(len(papers) for papers in results.values()) - len(merged),
        "n_lexically_dropped": n_dropped,
        "raw_by_source": results,
        "ranked_unique": ranked,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "A_LITERATURE_SEARCH_RAW.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"saved {len(merged)} unique papers from "
        f"{sum(len(papers) for papers in results.values())} source records to {path}"
    )


if __name__ == "__main__":
    main()
