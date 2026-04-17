#!/usr/bin/env python3
"""
Fetch recent arXiv preprints relevant to quantum computing × (quantum chemistry,
molecular simulation, materials, QML) and write a Markdown digest.

Uses the official arXiv API (https://arxiv.org/help/api) via the lightweight
`arxiv` PyPI package. Respect arXiv rate limits (built into Client).

Examples:
  python tools/arxiv_qchem_digest.py --days 7 --combined -o docs/arxiv_digest.md
  python tools/arxiv_qchem_digest.py --since 2026-01-01 --until 2026-12-31 --max-per-query 80 --combined
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Default search templates (edit to match your roadmap)
# Each entry: (section_title, arxiv_query_suffix_without_date)
# Date is appended as: AND submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
# ---------------------------------------------------------------------------
# Single OR expression: one API round-trip (friendlier to arXiv rate limits).
COMBINED_QUERY_BASE = (
    "("
    "cat:physics.chem-ph OR "
    "(cat:quant-ph AND ("
    'all:molecular OR all:chemistry OR all:"quantum chemistry" OR '
    "all:VQE OR all:Hamiltonian OR all:electronic OR all:wavefunction OR "
    'all:"machine learning" OR all:"neural quantum" OR all:DFT OR all:MD OR '
    "all:materials OR all:drug OR all:protein OR all:catalyst"
    ")) OR "
    "(cat:cond-mat.mtrl-sci AND ("
    "all:quantum OR all:VQE OR all:\"machine learning\" OR all:Hamiltonian"
    "))"
    ")"
)


def _section_for_primary_category(primary: str) -> str:
    if primary == "physics.chem-ph":
        return "physics.chem-ph (category)"
    if primary == "quant-ph":
        return "quant-ph ∩ {molecular, chemistry, VQE, …}"
    if primary.startswith("cond-mat"):
        return "cond-mat.mtrl-sci ∩ quantum / ML"
    return f"other (`{primary}`)"


DEFAULT_QUERY_BLOCKS: list[tuple[str, str]] = [
    (
        "physics.chem-ph (category)",
        "cat:physics.chem-ph",
    ),
    (
        "quant-ph ∩ {molecular, chemistry, VQE, …}",
        "cat:quant-ph AND ("
        'all:molecular OR all:chemistry OR all:"quantum chemistry" OR '
        "all:VQE OR all:Hamiltonian OR all:electronic OR all:wavefunction OR "
        'all:"machine learning" OR all:"neural quantum" OR all:DFT OR all:MD OR '
        "all:materials OR all:drug OR all:protein OR all:catalyst"
        ")",
    ),
    (
        "cond-mat.mtrl-sci ∩ quantum / ML",
        "cat:cond-mat.mtrl-sci AND ("
        "all:quantum OR all:VQE OR all:\"machine learning\" OR all:Hamiltonian"
        ")",
    ),
]


def fetch_results_combined(
    start: dt.date,
    end: dt.date,
    max_results: int,
    delay_seconds: float,
) -> list:
    """Single-query fetch; labels rows using primary_category."""
    import arxiv

    client = arxiv.Client(
        page_size=min(100, max_results),
        delay_seconds=delay_seconds,
        num_retries=5,
    )
    q = _build_query(COMBINED_QUERY_BASE, start, end)
    search = arxiv.Search(
        query=q,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    batch = _iter_results_with_retry(client, search)
    return [(_section_for_primary_category(r.primary_category), r) for r in batch]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_day(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _date_to_arxiv_range(
    start: dt.date, end: dt.date
) -> tuple[str, str]:
    """arXiv API uses UTC timestamps YYYYMMDDHHMM inclusive."""
    a = f"{start.strftime('%Y%m%d')}0000"
    b = f"{end.strftime('%Y%m%d')}2359"
    return a, b


def _build_query(base: str, start: dt.date, end: dt.date) -> str:
    a, b = _date_to_arxiv_range(start, end)
    return f"{base} AND submittedDate:[{a} TO {b}]"


def _short_id(entry_id: str) -> str:
    # http://arxiv.org/abs/1234.56789v2 -> 1234.56789
    if "/abs/" in entry_id:
        tail = entry_id.split("/abs/")[-1]
        return tail.split("v")[0]
    return entry_id


def _iter_results_with_retry(
    client: object,
    search: object,
    *,
    max_retries: int = 6,
) -> list:
    """Drain client.results(search) with sleeps on HTTP 429 (arXiv rate limit)."""
    import arxiv

    for attempt in range(max_retries):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as exc:
            if getattr(exc, "status", None) == 429 and attempt + 1 < max_retries:
                wait = 20 * (2**attempt)
                print(
                    f"arXiv rate limit (429), sleeping {wait}s before retry "
                    f"({attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise


def fetch_results(
    query_blocks: list[tuple[str, str]],
    start: dt.date,
    end: dt.date,
    max_per_query: int,
    delay_seconds: float,
    pause_between_queries: float,
) -> list:
    import arxiv

    client = arxiv.Client(
        page_size=min(100, max_per_query),
        delay_seconds=delay_seconds,
        num_retries=5,
    )
    all_results = []
    for i, (section, base) in enumerate(query_blocks):
        if i > 0:
            time.sleep(pause_between_queries)
        q = _build_query(base, start, end)
        search = arxiv.Search(
            query=q,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        batch = _iter_results_with_retry(client, search)
        for r in batch:
            all_results.append((section, r))
    # Dedupe by arxiv id, keep first occurrence (newest block order)
    by_id: dict[str, tuple[str, object]] = {}
    order: list[str] = []
    for section, r in all_results:
        sid = _short_id(r.entry_id)
        if sid not in by_id:
            order.append(sid)
            by_id[sid] = (section, r)
    return [by_id[i] for i in order]


def render_markdown(
    rows: list[tuple[str, object]],
    start: dt.date,
    end: dt.date,
    query_blocks: list[tuple[str, str]],
) -> str:
    import arxiv

    lines: list[str] = []
    lines.append("# arXiv digest: quantum × chemistry / materials / QML")
    lines.append("")
    lines.append(
        f"- **Window (UTC)**: `{start.isoformat()}` → `{end.isoformat()}`"
    )
    lines.append(
        f"- **Generated**: `{_utc_now().strftime('%Y-%m-%d %H:%M UTC')}`"
    )
    lines.append(
        "- **Source**: [arXiv API](https://arxiv.org/help/api) — verify claims in PDFs."
    )
    lines.append("")
    lines.append("## Query blocks")
    lines.append("")
    for title, base in query_blocks:
        a, b = _date_to_arxiv_range(start, end)
        full = _build_query(base, start, end)
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- `submittedDate:[{a} TO {b}]`")
        lines.append(f"- Full query: `{full}`")
        lines.append("")
    lines.append("## Papers")
    lines.append("")
    if not rows:
        lines.append("_No results in this window (try widening dates or queries)._")
        lines.append("")
        return "\n".join(lines)

    for section, r in rows:
        assert isinstance(r, arxiv.Result)
        aid = _short_id(r.entry_id)
        title = r.title.replace("\n", " ").strip()
        authors = ", ".join(a.name for a in r.authors[:8])
        if len(r.authors) > 8:
            authors += ", et al."
        pub = r.published.strftime("%Y-%m-%d")
        cats = ", ".join(r.categories)
        abs_url = f"https://arxiv.org/abs/{aid}"
        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        lines.append(f"### [{aid}]({abs_url}) — {title}")
        lines.append("")
        lines.append(f"- **Matched block**: {section}")
        lines.append(f"- **Authors**: {authors}")
        lines.append(f"- **Submitted**: {pub} (arXiv `published`)")
        lines.append(f"- **Categories**: `{cats}`")
        lines.append(f"- **PDF**: [{pdf_url}]({pdf_url})")
        lines.append("")
        summary = r.summary.replace("\n", " ").strip()
        if len(summary) > 1200:
            summary = summary[:1197] + "..."
        lines.append(f"> {summary}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a Markdown digest from arXiv (QChem/materials/QML)."
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="If set, end=today UTC and start=end-days (overrides --since/--until).",
    )
    p.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date inclusive (YYYY-MM-DD), UTC calendar day.",
    )
    p.add_argument(
        "--until",
        type=str,
        default=None,
        help="End date inclusive (YYYY-MM-DD). Default: today UTC.",
    )
    p.add_argument(
        "--max-per-query",
        type=int,
        default=100,
        help="Max results per query block (arXiv caps at ~30k but be polite).",
    )
    p.add_argument(
        "--delay-seconds",
        type=float,
        default=4.0,
        help="Client delay between paginated API calls (arXiv etiquette).",
    )
    p.add_argument(
        "--pause-between-queries",
        type=float,
        default=8.0,
        help="Extra sleep (seconds) between separate query blocks to avoid 429.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/arxiv_qchem_digest.md"),
        help="Output Markdown path.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Use one OR-query (fewer HTTP requests, less likely to hit 429). "
            "Section labels are inferred from each paper's primary arXiv category."
        ),
    )
    args = p.parse_args(argv)

    today = _utc_now().date()
    if args.days is not None:
        end_d = _parse_day(args.until) if args.until else today
        start_d = end_d - dt.timedelta(days=max(0, args.days))
    else:
        end_d = _parse_day(args.until) if args.until else today
        if args.since:
            start_d = _parse_day(args.since)
        else:
            start_d = end_d - dt.timedelta(days=14)

    if start_d > end_d:
        print("error: --since after --until", file=sys.stderr)
        return 2

    try:
        if args.combined:
            rows = fetch_results_combined(
                start_d,
                end_d,
                max_results=args.max_per_query,
                delay_seconds=args.delay_seconds,
            )
            blocks_for_md = [
                ("Single combined OR-query", COMBINED_QUERY_BASE),
            ]
        else:
            rows = fetch_results(
                DEFAULT_QUERY_BLOCKS,
                start_d,
                end_d,
                max_per_query=args.max_per_query,
                delay_seconds=args.delay_seconds,
                pause_between_queries=args.pause_between_queries,
            )
            blocks_for_md = list(DEFAULT_QUERY_BLOCKS)
    except ImportError:
        print(
            "Missing dependency: install with  pip install arxiv",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        import arxiv

        if isinstance(exc, arxiv.HTTPError) and exc.status == 429:
            print(
                "arXiv returned HTTP 429 (too many requests). "
                "Wait several minutes and retry, or increase "
                "--delay-seconds / --pause-between-queries.",
                file=sys.stderr,
            )
            return 1
        raise

    md = render_markdown(rows, start_d, end_d, blocks_for_md)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {len(rows)} papers to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
