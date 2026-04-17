#!/usr/bin/env python3
"""
Resolve Markdown links of the form:
  [Label](https://arxiv.org/search/?query=...&searchtype=all)
into concrete arXiv IDs / abs URLs so you can batch-import into Zotero.

Outputs (by default under ./zotero_import_full_v2/):
  - identifiers_all.txt              arXiv identifiers (one per line)
  - abs_urls_all.txt                 arXiv abs URLs (one per line)
  - per_file/<md_basename>.txt       arXiv identifiers for each source file
  - report.csv                       resolution report with scores
  - unresolved.csv                   unresolved entries for manual handling
  - cache.json                       persistent cache to resume runs

Typical usage:
  python3 tools/zotero_resolve_arxiv_search_links.py \\
    docs/literature/qml_literature_complete_2007_2025.md docs/literature/qaoa_literature_and_notes.md
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import difflib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


MD_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((https?://arxiv\.org/search/\?query=[^)]+)\)",
    re.IGNORECASE,
)

ARXIV_ID_RE = re.compile(r"/abs/([^/?#]+)", re.IGNORECASE)
VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


def _norm(s: str) -> str:
    s = s.strip().lower()
    # collapse whitespace
    s = " ".join(s.split())
    # remove most punctuation for similarity matching
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(s.split())
    return s


def _label_to_title_and_surname(label: str) -> tuple[str, str | None]:
    # Common label format:
    #   "Author, 2022 - Paper Title"
    #   "Author & Coauthor, 2019 - Paper Title"
    # We'll treat the part after " - " as the title.
    title = label
    if " - " in label:
        title = label.split(" - ", 1)[1].strip()
    # surname heuristic: first token before comma
    surname = None
    left = label.split(" - ", 1)[0]
    if "," in left:
        surname_raw = left.split(",", 1)[0].strip()
        # If multiple authors are present ("Farhi & Goldstone", "A and B"), keep only the first surname.
        surname_raw = re.split(r"\s+|&|and", surname_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        surname = re.sub(r"[^A-Za-z\-']", "", surname_raw) or None
    return title, surname


def _url_query_text(search_url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(search_url)
        qs = urllib.parse.parse_qs(parsed.query)
        q = (qs.get("query", [""]) or [""])[0]
        q = q.replace("+", " ")
        q = urllib.parse.unquote(q)
        q = " ".join(q.split())
        return q
    except Exception:
        return ""


def _fetch(url: str, timeout_s: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "QML-Zotero-Resolver/1.0 (+local script)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read()


def _arxiv_api(search_query: str, max_results: int = 5) -> list[dict]:
    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(max_results),
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    data = _fetch(url)

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    out: list[dict] = []
    for entry in root.findall("a:entry", ns):
        id_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        title = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""
        arxiv_id = VERSION_SUFFIX_RE.sub("", arxiv_id)
        if not arxiv_id:
            continue
        out.append(
            {
                "id": arxiv_id,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "title": title,
                "published": published,
                "authors": authors,
            }
        )
    return out


def _best_match(candidates: list[dict], want_title: str) -> tuple[dict | None, float]:
    if not candidates:
        return None, 0.0
    want = _norm(want_title)
    best = None
    best_score = 0.0
    for c in candidates:
        got = _norm(c.get("title", ""))
        score = difflib.SequenceMatcher(None, want, got).ratio()
        if score > best_score:
            best_score = score
            best = c
    return best, best_score


def resolve_one(label: str, search_url: str, sleep_s: float, use_fallback: bool) -> dict:
    title, surname = _label_to_title_and_surname(label)
    query_text = _url_query_text(search_url)

    # Performance note:
    # For large lists (1000+), keep arXiv API calls minimal. We do:
    #   - 1 query based on the existing arxiv.org/search keywords
    #   - 1 fallback query based on exact title if needed
    tried: list[str] = []
    best_overall = None
    best_overall_score = 0.0
    best_overall_query = ""

    qt = query_text or title or label
    qt = re.sub(r"\\b(19\\d{2}|20\\d{2})\\b", "", qt).strip()
    qt = " ".join(qt.split())
    tokens = [t for t in re.split(r"\\s+", qt) if t][:14]
    q_primary = ""
    if tokens:
        q_primary = "all:" + " AND all:".join(tokens)
        if surname:
            q_primary = f"{q_primary} AND au:{surname}"

    # Avoid fallback on very short/ambiguous titles like "QAOA".
    title_ok_for_fallback = bool(title) and len(_norm(title)) >= 18 and len(_norm(title).split()) >= 3
    q_fallback = f'ti:"{title}"' if (use_fallback and title_ok_for_fallback) else ""

    for q in [q_primary, q_fallback]:
        if not q:
            continue
        tried.append(q)
        try:
            candidates = _arxiv_api(q, max_results=10)
        except Exception:
            candidates = []
        best, score = _best_match(candidates, title or query_text or label)
        if score > best_overall_score and best:
            best_overall = best
            best_overall_score = score
            best_overall_query = q
        # If we already have a strong hit, stop early
        if best_overall_score >= 0.92:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)

    return {
        "label": label,
        "search_url": search_url,
        "want_title": title,
        "surname": surname or "",
        "query_text": query_text,
        "best_id": (best_overall or {}).get("id", ""),
        "best_abs_url": (best_overall or {}).get("abs_url", ""),
        "best_title": (best_overall or {}).get("title", ""),
        "best_published": (best_overall or {}).get("published", ""),
        "best_score": round(best_overall_score, 4),
        "best_query": best_overall_query or "",
        "tried_queries": " | ".join(tried),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Markdown files to scan")
    ap.add_argument(
        "--outdir",
        default="zotero_import_full_v2",
        help="Output directory (relative to CWD); canonical full run uses zotero_import_full_v2",
    )
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API calls (politeness)")
    ap.add_argument("--min-score", type=float, default=0.78, help="Min similarity score to accept as resolved")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N links (0 = no limit)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers for API requests")
    ap.add_argument("--no-fallback", action="store_true", help="Disable title-based fallback query (faster)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    per_file_dir = os.path.join(outdir, "per_file")
    os.makedirs(per_file_dir, exist_ok=True)

    cache_path = os.path.join(outdir, "cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Collect links
    links: list[tuple[str, str, str]] = []  # (src_file, label, search_url)
    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        for m in MD_LINK_RE.finditer(text):
            label = m.group(1).strip()
            url = m.group(2).strip()
            links.append((p, label, url))

    if args.limit and args.limit > 0:
        links = links[: args.limit]

    # Resolve with caching (parallel for speed)
    resolved_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    per_file_ids: dict[str, list[str]] = {}
    all_ids: list[str] = []
    all_abs: list[str] = []

    # De-dup tasks but keep mapping back to source files (same link can appear in multiple files)
    tasks_by_key: dict[str, tuple[str, str]] = {}  # key -> (label, url)
    keys_in_order: list[str] = []
    sources_for_key: dict[str, list[str]] = {}  # key -> [source_file...]
    for (src, label, url) in links:
        key = f"{label}||{url}"
        if key not in tasks_by_key:
            tasks_by_key[key] = (label, url)
            keys_in_order.append(key)
        sources_for_key.setdefault(key, []).append(src)

    lock = threading.Lock()

    def work(key: str) -> tuple[str, dict]:
        label, url = tasks_by_key[key]
        row = cache.get(key)
        if not row:
            row = resolve_one(label=label, search_url=url, sleep_s=args.sleep, use_fallback=(not args.no_fallback))
        return key, row

    to_run = [k for k in keys_in_order if k not in cache]
    if to_run:
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            for key, row in ex.map(work, to_run):
                with lock:
                    cache[key] = row

    # Now build outputs per source file
    for key in keys_in_order:
        row = dict(cache[key])
        label, url = tasks_by_key[key]
        # For reporting, pick the first source as the "main" location (others are duplicated rows logically)
        for src in sources_for_key.get(key, []):
            row_with_src = dict(row)
            row_with_src["source_file"] = src
            row_with_src["label"] = label
            row_with_src["search_url"] = url

            is_ok = bool(row_with_src.get("best_id")) and float(row_with_src.get("best_score") or 0) >= args.min_score
            if is_ok:
                arxiv_id = row_with_src["best_id"]
                abs_url = row_with_src["best_abs_url"]
                all_ids.append(arxiv_id)
                all_abs.append(abs_url)
                per_file_ids.setdefault(os.path.basename(src), []).append(arxiv_id)
                resolved_rows.append(row_with_src)
            else:
                unresolved_rows.append(row_with_src)

    # Final cache write
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Deduplicate while preserving order
    def dedup(seq: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    all_ids = dedup(all_ids)
    all_abs = dedup(all_abs)

    with open(os.path.join(outdir, "identifiers_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_ids) + ("\n" if all_ids else ""))
    with open(os.path.join(outdir, "abs_urls_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_abs) + ("\n" if all_abs else ""))

    for base, ids in per_file_ids.items():
        ids = dedup(ids)
        out_path = os.path.join(per_file_dir, base.replace(".md", "") + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))

    report_path = os.path.join(outdir, "report.csv")
    fieldnames = [
        "source_file",
        "label",
        "search_url",
        "want_title",
        "surname",
        "query_text",
        "best_id",
        "best_abs_url",
        "best_title",
        "best_published",
        "best_score",
        "best_query",
        "tried_queries",
    ]
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in resolved_rows + unresolved_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    unresolved_path = os.path.join(outdir, "unresolved.csv")
    with open(unresolved_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in unresolved_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Scanned links: {len(links)}")
    print(f"Resolved (score>={args.min_score}): {len(resolved_rows)}")
    print(f"Unresolved: {len(unresolved_rows)}")
    print(f"Unique IDs written: {len(all_ids)}")
    print(f"Output dir: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

