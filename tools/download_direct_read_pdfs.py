#!/usr/bin/env python3
"""Download PDFs for idea-relevant direct-read papers."""

from __future__ import annotations

import csv
import html
import json
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, quote, urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "llm_deepseek_analysis" / "idea_deep_read_papers.csv"
OUT_DIR = ROOT / "llm_deepseek_analysis" / "direct_read_pdfs"
MANIFEST = OUT_DIR / "download_manifest.csv"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }
)


def safe_filename(text: str, max_len: int = 150) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(". ")
    return text[:max_len].rstrip(". ")


def pdf_url_for(row: dict[str, str]) -> tuple[str | None, str]:
    url = row["url"].strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "aclanthology.org" in host:
        clean = url.rstrip("/")
        return clean + ".pdf", "derived-acl"

    if "openreview.net" in host:
        paper_id = parse_qs(parsed.query).get("id", [""])[0]
        if paper_id:
            return f"https://openreview.net/pdf?id={quote(paper_id, safe='')}", "derived-openreview"
        return None, "missing-openreview-id"

    if "openaccess.thecvf.com" in host:
        if "/html/" in url and url.endswith(".html"):
            return url.replace("/html/", "/papers/").removesuffix(".html") + ".pdf", "derived-cvf"
        return None, "unsupported-cvf-url"

    if "ojs.aaai.org" in host:
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            page = resp.text
            meta = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', page, re.I)
            if meta:
                return html.unescape(meta.group(1)), "scraped-aaai-meta"
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', page, re.I)
            for href in hrefs:
                if "/article/download/" in href or "/article/view/" in href and "download" in href:
                    return urljoin(url, html.unescape(href)), "scraped-aaai-link"
        except Exception as exc:  # noqa: BLE001
            return None, f"aaai-scrape-error: {exc!r}"
        return None, "aaai-pdf-link-not-found"

    return None, f"unsupported-host:{host}"


def looks_like_pdf(content: bytes, content_type: str) -> bool:
    head = content[:8].lstrip()
    return head.startswith(b"%PDF") or "pdf" in content_type.lower()


def download_one(row: dict[str, str]) -> dict[str, str]:
    conf = row["conference"]
    idx = row["idx"]
    title = row["title"]
    filename = f"{conf}_{int(idx):04d}_{safe_filename(title)}.pdf"
    target = OUT_DIR / filename

    result = {
        "conference": conf,
        "idx": idx,
        "title": title,
        "source_url": row["url"],
        "pdf_url": "",
        "file": str(target),
        "status": "",
        "bytes": "0",
        "method": "",
        "error": "",
    }

    if target.exists() and target.stat().st_size > 1024:
        result["status"] = "exists"
        result["bytes"] = str(target.stat().st_size)
        return result

    pdf_url, method = pdf_url_for(row)
    result["method"] = method
    if not pdf_url:
        result["status"] = "failed"
        result["error"] = method
        return result
    result["pdf_url"] = pdf_url

    last_error = None
    for attempt in range(6):
        try:
            with SESSION.get(pdf_url, timeout=90, stream=True, allow_redirects=True) as resp:
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = int(retry_after) if retry_after and retry_after.isdigit() else min(180, 20 * (attempt + 1))
                    raise RuntimeError(f"rate limited; retry after {delay}s")
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                chunks = []
                total = 0
                for chunk in resp.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        chunks.append(chunk)
                        total += len(chunk)
                content = b"".join(chunks)
                if total < 1024:
                    raise RuntimeError(f"download too small: {total} bytes")
                if not looks_like_pdf(content, content_type):
                    raise RuntimeError(f"response does not look like PDF; Content-Type={content_type!r}")
                target.write_bytes(content)
                result["status"] = "downloaded"
                result["bytes"] = str(target.stat().st_size)
                return result
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if "rate limited" in last_error:
                match = re.search(r"retry after (\d+)s", last_error)
                delay = int(match.group(1)) if match else min(180, 20 * (attempt + 1))
            else:
                delay = 3 * (attempt + 1)
            time.sleep(delay)

    result["status"] = "failed"
    result["error"] = last_error or "unknown error"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds before each submitted request.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(INPUT_CSV.open(encoding="utf-8")))
    previous_results: dict[tuple[str | None, str | None], dict[str, str]] = {}
    if MANIFEST.exists():
        for row in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            previous_results[(row.get("conference"), row.get("idx"))] = row

    if args.only_failed and MANIFEST.exists():
        failed_keys = set()
        for row in previous_results.values():
            if row.get("status") == "failed":
                failed_keys.add((row.get("conference"), row.get("idx")))
        rows = [row for row in rows if (row.get("conference"), row.get("idx")) in failed_keys]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for row in rows:
            futures.append(executor.submit(download_one, row))
            if args.delay:
                time.sleep(args.delay)
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"{i}/{len(rows)} {result['status']}: {result['conference']} {result['idx']} {result['title'][:80]}")

    if args.only_failed:
        for result in results:
            previous_results[(result.get("conference"), result.get("idx"))] = result
        results = list(previous_results.values())

    results.sort(key=lambda r: (r["conference"], int(r["idx"])))
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conference",
                "idx",
                "title",
                "status",
                "bytes",
                "method",
                "source_url",
                "pdf_url",
                "file",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "total": len(results),
        "downloaded_or_existing": sum(r["status"] in {"downloaded", "exists"} for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "manifest": str(MANIFEST),
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
