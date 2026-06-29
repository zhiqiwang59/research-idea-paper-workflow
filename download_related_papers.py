import concurrent.futures as futures
import csv
import html
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests


BASE_DIR = Path("papers_2026")
INPUT_CSV = BASE_DIR / "ALL_2026_Semantic_Bridge_related_papers.csv"
DOWNLOAD_DIR = BASE_DIR / "downloaded_papers"
MANIFEST_CSV = DOWNLOAD_DIR / "download_manifest.csv"
FAILED_CSV = DOWNLOAD_DIR / "download_failed.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; related-paper-downloader/1.0)"}
TIMEOUT = 60
MAX_WORKERS = 1


def request_get(url, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else 30 * (attempt + 1)
                time.sleep(delay)
                response.raise_for_status()
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def clean_filename(value, max_len=140):
    value = html.unescape(value or "")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:max_len].rstrip(" .") or "paper"


def openreview_pdf_url(page_url):
    paper_id = parse_qs(urlparse(page_url).query).get("id", [""])[0]
    if not paper_id:
        raise ValueError(f"OpenReview URL has no id: {page_url}")
    return f"https://openreview.net/pdf?id={paper_id}"


def page_pdf_url(page_url, conference):
    conference = conference.upper()
    if conference in {"ICLR", "ICML"} or "openreview.net/forum" in page_url:
        return openreview_pdf_url(page_url)

    page = request_get(page_url).text

    if conference == "AAAI":
        match = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', page, re.I)
        if match:
            return html.unescape(match.group(1))
        match = re.search(r'<a[^>]+class=["\'][^"\']*\bpdf\b[^"\']*["\'][^>]+href=["\']([^"\']+)["\']', page, re.I)
        if match:
            return html.unescape(match.group(1)).replace("/article/view/", "/article/download/")

    if conference == "ACL":
        match = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', page, re.I)
        if match:
            return html.unescape(match.group(1))
        return page_url.rstrip("/") + ".pdf"

    if conference == "CVPR":
        links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', page, re.I)
        paper_links = [urljoin(page_url, link) for link in links if "supplemental" not in link.lower()]
        if paper_links:
            return paper_links[0]

    match = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']', page, re.I)
    if match:
        return html.unescape(match.group(1))
    raise ValueError(f"Could not find PDF link for {page_url}")


def is_pdf_response(response):
    content_type = response.headers.get("content-type", "").lower()
    return response.content.startswith(b"%PDF") or "application/pdf" in content_type


def download_one(row):
    conference = row["conference"]
    rank = int(row["rank"])
    title = row["title"]
    filename = f"{conference}_{rank:02d}_{clean_filename(title)}.pdf"
    output_path = DOWNLOAD_DIR / filename

    if output_path.exists() and output_path.stat().st_size > 1000:
        return {
            "status": "skipped",
            "conference": conference,
            "rank": rank,
            "title": title,
            "source_url": row["url"],
            "pdf_url": "",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "error": "",
        }

    pdf_url = page_pdf_url(row["url"], conference)
    if "openreview.net/pdf" in pdf_url:
        time.sleep(4)
    response = request_get(pdf_url, allow_redirects=True)
    if not is_pdf_response(response):
        raise ValueError(f"Downloaded content is not a PDF: {pdf_url} ({response.headers.get('content-type', '')})")

    output_path.write_bytes(response.content)
    return {
        "status": "downloaded",
        "conference": conference,
        "rank": rank,
        "title": title,
        "source_url": row["url"],
        "pdf_url": response.url,
        "path": str(output_path),
        "bytes": len(response.content),
        "error": "",
    }


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    successes = []
    failures = []
    done = 0
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(download_one, row): row for row in rows}
        for future in futures.as_completed(future_map):
            row = future_map[future]
            done += 1
            try:
                result = future.result()
                successes.append(result)
                print(f"[{done}/{len(rows)}] {result['status']}: {result['conference']} #{result['rank']} {result['title']}", flush=True)
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "conference": row["conference"],
                    "rank": row["rank"],
                    "title": row["title"],
                    "source_url": row["url"],
                    "pdf_url": "",
                    "path": "",
                    "bytes": 0,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[{done}/{len(rows)}] failed: {row['conference']} #{row['rank']} {row['title']} - {exc}", flush=True)

    fields = ["status", "conference", "rank", "title", "source_url", "pdf_url", "path", "bytes", "error"]
    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(successes, key=lambda item: (item["conference"], int(item["rank"]))))

    with FAILED_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(failures, key=lambda item: (item["conference"], int(item["rank"]))))

    print()
    print(f"Downloaded/skipped: {len(successes)}")
    print(f"Failed: {len(failures)}")
    print(f"Folder: {DOWNLOAD_DIR.resolve()}")
    print(f"Manifest: {MANIFEST_CSV.resolve()}")
    print(f"Failures: {FAILED_CSV.resolve()}")


if __name__ == "__main__":
    main()
