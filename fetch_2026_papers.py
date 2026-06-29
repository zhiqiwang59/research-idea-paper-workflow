import concurrent.futures as futures
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests


OUT_DIR = Path("papers_2026")
CACHE_DIR = OUT_DIR / ".cache"
USER_AGENT = "Mozilla/5.0 (compatible; paper-metadata-fetcher/1.0)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 40


def clean_text(value):
    if value is None:
        return ""
    value = re.sub(r"<span[^>]*class=['\"]?acl-fixed-case['\"]?[^>]*>(.*?)</span>", r"\1", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value, flags=re.S)
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def get(url, **kwargs):
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
            response.raise_for_status()
            return response
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def content_value(content, key):
    value = content.get(key, "")
    if isinstance(value, dict):
        value = value.get("value", "")
    return value


def write_markdown(conference, source_urls, papers, status=None):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{conference}_2026.md"
    lines = [
        f"# {conference} 2026 Papers",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Count: {len(papers)}",
        "",
        "Sources:",
    ]
    for url in source_urls:
        lines.append(f"- {url}")
    if status:
        lines += ["", f"Status: {status}"]
    lines.append("")

    for index, paper in enumerate(papers, 1):
        lines.append(f"## {index}. {paper['title']}")
        if paper.get("authors"):
            lines.append(f"Authors: {paper['authors']}")
        if paper.get("url"):
            lines.append(f"URL: {paper['url']}")
        lines.append("")
        lines.append("Abstract:")
        lines.append(paper.get("abstract") or "N/A")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_cache(name):
    path = CACHE_DIR / f"{name}.jsonl"
    cached = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cached[item["url"]] = item
        except Exception:
            continue
    return cached


def append_cache(name, item):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_openreview(conference, venueid):
    papers = []
    offset = 0
    limit = 1000
    while True:
        params = {
            "content.venueid": venueid,
            "limit": limit,
            "offset": offset,
        }
        data = get("https://api2.openreview.net/notes", params=params).json()
        notes = data.get("notes", [])
        if not notes:
            break
        for note in notes:
            content = note.get("content", {})
            title = clean_text(content_value(content, "title"))
            abstract = clean_text(content_value(content, "abstract"))
            authors = content_value(content, "authors")
            if isinstance(authors, list):
                authors = ", ".join(authors)
            papers.append(
                {
                    "title": title,
                    "authors": clean_text(authors),
                    "abstract": abstract,
                    "url": f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}",
                }
            )
        if len(notes) < limit:
            break
        offset += limit
    return write_markdown(conference, [f"https://openreview.net/group?id={venueid}"], papers)


def fetch_acl():
    url = "https://aclanthology.org/events/acl-2026/"
    text = get(url).text
    blocks = re.findall(r'<div class="d-sm-flex align-items-stretch mb-3">(.*?)(?=<div class="d-sm-flex align-items-stretch mb-3">|</main>)', text, re.S)
    papers = []
    for block in blocks:
        m = re.search(r"<strong><a[^>]+href=(['\"]?)(/2026\.acl-[^/]+/)\1[^>]*>(.*?)</a></strong>", block, re.S)
        if not m:
            continue
        title = clean_text(m.group(3))
        if title.lower().startswith("proceedings of"):
            continue
        abs_match = re.search(r'<div class="card-body p-3 small">(.*?)</div>', block, re.S)
        authors_area = block.split("</strong><br>", 1)[1].split("</span>", 1)[0] if "</strong><br>" in block else ""
        authors = clean_text(authors_area.replace("|", ", "))
        papers.append(
            {
                "title": title,
                "authors": authors,
                "abstract": clean_text(abs_match.group(1)) if abs_match else "",
                "url": urljoin(url, m.group(2)),
            }
        )
    return write_markdown("ACL", [url], papers)


def cvpr_abstract(item):
    title, path = item
    url = urljoin("https://openaccess.thecvf.com/", path)
    page = get(url).text
    abs_match = re.search(r'<div id="abstract">\s*(.*?)\s*</div>', page, re.S)
    authors_match = re.search(r'<div id="authors">\s*<br><b><i>(.*?)</i></b>;', page, re.S)
    return {
        "title": title,
        "authors": clean_text(authors_match.group(1)) if authors_match else "",
        "abstract": clean_text(abs_match.group(1)) if abs_match else "",
        "url": url,
    }


def fetch_cvpr():
    url = "https://openaccess.thecvf.com/CVPR2026?day=all"
    text = get(url).text
    items = [
        (clean_text(title), href)
        for href, title in re.findall(r'<dt class="ptitle"><br><a href="([^"]+)">(.*?)</a></dt>', text, re.S)
    ]
    cache = load_cache("cvpr")
    papers_by_url = dict(cache)
    remaining = [item for item in items if urljoin("https://openaccess.thecvf.com/", item[1]) not in papers_by_url]
    print(f"CVPR: {len(items)} papers, {len(remaining)} remaining", flush=True)
    done = 0
    with futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_map = {executor.submit(cvpr_abstract, item): item for item in remaining}
        for future in futures.as_completed(future_map):
            paper = future.result()
            papers_by_url[paper["url"]] = paper
            append_cache("cvpr", paper)
            done += 1
            if done % 100 == 0:
                print(f"CVPR: fetched {done}/{len(remaining)}", flush=True)
    papers = [papers_by_url[urljoin("https://openaccess.thecvf.com/", href)] for _, href in items]
    return write_markdown("CVPR", [url], papers)


def aaai_article(item):
    title, article_url, authors = item
    page = get(article_url).text
    abs_match = re.search(r'<section class="item abstract">\s*<h2 class="label">Abstract</h2>\s*(.*?)\s*</section>', page, re.S)
    return {
        "title": title,
        "authors": authors,
        "abstract": clean_text(abs_match.group(1)) if abs_match else "",
        "url": article_url,
    }


def fetch_aaai():
    archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    archive = get(archive_url).text
    issue_urls = []
    for issue_url, title in re.findall(r'<a[^>]+href="([^"]*/issue/view/[^"]+)"[^>]*>(.*?)</a>', archive, re.S):
        label = clean_text(title)
        if label.startswith("AAAI-26 Technical Tracks"):
            issue_urls.append(issue_url)

    items = []
    for issue_url in issue_urls:
        issue = get(issue_url).text
        blocks = re.findall(r'<div class="obj_article_summary">(.*?)</div>\s*</li>', issue, re.S)
        for block in blocks:
            m = re.search(r'<h3 class="title">\s*<a[^>]+href="([^"]+)">\s*(.*?)\s*</a>\s*</h3>', block, re.S)
            if not m:
                continue
            authors_match = re.search(r'<div class="authors">\s*(.*?)\s*</div>', block, re.S)
            items.append((clean_text(m.group(2)), m.group(1), clean_text(authors_match.group(1)) if authors_match else ""))

    cache = load_cache("aaai")
    papers_by_url = dict(cache)
    remaining = [item for item in items if item[1] not in papers_by_url]
    print(f"AAAI: {len(items)} papers, {len(remaining)} remaining", flush=True)
    done = 0
    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        future_map = {executor.submit(aaai_article, item): item for item in remaining}
        for future in futures.as_completed(future_map):
            paper = future.result()
            papers_by_url[paper["url"]] = paper
            append_cache("aaai", paper)
            done += 1
            if done % 100 == 0:
                print(f"AAAI: fetched {done}/{len(remaining)}", flush=True)
    papers = [papers_by_url[item[1]] for item in items]
    return write_markdown("AAAI", [archive_url] + issue_urls, papers)


def write_unavailable_files():
    write_markdown(
        "NeurIPS",
        ["https://openreview.net/group?id=NeurIPS.cc/2026/Conference"],
        [],
        "No public NeurIPS 2026 accepted-paper records were available from OpenReview at generation time.",
    )
    write_markdown(
        "ICCV",
        ["https://openaccess.thecvf.com/"],
        [],
        "ICCV is not a 2026 conference year; no ICCV 2026 proceedings were available at generation time.",
    )


def main():
    targets = {arg.upper() for arg in sys.argv[1:]} or {"AAAI", "ICLR", "ICML", "ACL", "CVPR", "NEURIPS", "ICCV"}
    outputs = []
    if "AAAI" in targets:
        outputs.append(fetch_aaai())
    if "ICLR" in targets:
        outputs.append(fetch_openreview("ICLR", "ICLR.cc/2026/Conference"))
    if "ICML" in targets:
        outputs.append(fetch_openreview("ICML", "ICML.cc/2026/Conference"))
    if "ACL" in targets:
        outputs.append(fetch_acl())
    if "CVPR" in targets:
        outputs.append(fetch_cvpr())
    if "NEURIPS" in targets or "ICCV" in targets:
        if "NEURIPS" in targets:
            write_markdown(
                "NeurIPS",
                ["https://openreview.net/group?id=NeurIPS.cc/2026/Conference"],
                [],
                "No public NeurIPS 2026 accepted-paper records were available from OpenReview at generation time.",
            )
            outputs.append(OUT_DIR / "NeurIPS_2026.md")
        if "ICCV" in targets:
            write_markdown(
                "ICCV",
                ["https://openaccess.thecvf.com/"],
                [],
                "ICCV is not a 2026 conference year; no ICCV 2026 proceedings were available at generation time.",
            )
            outputs.append(OUT_DIR / "ICCV_2026.md")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": [str(path) for path in outputs],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
