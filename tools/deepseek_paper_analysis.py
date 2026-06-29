#!/usr/bin/env python3
"""
Analyze conference paper abstracts with DeepSeek in parallel.

The script reads the original *_2026.md conference files, asks DeepSeek to
understand each title+abstract, and writes structured JSONL plus Markdown/CSV
summaries. It is resumable: completed paper IDs are skipped on later runs.

Required environment variable:
  DEEPSEEK_API_KEY

Useful optional environment variables:
  DEEPSEEK_MODEL=deepseek-v4-flash
  DEEPSEEK_BASE_URL=https://api.deepseek.com
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

HOTSPOT_CATEGORIES = [
    "LLM agents / tool use / planning",
    "Long-horizon interaction / long-context control",
    "Memory / state modeling / context management",
    "Reasoning / math / logic",
    "RAG / retrieval / knowledge grounding",
    "Evaluation / benchmarks / diagnostics",
    "Safety / alignment / robustness / hallucination",
    "Training / fine-tuning / alignment optimization",
    "Efficiency / compression / inference optimization",
    "Multimodal LLM / VLM / MLLM",
    "Code / software engineering",
    "Domain applications",
    "Other LLM-related work",
    "Not LLM-related",
]


RELATEDNESS_LEVELS = [
    "directly worth reading",
    "probably worth reading",
    "background only",
    "not relevant",
]


IDEA_FALLBACK = """Describe your research idea here.

Include the problem, target domain, expected method direction, evaluation needs,
and what kinds of papers would be useful as related work, baselines, benchmarks,
or method inspiration.
"""


ENTRY_RE = re.compile(
    r"^##\s+(\d+)\.\s+(.+?)\r?\n"
    r"Authors:\s*(.*?)\r?\n"
    r"URL:\s*(.*?)\r?\n\r?\n"
    r"Abstract:\r?\n"
    r"(.*?)(?=\r?\n##\s+\d+\.|\Z)",
    re.S | re.M,
)


@dataclass(frozen=True)
class Paper:
    conference: str
    idx: int
    title: str
    authors: str
    url: str
    abstract: str

    @property
    def paper_id(self) -> str:
        return f"{self.conference}:{self.idx}"


def parse_papers(path: Path) -> list[Paper]:
    text = path.read_text(encoding="utf-8", errors="replace")
    conference = path.stem.replace("_2026", "")
    papers: list[Paper] = []
    for match in ENTRY_RE.finditer(text):
        papers.append(
            Paper(
                conference=conference,
                idx=int(match.group(1)),
                title=match.group(2).strip(),
                authors=match.group(3).strip(),
                url=match.group(4).strip(),
                abstract=re.sub(r"\s+", " ", match.group(5)).strip(),
            )
        )
    return papers


def load_idea(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return IDEA_FALLBACK


def build_prompt(idea: str, paper: Paper) -> list[dict[str, str]]:
    system = (
        "You are a careful research assistant doing citation triage. "
        "Read the paper title and abstract semantically, not by keyword matching. "
        "Return only valid JSON, with no markdown."
    )
    user = {
        "research_idea": idea,
        "paper": {
            "conference": paper.conference,
            "index": paper.idx,
            "title": paper.title,
            "abstract": paper.abstract,
        },
        "task": (
            "Judge whether this paper is LLM-related, classify its research "
            "hotspot, and decide whether it is worth deep reading for the "
            "research idea. Be strict: a paper is worth deep reading only if it "
            "could inform the problem framing, method, benchmark, baseline, "
            "or positioning of the supplied research idea."
        ),
        "allowed_hotspot_categories": HOTSPOT_CATEGORIES,
        "allowed_relatedness_levels": RELATEDNESS_LEVELS,
        "required_json_schema": {
            "is_llm_related": "boolean",
            "llm_related_confidence": "number from 0 to 1",
            "primary_hotspot": "one allowed hotspot category",
            "secondary_hotspots": ["zero or more allowed hotspot categories"],
            "hotspot_rationale": "one concise sentence",
            "idea_relatedness": "one allowed relatedness level",
            "idea_score": "integer from 0 to 100",
            "relationship_types": [
                "agent planning",
                "long-horizon interaction",
                "memory or state modeling",
                "semantic drift or consistency",
                "feedback or self-correction",
                "RAG or knowledge grounding",
                "evaluation benchmark",
                "safety or robustness",
                "inference-time control",
                "theoretical framework",
                "not applicable",
            ],
            "why_relevant_or_not": "two concise sentences",
            "suggested_use": (
                "one of: direct related work, baseline, benchmark, method "
                "inspiration, motivation/background, ignore"
            ),
            "one_line_summary": "one sentence summary of the paper",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def post_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def analyze_one(
    paper: Paper,
    idea: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    messages = build_prompt(idea, paper)
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = post_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
            )
            content = response["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            usage = response.get("usage", {})
            return {
                "paper_id": paper.paper_id,
                "conference": paper.conference,
                "idx": paper.idx,
                "title": paper.title,
                "authors": paper.authors,
                "url": paper.url,
                "abstract": paper.abstract,
                "model": model,
                "analysis": parsed,
                "usage": usage,
                "ok": True,
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = repr(exc)
            sleep_s = min(60, (2**attempt) + random.random())
            if attempt < retries:
                time.sleep(sleep_s)
    return {
        "paper_id": paper.paper_id,
        "conference": paper.conference,
        "idx": paper.idx,
        "title": paper.title,
        "authors": paper.authors,
        "url": paper.url,
        "abstract": paper.abstract,
        "model": model,
        "analysis": {},
        "usage": {},
        "ok": False,
        "error": last_error,
    }


def read_completed(jsonl_path: Path) -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("ok") and item.get("paper_id"):
                completed.add(item["paper_id"])
    return completed


def append_writer(path: Path, output_queue: "queue.Queue[dict[str, Any] | None]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        while True:
            item = output_queue.get()
            if item is None:
                output_queue.task_done()
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            output_queue.task_done()


def load_results(jsonl_path: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    rows_without_id = []
    if not jsonl_path.exists():
        return []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                paper_id = row.get("paper_id")
                if paper_id:
                    # Keep the latest successful judgment for resumable runs
                    # that may have been interrupted after writing duplicates.
                    if row.get("ok") or paper_id not in by_id:
                        by_id[paper_id] = row
                else:
                    rows_without_id.append(row)
    return list(by_id.values()) + rows_without_id


def analysis_value(row: dict[str, Any], key: str, default: Any = "") -> Any:
    analysis = row.get("analysis") or {}
    return analysis.get(key, default)


def write_reports(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in results if r.get("ok")]

    by_conf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_conf[row["conference"]].append(row)

    summary_path = out_dir / "conference_hotspot_summary.md"
    summary_csv_path = out_dir / "conference_hotspot_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conference",
                "analyzed_papers",
                "llm_related_papers",
                "llm_share",
                "directly_worth_reading",
                "probably_worth_reading",
                "top_hotspots",
            ],
        )
        writer.writeheader()
        for conf, rows in sorted(by_conf.items()):
            llm_count = sum(bool(analysis_value(r, "is_llm_related", False)) for r in rows)
            direct = sum(analysis_value(r, "idea_relatedness") == "directly worth reading" for r in rows)
            probable = sum(analysis_value(r, "idea_relatedness") == "probably worth reading" for r in rows)
            hotspots = Counter(analysis_value(r, "primary_hotspot", "Unknown") for r in rows)
            writer.writerow(
                {
                    "conference": conf,
                    "analyzed_papers": len(rows),
                    "llm_related_papers": llm_count,
                    "llm_share": f"{(llm_count / len(rows) * 100) if rows else 0:.1f}%",
                    "directly_worth_reading": direct,
                    "probably_worth_reading": probable,
                    "top_hotspots": "; ".join(f"{k}: {v}" for k, v in hotspots.most_common(8)),
                }
            )

    lines = [
        "# Conference Hotspot Summary",
        "",
        "This report is generated from DeepSeek JSON judgments over paper titles and abstracts.",
        "",
        "| Conference | Analyzed | LLM-related | LLM Share | Direct Reads | Probable Reads | Top Hotspots |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for conf, rows in sorted(by_conf.items()):
        llm_count = sum(bool(analysis_value(r, "is_llm_related", False)) for r in rows)
        direct = sum(analysis_value(r, "idea_relatedness") == "directly worth reading" for r in rows)
        probable = sum(analysis_value(r, "idea_relatedness") == "probably worth reading" for r in rows)
        hotspots = Counter(analysis_value(r, "primary_hotspot", "Unknown") for r in rows)
        top_hotspots = "; ".join(f"{k}: {v}" for k, v in hotspots.most_common(5))
        lines.append(
            f"| {conf} | {len(rows)} | {llm_count} | {(llm_count / len(rows) * 100) if rows else 0:.1f}% | "
            f"{direct} | {probable} | {top_hotspots} |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    related_rows = [
        r
        for r in ok_rows
        if analysis_value(r, "idea_relatedness")
        in {"directly worth reading", "probably worth reading"}
    ]
    related_rows.sort(
        key=lambda r: (
            r["conference"],
            -int(analysis_value(r, "idea_score", 0) or 0),
            r["idx"],
        )
    )

    related_csv = out_dir / "idea_deep_read_papers.csv"
    with related_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conference",
                "idx",
                "score",
                "relatedness",
                "title",
                "primary_hotspot",
                "relationship_types",
                "suggested_use",
                "why_relevant_or_not",
                "url",
                "authors",
                "abstract",
            ],
        )
        writer.writeheader()
        for row in related_rows:
            writer.writerow(
                {
                    "conference": row["conference"],
                    "idx": row["idx"],
                    "score": analysis_value(row, "idea_score", 0),
                    "relatedness": analysis_value(row, "idea_relatedness"),
                    "title": row["title"],
                    "primary_hotspot": analysis_value(row, "primary_hotspot"),
                    "relationship_types": "; ".join(analysis_value(row, "relationship_types", [])),
                    "suggested_use": analysis_value(row, "suggested_use"),
                    "why_relevant_or_not": analysis_value(row, "why_relevant_or_not"),
                    "url": row["url"],
                    "authors": row["authors"],
                    "abstract": row["abstract"],
                }
            )

    related_md = out_dir / "idea_deep_read_papers.md"
    lines = [
        "# Papers Worth Deep Reading for the Research Idea",
        "",
        "These papers were selected by DeepSeek from title and abstract understanding.",
        "",
    ]
    for conf in sorted({r["conference"] for r in related_rows}):
        conf_rows = [r for r in related_rows if r["conference"] == conf]
        lines.extend([f"## {conf}", ""])
        for row in conf_rows:
            score = analysis_value(row, "idea_score", 0)
            relatedness = analysis_value(row, "idea_relatedness")
            lines.extend(
                [
                    f"### {row['idx']}. {row['title']}",
                    "",
                    f"- Score: {score}",
                    f"- Relatedness: {relatedness}",
                    f"- Primary hotspot: {analysis_value(row, 'primary_hotspot')}",
                    f"- Relationship types: {', '.join(analysis_value(row, 'relationship_types', []))}",
                    f"- Suggested use: {analysis_value(row, 'suggested_use')}",
                    f"- URL: {row['url']}",
                    f"- Authors: {row['authors']}",
                    "",
                    f"Why relevant: {analysis_value(row, 'why_relevant_or_not')}",
                    "",
                    f"Summary: {analysis_value(row, 'one_line_summary')}",
                    "",
                ]
            )
    related_md.write_text("\n".join(lines), encoding="utf-8")

    # Per-conference detailed CSVs.
    for conf, rows in sorted(by_conf.items()):
        path = out_dir / f"{conf}_llm_judgments.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "idx",
                    "title",
                    "is_llm_related",
                    "llm_related_confidence",
                    "primary_hotspot",
                    "secondary_hotspots",
                    "idea_relatedness",
                    "idea_score",
                    "relationship_types",
                    "suggested_use",
                    "why_relevant_or_not",
                    "url",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "idx": row["idx"],
                        "title": row["title"],
                        "is_llm_related": analysis_value(row, "is_llm_related"),
                        "llm_related_confidence": analysis_value(row, "llm_related_confidence"),
                        "primary_hotspot": analysis_value(row, "primary_hotspot"),
                        "secondary_hotspots": "; ".join(analysis_value(row, "secondary_hotspots", [])),
                        "idea_relatedness": analysis_value(row, "idea_relatedness"),
                        "idea_score": analysis_value(row, "idea_score"),
                        "relationship_types": "; ".join(analysis_value(row, "relationship_types", [])),
                        "suggested_use": analysis_value(row, "suggested_use"),
                        "why_relevant_or_not": analysis_value(row, "why_relevant_or_not"),
                        "url": row["url"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=Path("papers_2026"))
    parser.add_argument("--idea-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("llm_deepseek_analysis"))
    parser.add_argument("--conferences", default="all", help="Comma-separated names, e.g. ACL,CVPR,ICLR")
    parser.add_argument("--limit", type=int, default=0, help="Limit papers per conference for a dry run. 0 means no limit.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-api", action="store_true", help="Only rebuild reports from existing JSONL results.")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "paper_llm_judgments.jsonl"

    if args.skip_api:
        write_reports(load_results(jsonl_path), args.out_dir)
        return 0

    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY is not set. Set it in your shell environment and rerun.", file=sys.stderr)
        return 2

    idea = load_idea(args.idea_file)
    requested = None
    if args.conferences.lower() != "all":
        requested = {item.strip().upper() for item in args.conferences.split(",") if item.strip()}

    all_papers: list[Paper] = []
    for path in sorted(args.papers_dir.glob("*_2026.md")):
        conf = path.stem.replace("_2026", "").upper()
        if requested and conf not in requested:
            continue
        papers = parse_papers(path)
        if args.limit:
            papers = papers[: args.limit]
        all_papers.extend(papers)

    completed = read_completed(jsonl_path)
    remaining = [paper for paper in all_papers if paper.paper_id not in completed]
    print(f"Loaded {len(all_papers)} papers; {len(completed)} already complete; {len(remaining)} remaining.")

    output_queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    writer = threading.Thread(target=append_writer, args=(jsonl_path, output_queue), daemon=True)
    writer.start()

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_one, paper, idea, api_key, base_url, model, args.timeout, args.retries): paper
            for paper in remaining
        }
        for future in as_completed(futures):
            item = future.result()
            output_queue.put(item)
            done += 1
            if done % 25 == 0 or done == len(remaining):
                print(f"Processed {done}/{len(remaining)} in this run.")

    output_queue.put(None)
    output_queue.join()
    writer.join()

    write_reports(load_results(jsonl_path), args.out_dir)
    print(f"Wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
