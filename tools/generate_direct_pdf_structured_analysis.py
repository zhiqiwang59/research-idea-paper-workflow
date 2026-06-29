from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("llm_deepseek_analysis/direct_read_pdfs")
OUT = Path("llm_deepseek_analysis/direct_read_structured_analysis")
PACKS = OUT / "_reading_packs"
PROGRESS = OUT / "analysis_progress.jsonl"
INDEX = OUT / "INDEX.md"


PROMPT = """你是一名结构科研分析助手。目标不是总结论文，而是提炼论文的核心结构、数学方法、实验证据与研究价值，为后续科研复现和创新提供基础。

请严格使用固定四段式 Markdown：

# <论文标题>

## Problem
- 论文解决什么问题？
- 为什么现有方法不足？
- 核心假设（如有）。

## Method
- 按模块说明整体流程、每个模块作用、训练目标、优化方式。
- 保留关键数学公式：损失函数、概率模型、推导公式、核心算法。
- 如果流程不完整，可依据公开文献进行合理补全，并注明“合理推断”。
- 最后明确：与已有方法最大的结构区别；真正创新点。

## Experiment
- 数据集、数据规模（如果重要）。
- Baseline。
- 评价指标。
- 关键实验结果：是否 SOTA、提升多少、哪些模块最有效。
- 仅保留非常规训练超参数。

## Analysis
- 方法优势、局限、适用条件、核心结构假设、可能失败场景。
- Research Opportunity：真正创新在哪里、可与哪些方法结合、哪些模块可替换、哪些假设值得放松、哪些结构方向值得继续研究。

输出原则：
- 所有推断必须注明来源：原文 / 公开文献 / 合理推断。
- Method 保留全部关键公式；Experiment 只保留关键结果；Analysis 重在创新、本质假设、局限与未来研究机会。
- 不要写冗长背景，不要填充常规训练细节。
"""


@dataclass(frozen=True)
class Paper:
    idx: int
    filename: str
    path: Path

    @property
    def title(self) -> str:
        stem = self.path.stem
        stem = re.sub(r"^[A-Z]+_\d+_", "", stem)
        return stem.replace("_", ": ", 1).strip()


def list_papers() -> list[Paper]:
    return [
        Paper(idx=i, filename=p.name, path=p)
        for i, p in enumerate(sorted(ROOT.glob("*.pdf"), key=lambda x: x.name.lower()), 1)
    ]


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[:/\\?*\"<>|]+", "", path.stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    return stem[:180]


def run_pdftotext(path: Path) -> str:
    cp = subprocess.run(
        ["pdftotext", "-nopgbrk", "-enc", "UTF-8", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.decode("utf-8", errors="replace"))
    text = cp.stdout.decode("utf-8", errors="replace").replace("\x0c", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def strip_references(text: str) -> tuple[str, bool]:
    matches = list(re.finditer(r"(?im)^\s*(references|bibliography)\s*$", text))
    if not matches:
        return text, False
    cut = matches[-1].start()
    if cut < len(text) * 0.55:
        return text, False
    return text[:cut].strip(), True


def section_window(text: str, headings: list[str], width: int) -> str:
    chunks: list[str] = []
    for heading in headings:
        m = re.search(rf"(?im)^\s*(\d+\.?\s*)?{re.escape(heading)}\s*$", text)
        if m:
            chunks.append(text[m.start() : m.start() + width].strip())
    return "\n\n".join(chunks)


def make_pack(paper: Paper, max_chars: int) -> tuple[str, dict[str, object]]:
    raw = run_pdftotext(paper.path)
    body, refs_removed = strip_references(raw)
    abstract = section_window(body, ["Abstract"], 10_000) or body[:10_000]
    method = section_window(
        body,
        [
            "Method",
            "Methods",
            "Methodology",
            "Approach",
            "Framework",
            "Model",
            "Training",
        ],
        28_000,
    )
    experiment = section_window(
        body,
        ["Experiment", "Experiments", "Experimental Setup", "Evaluation", "Results"],
        24_000,
    )
    formula_lines = [
        line.strip()
        for line in body.splitlines()
        if re.search(r"(\\math|\\sum|\\arg|\\nabla|\\theta|L\s*=|R\s*=|P\(|Pr\(|Equation|Reward|loss)", line)
    ][:160]
    pack = f"""论文元信息
序号：{paper.idx}
文件名：{paper.filename}
标题：{paper.title}
本地路径：{paper.path.resolve()}
参考文献移除：{refs_removed}

【摘要/引言片段】
{abstract}

【方法相关片段】
{method}

【实验相关片段】
{experiment}

【疑似公式/目标/奖励片段】
{chr(10).join(formula_lines)}
"""
    if len(pack) > max_chars:
        pack = pack[:max_chars]
    meta = {
        "idx": paper.idx,
        "filename": paper.filename,
        "title": paper.title,
        "chars": len(pack),
        "refs_removed": refs_removed,
    }
    return pack, meta


def load_completed() -> set[str]:
    completed: set[str] = set()
    if not PROGRESS.exists():
        return completed
    for line in PROGRESS.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "completed":
            completed.add(str(row.get("filename")))
    return completed


def append_progress(row: dict[str, object]) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(), **row}
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_index(papers: list[Paper]) -> None:
    completed = load_completed()
    lines = [
        "# Direct-read PDF Structured Analysis Index",
        "",
        f"Total PDFs: {len(papers)}",
        f"Completed: {len(completed)}",
        "",
        "| # | Paper | Analysis | Status |",
        "|---:|---|---|---|",
    ]
    for paper in papers:
        name = safe_stem(paper.path) + ".md"
        status = "completed" if paper.filename in completed else "pending"
        link = f"[{name}](./{name})" if (OUT / name).exists() else ""
        lines.append(f"| {paper.idx} | {paper.title} | {link} | {status} |")
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-packs", action="store_true")
    parser.add_argument("--max-chars", type=int, default=80_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    PACKS.mkdir(parents=True, exist_ok=True)
    papers = list_papers()
    selected = papers[args.start - 1 :]
    if args.limit:
        selected = selected[: args.limit]

    for paper in selected:
        pack, meta = make_pack(paper, args.max_chars)
        if args.prepare_packs:
            pack_path = PACKS / (safe_stem(paper.path) + ".txt")
            pack_path.write_text(PROMPT + "\n\n" + pack, encoding="utf-8")
            append_progress({**meta, "status": "pack-prepared", "pack": str(pack_path)})
            print(f"prepared {paper.idx}: {paper.filename}", flush=True)
        time.sleep(args.sleep)

    write_index(papers)


if __name__ == "__main__":
    main()
