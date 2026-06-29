from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


ROOT = Path("llm_deepseek_analysis/direct_read_pdfs")
OUT = Path("llm_deepseek_analysis/direct_read_reports_structural")
PROGRESS = OUT / "progress.jsonl"


SYSTEM_PROMPT = """你是一名结构科研分析助手。目标不是总结论文，而是提炼论文的核心结构、数学方法、实验证据与研究价值，为后续科研复现和创新提供基础。

输出必须为中文 Markdown，采用固定四段式：

## Problem
- 论文解决什么问题？
- 为什么现有方法不足？
- 给出论文的核心假设（如有）。

## Method
按模块说明整个方法：
- 整体流程
- 每个模块作用
- 训练目标
- 优化方式

保留所有关键数学公式：
- 损失函数
- 概率模型
- 推导公式
- 核心算法

如果论文流程不完整，可依据公开文献进行合理补全，并注明属于“合理推断”或“公开文献”。

最后总结：
- 与已有方法最大的结构区别
- 方法真正的创新点

## Experiment
仅保留核心实验信息：
- 数据集：使用的数据集、重要规模
- Baseline：对比方法
- Metric：评价指标
- Result：主要实验结论、是否 SOTA、提升多少、最有效模块（Ablation）

训练超参数仅保留非常规设置；普通 Adam、Batch Size 等无需展开。

## Analysis
必须包含：
- 方法优势
- 局限
- 适用条件
- 核心结构假设
- 可能失败的场景
- 创新与研究机会：真正创新在哪里、可与哪些方法结合、哪些模块可替换、哪些假设值得放松、哪些方向值得继续研究

输出原则：
- 保留问题背景、方法流程、核心数学公式、实验设计、数据集、实验结果、方法创新、有价值分析。
- 忽略冗长背景、重复解释、作者写作性描述、无意义扩展、常规训练细节、无贡献消融。
- 所有推断必须注明来源：原文 / 公开文献 / 合理推断。
- 若原文未给出公式、训练细节或实验结果，必须明确写“原文未给出”，不要编造数值。
- 目标是以最少输出保留最多信息密度，使读者能够快速理解、复现并进一步发展该方法。
- 不要寒暄、不要说明“请查收”、不要输出任务确认；正文必须直接从“## Problem”开始。
"""


@dataclass(frozen=True)
class Paper:
    conference: str
    idx: str
    title: str
    source_url: str
    pdf_url: str
    path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest() -> list[Paper]:
    manifest = ROOT / "download_manifest.csv"
    papers: list[Paper] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "downloaded":
                continue
            path = Path(row.get("file", ""))
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists() and path.suffix.lower() == ".pdf":
                papers.append(
                    Paper(
                        conference=row.get("conference", ""),
                        idx=row.get("idx", ""),
                        title=row.get("title", path.stem),
                        source_url=row.get("source_url", ""),
                        pdf_url=row.get("pdf_url", ""),
                        path=path,
                    )
                )
    return papers


def safe_stem(paper: Paper) -> str:
    stem = f"{paper.conference}_{int(paper.idx):04d}_{paper.title}" if paper.idx.isdigit() else f"{paper.conference}_{paper.idx}_{paper.title}"
    stem = re.sub(r"[:/\\?*\"<>|]+", "", stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    return stem[:170]


def pdf_to_text(path: Path) -> str:
    cp = subprocess.run(
        ["pdftotext", "-nopgbrk", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cp.returncode != 0:
        err = cp.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"pdftotext failed: {err}")
    text = cp.stdout.decode("utf-8", errors="ignore")
    text = text.replace("\x0c", "\n")
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


def make_reading_pack(paper: Paper, max_chars: int) -> tuple[str, dict[str, object]]:
    raw = pdf_to_text(paper.path)
    no_refs, refs_removed = strip_references(raw)
    original_chars = len(no_refs)
    truncated = original_chars > max_chars
    if truncated:
        no_refs = no_refs[:max_chars]
    sha = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    meta = {
        "text_sha256_prefix": sha,
        "refs_removed": refs_removed,
        "truncated": truncated,
        "input_chars": len(no_refs),
        "original_chars_without_refs": original_chars,
    }
    header = f"""论文元信息
标题：{paper.title}
会议/序号：{paper.conference} #{paper.idx}
来源页面：{paper.source_url}
PDF URL：{paper.pdf_url}
本地文件：{paper.path}
参考文献是否移除：{refs_removed}
正文是否因长度截断：{truncated}
输入正文字符数：{len(no_refs)}

论文正文开始
"""
    return header + no_refs, meta


def response_text(response: object) -> str:
    txt = getattr(response, "output_text", None)
    if txt:
        return txt.strip()
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def call_llm(client: OpenAI, model: str, paper: Paper, reading_pack: str, max_output_tokens: int) -> str:
    user_content = (
        "请直接基于下面 PDF 抽取正文进行结构科研分析。不要输出模板空话；公式和实验结果只保留原文中能支持的内容。\n\n"
        + reading_pack
    )
    if model.startswith("deepseek-"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_output_tokens,
        )
        text = response.choices[0].message.content or ""
    else:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_output_tokens=max_output_tokens,
        )
        text = response_text(response)
    if not text:
        raise RuntimeError("empty model response")
    problem_pos = text.find("## Problem")
    if problem_pos > 0:
        text = text[problem_pos:]
    return text


def load_done() -> set[str]:
    done: set[str] = set()
    if not PROGRESS.exists():
        return done
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            done.add(str(row.get("file")))
    return done


def append_progress(row: dict[str, object]) -> None:
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_index(papers: list[Paper]) -> None:
    rows = []
    for paper in papers:
        out_name = safe_stem(paper) + ".md"
        out_path = OUT / out_name
        status = "ok" if out_path.exists() and out_path.stat().st_size > 1200 else "pending"
        rows.append((paper.conference, paper.idx, paper.title, out_name, status))
    with (OUT / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("# Direct PDF Structural Reports\n\n")
        f.write(f"Updated: {now_iso()}\n\n")
        f.write("| Conference | Idx | Paper | Report | Status |\n|---|---:|---|---|---|\n")
        for conf, idx, title, out_name, status in rows:
            f.write(f"| {conf} | {idx} | {title} | [{out_name}](./{out_name}) | {status} |\n")


def write_report(paper: Paper, model: str, meta: dict[str, object], report: str) -> Path:
    out_path = OUT / (safe_stem(paper) + ".md")
    provenance = f"""# {paper.title}

> 生成方式：直接读取 PDF 正文后由 LLM 生成结构科研分析  
> 模型：`{model}`  
> 会议/序号：{paper.conference} #{paper.idx}  
> 来源页面：{paper.source_url}  
> PDF URL：{paper.pdf_url}  
> 本地文件：`{paper.path}`  
> 输入正文字符数：{meta["input_chars"]}  
> 参考文献移除：{meta["refs_removed"]}  
> 因长度截断：{meta["truncated"]}  
> 正文哈希前缀：`{meta["text_sha256_prefix"]}`

"""
    out_path.write_text(provenance + report.strip() + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--max-chars", type=int, default=150_000)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1, help="1-based paper index")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    papers = read_manifest()
    write_index(papers)

    if not args.base_url and args.model.startswith("deepseek-"):
        args.base_url = "https://api.deepseek.com"

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY is missing. Initialized {OUT / 'INDEX.md'} but did not generate reports. "
            "Set OPENAI_API_KEY before running. For DeepSeek/OpenAI-compatible endpoints also set "
            "OPENAI_BASE_URL and --model."
        )

    done = load_done()
    selected = papers[args.start - 1 :]
    if args.limit:
        selected = selected[: args.limit]

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    for absolute_idx, paper in enumerate(selected, args.start):
        out_name = safe_stem(paper) + ".md"
        out_path = OUT / out_name
        print(f"[{absolute_idx}/{len(papers)}] {paper.conference}_{paper.idx} {paper.title}", flush=True)

        if not args.force and str(paper.path) in done and out_path.exists() and out_path.stat().st_size > 1200:
            print("  skip: already completed", flush=True)
            continue

        progress_base = {
            "time": now_iso(),
            "conference": paper.conference,
            "idx": paper.idx,
            "title": paper.title,
            "file": str(paper.path),
            "report": str(out_path),
        }
        try:
            reading_pack, meta = make_reading_pack(paper, args.max_chars)
            report = ""
            last_exc: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    report = call_llm(client, args.model, paper, reading_pack, args.max_output_tokens)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < args.retries:
                        time.sleep(min(30, 2 ** attempt * 5))
            if last_exc is not None:
                raise last_exc
            write_report(paper, args.model, meta, report)
            append_progress({**progress_base, "status": "ok", **meta})
            write_index(papers)
            time.sleep(args.sleep)
        except Exception as exc:
            append_progress({**progress_base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_index(papers)
            if args.stop_on_error:
                raise
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            continue

    write_index(papers)


if __name__ == "__main__":
    main()
