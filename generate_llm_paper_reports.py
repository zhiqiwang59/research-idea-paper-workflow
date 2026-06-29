from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


ROOT = Path("papers_2026/downloaded_papers")
OUT = Path("papers_2026/paper_reports_llm_zh")


ANALYSIS_PROMPT = """你是结构科研分析助手，目标是协助用户穿透论文方法的结构本质、假设逻辑与生成潜力，而非简化信息。你的任务包括：结构解构、流程重建、方法对比、假设批判、创新引导与自主扩展。

请基于用户提供的论文正文进行逐篇深读分析。要求：
1. 优先依据论文原文；所有推断、补全、跨文献关联必须显式标注依据来源：原文 / 公开文献或常识性结构知识 / 结构补全。
2. 不要写成泛泛模板；每一节都要围绕该论文的具体方法、模块、训练、实验和假设展开。
3. 若论文正文缺失训练细节、公式、实验设置等，必须写明“原文未给出”，再给出结构补全建议。
4. 输出中文 Markdown。包含 Mermaid 流程图、关键公式、伪代码、结构对比表。
5. 保持科研分析深度：可解构、可推演、可再造。

必须使用以下标准模板：

## 结构化摘要（四段式）
- 问题动机
- 方法结构
- 实验与评估
- 批判性评价

## 一、结构性解构（Information Decomposition）
### 1. 研究问题与动机（What / Why）
- 明确方法解决的具体问题及其在现有研究中的挑战。
- 拆解背景结构与结构性动因，如有公式化假设，需明确列出。
- 若论文中信息不全，需结合公开文献、自主知识库或领域常识，补全结构性信息。

### 2. 方法流程与模块化结构
- 全流程模块拆解：模型结构、各组件功能、损失设计、优化方案、训练过程等。
- 强调结构差异性与创新点：逐一指出与标准流程/主流方法的具体区别。
- 结合流程图/伪代码/公式等方式还原方法全景，保障可复现性与结构可操作性。
- 若方法逻辑不明、自变量未定义或流程省略，须主动推演可能设计，并标明假设依据。

### 3. 实验设置与评估方式
- 明确数据集、对比基线、指标定义，特别是自定义指标（需公式化说明）。
- 训练细节（Batch size、优化器、学习率、训练轮次等）完整列出，标明非常规设定。
- 若实验配置不充分或评估偏窄，需提出建议补充路径及其结构性意义。

### 4. 核心贡献与结构创新
- 梳理方法相较前人工作的结构差异与关键突破点（组件设计、流程调度、优化机制等），并结合引用文献进行对照。
- 所有创新点均需结合结构逻辑与数学形式加以解释。

## 二、批判性定位（Critical Framing）
### 1. 与现有工作的结构对比
- 选取关键对比工作，进行逐项结构分析：模型结构、假设形式、推理流程、泛化机制等。
- 明确该方法的优势、折中与潜在代价，避免泛泛而谈。

### 2. 结构假设与适用边界
- 指出该方法适用的数据类型、任务属性、推理前提等。
- 明确其结构性假设背后的理论支撑与可能的不适用场景。

### 3. 潜在结构风险或未解之处
- 分析可能的瓶颈：如模型复杂度、稳定性、收敛性问题、数据依赖、任务泛化困难等。
- 若无明确缺陷表达，需主动探测结构的薄弱环节或可能留白，作为潜在研究入口。

## 三、生成性引导（Generative Expansion）
### 1. 跨任务/跨领域迁移路径
- 推演该方法迁移至不同任务/模态/数据结构下所需调整的结构组件与训练机制。
- 预测适配难点、结构重构建议及其潜在表达优势。

### 2. 结构融合与组合范式探索
- 主动联想当前方法是否可与其他范式（如图神经网络、自监督学习、元学习等）融合。
- 明确融合点、结构兼容方式与可能提升路径，提供结构草图或流程构想。

### 3. 替代性结构假设与优化路径
- 主动提出可替代的结构设定（如不同的损失函数、模型单元、推理机制等），并评估其理论与实验可行性。
- 提出潜在结构变体与调参路径，探索其改进潜力或简化可能。

## 四、自主扩展机制（Autonomous Expansion）
### 1. 信息缺失场景的结构补全
- 如原文未给出全部公式、流程或实验细节，需结合已有知识或同类工作完成推理式结构补全，并注明“结构补全”来源。

### 2. 关键文献/方法主动补链
- 自动关联与该方法相关的核心文献、基础理论或相近方法，建立结构系谱链条。
- 若用户未提供对比项，系统应主动给出最具代表性的对比工作（标准范式/主流结构）。

### 3. 生成性问题提出与重构可能探测
- 主动提出方法可拆解的结构单元及其重构可能，辅助用户发现研究潜力与创新空间。
- 给出针对性生成性问题，用于科研路径演进。
"""


@dataclass
class Paper:
    conference: str
    rank: str
    title: str
    source_url: str
    path: Path


def read_manifest() -> list[Paper]:
    papers: list[Paper] = []
    with (ROOT / "download_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists() and path.suffix.lower() == ".pdf":
                papers.append(
                    Paper(
                        conference=row.get("conference", ""),
                        rank=row.get("rank", ""),
                        title=row.get("title", path.stem),
                        source_url=row.get("source_url", ""),
                        path=path,
                    )
                )
    return papers


def pdf_to_text(path: Path) -> str:
    cp = subprocess.run(
        ["pdftotext", "-nopgbrk", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cp.returncode != 0:
        err = cp.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"pdftotext failed for {path}: {err}")
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


def make_reading_pack(paper: Paper, max_chars: int) -> tuple[str, bool, int, bool]:
    raw = pdf_to_text(paper.path)
    no_refs, refs_removed = strip_references(raw)
    truncated = len(no_refs) > max_chars
    if truncated:
        # Preserve the first part of the paper, where method and experiments usually live.
        # The report must disclose truncation in its provenance header.
        no_refs = no_refs[:max_chars]
    header = f"""论文元信息
标题：{paper.title}
会议/序号：{paper.conference} #{paper.rank}
来源：{paper.source_url}
本地文件：{paper.path}
参考文献是否移除：{refs_removed}
正文是否因长度截断：{truncated}
输入正文字符数：{len(no_refs)}

论文正文开始
"""
    return header + no_refs, truncated, len(no_refs), refs_removed


def safe_name(paper: Paper) -> str:
    if paper.rank.isdigit():
        stem = f"{paper.conference}_{int(paper.rank):02d}_{paper.title}"
    else:
        stem = f"{paper.conference}_{paper.rank}_{paper.title}"
    stem = re.sub(r"[:/\\?*\"<>|]+", "", stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    return stem[:170] + ".md"


def response_text(response) -> str:
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def call_llm(client: OpenAI, model: str, paper: Paper, reading_pack: str, max_output_tokens: int) -> str:
    user_content = f"""请深读并分析以下论文。不要只复述摘要；要围绕论文具体方法重建结构、流程、实验和批判性边界。

{reading_pack}
"""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_output_tokens,
    )
    text = response_text(response)
    if not text:
        raise RuntimeError("empty model response")
    return text


def write_index(rows: list[tuple[str, str, str, str, str]]) -> None:
    with (OUT / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("# LLM 深读论文结构分析报告索引\n\n")
        f.write("这些报告由 OpenAI API 逐篇读取论文正文后生成，不是规则模板填充版。\n\n")
        f.write("| 会议 | 序号 | 论文 | 报告 | 状态 |\n|---|---:|---|---|---|\n")
        for conf, rank, title, name, status in rows:
            f.write(f"| {conf} | {rank} | {title} | [{name}](./{name}) | {status} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--max-chars", type=int, default=160_000)
    parser.add_argument("--max-output-tokens", type=int, default=12_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1, help="1-based paper index")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Set it before running LLM report generation.")

    OUT.mkdir(parents=True, exist_ok=True)
    papers = read_manifest()
    selected = papers[args.start - 1 :]
    if args.limit:
        selected = selected[: args.limit]

    client = OpenAI()
    rows: list[tuple[str, str, str, str, str]] = []
    all_by_name = {safe_name(p): p for p in papers}

    for idx, paper in enumerate(selected, args.start):
        out_path = OUT / safe_name(paper)
        progress = f"[{idx}/{len(papers)}] {paper.conference}_{paper.rank} {paper.title[:80]}"
        print(progress.encode("ascii", errors="ignore").decode("ascii"), flush=True)

        if out_path.exists() and out_path.stat().st_size > 2000 and not args.force:
            rows.append((paper.conference, paper.rank, paper.title, out_path.name, "ok-existing"))
            continue

        try:
            pack, truncated, char_count, refs_removed = make_reading_pack(paper, args.max_chars)
            report = call_llm(client, args.model, paper, pack, args.max_output_tokens)
            provenance = f"""# {paper.title}

> 生成方式：LLM 深读论文正文后生成  
> 模型：`{args.model}`  
> 会议/序号：{paper.conference} #{paper.rank}  
> 来源：{paper.source_url}  
> 本地文件：`{paper.path}`  
> 输入正文字符数：{char_count}  
> 参考文献移除：{refs_removed}  
> 因长度截断：{truncated}

"""
            out_path.write_text(provenance + report.strip() + "\n", encoding="utf-8")
            rows.append((paper.conference, paper.rank, paper.title, out_path.name, "ok"))
            time.sleep(args.sleep)
        except Exception as exc:
            out_path.write_text(
                f"# {paper.title}\n\n生成失败：{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            rows.append((paper.conference, paper.rank, paper.title, out_path.name, f"failed: {type(exc).__name__}"))

    # Include already generated reports outside the selected slice when rebuilding the index.
    existing = {name for _, _, _, name, _ in rows}
    for name, paper in all_by_name.items():
        if name in existing:
            continue
        out_path = OUT / name
        if out_path.exists() and out_path.stat().st_size > 2000:
            rows.append((paper.conference, paper.rank, paper.title, name, "ok-existing"))
    rows.sort(key=lambda r: (r[0], int(r[1]) if str(r[1]).isdigit() else 9999, r[2]))
    write_index(rows)


if __name__ == "__main__":
    main()
