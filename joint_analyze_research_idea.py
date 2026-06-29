from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from openai import OpenAI


ROOT = Path("llm_deepseek_analysis/direct_read_reports_structural")
TAXONOMY = ROOT / "method_improvement_taxonomy.csv"
DEFAULT_IDEA = Path("examples/idea.txt")
OUT = ROOT / "RESEARCH_IDEA_JOINT_ANALYSIS_C01_C07_C08.md"


PROMPT = """你是一名科研选题与方法论分析助手。用户提出了一个研究 idea。
你将收到：
1. 用户的完整 idea；
2. C01 规划-搜索-控制类、C07 表示编辑-激活/潜变量控制类、C08 世界模型-信念状态类的已有论文结构化报告摘录。

请做“联合分析”，目标不是逐篇总结，而是回答：这些已有论文如何启发、支撑、挑战、改造用户的研究 idea。

输出中文 Markdown，结构必须如下：

## 1. 总体定位
- 判断用户 idea 与 C08/C07/C01 的关系：主类、副类、交叉点。
- 用一句话给出最准确的研究定位。

## 2. 三类论文分别能提供什么启发
### C08 世界模型-信念状态类
### C07 表示编辑-激活/潜变量控制类
### C01 规划-搜索-控制类
每类都要包括：
- 已有论文的共同结构
- 对用户 idea 的直接启发
- 可以借用的数学/算法模块
- 需要避免的局限

## 3. 逐篇映射表
用表格列出每篇论文：
论文 | 类别 | 可借鉴模块 | 对用户 idea 的作用 | 应避免/可改进点

## 4. 对用户 idea 的结构性改造建议
围绕用户 idea 的核心概念、方法模块、实验设计、评价指标、适用边界和可复现路径提出具体改造。

## 5. 可形成的论文贡献点
给出 4-6 个可以写进论文 introduction/contribution 的贡献点，必须具体。

## 6. 最小可行实验路线
提出一个可复现的 MVP：
- 任务/数据集
- baseline
- metric
- ablation
- 关键实现

## 7. 最大风险与规避
指出该 idea 最可能被 reviewer 质疑的点，并给出规避方案。

要求：
- 不要泛泛而谈。
- 必须显式引用论文标题来说明启发来源。
- 不要编造摘录中没有的具体数值。
- 把“公开文献常识/合理推断”与“摘录依据”区分开。
"""


def section(text: str, name: str) -> str:
    m = re.search(rf"(?is)^##\s*{re.escape(name)}\s*(.*?)(?=^##\s*\w+|\Z)", text, re.M)
    return m.group(1).strip() if m else ""


def compact(text: str, limit: int = 4200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def build_pack(idea_file: Path) -> str:
    idea = idea_file.read_text(encoding="utf-8", errors="ignore")
    rows = []
    with TAXONOMY.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["primary_category_id"] in {"C01", "C07", "C08"}:
                rows.append(row)
    rows.sort(key=lambda r: (r["primary_category_id"], r["title"]))

    parts = ["# USER IDEA\n", idea, "\n\n# RELATED PAPERS FROM C01/C07/C08\n"]
    for row in rows:
        path = ROOT / row["file"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        method = compact(section(text, "Method"), 3200)
        analysis = compact(section(text, "Analysis"), 1800)
        parts.append(
            f"\n## {row['title']}\n"
            f"- Category: {row['primary_category_id']} {row['primary_category']}\n"
            f"- Taxonomy reason: {row['reason']}\n"
            f"- Method keywords: {row['method_keywords']}\n"
            f"\n### Method excerpt\n{method}\n"
            f"\n### Analysis excerpt\n{analysis}\n"
        )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-file", type=Path, default=DEFAULT_IDEA)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--max-tokens", type=int, default=14000)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")
    if not args.idea_file.exists():
        raise SystemExit(f"Idea file not found: {args.idea_file}")
    pack = build_pack(args.idea_file)
    client = OpenAI(base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": pack},
        ],
        max_tokens=args.max_tokens,
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    OUT.write_text(text.strip() + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
