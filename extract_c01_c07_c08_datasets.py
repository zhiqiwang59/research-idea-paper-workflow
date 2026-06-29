from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from openai import OpenAI


ROOT = Path("llm_deepseek_analysis/direct_read_reports_structural")
IN = ROOT / "c01_c07_c08_experiment_extracts.csv"
OUT_CSV = ROOT / "C01_C07_C08_DATASETS.csv"
OUT_MD = ROOT / "C01_C07_C08_DATASETS.md"


PROMPT = """你是一名科研文献实验信息抽取助手。用户提供 28 篇论文的结构化报告 Experiment 段，均来自 C01/C07/C08 三类。

任务：抽取每篇论文实验中使用的数据集、benchmark、任务环境或评测集，整理为结构化表格。

要求：
- 只抽取 Experiment 段中出现的信息，不要编造。
- 如果没有具体数据集名称，写“原文未给出”或“未使用标准数据集/理论分析为主”。
- dataset_or_benchmark 可包含多个名称，用中文分号分隔。
- task_type 用简短中文概括，如“数学推理/代码生成/长程RAG/具身导航/激活引导”等。
- notes 写规模、特殊设置、是否自建数据集、是否只给任务类型等关键信息。
- 输出严格 JSON 数组，不要 Markdown。

每项格式：
{
  "category_id": "C01",
  "title": "...",
  "dataset_or_benchmark": "...",
  "task_type": "...",
  "notes": "..."
}
"""


def parse_json_array(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--max-tokens", type=int, default=8000)
    args = parser.parse_args()

    rows = []
    with IN.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "category_id": row["category_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "experiment": row["experiment"],
                }
            )

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")
    client = OpenAI(base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(rows, ensure_ascii=False)},
        ],
        max_tokens=args.max_tokens,
        temperature=0,
    )
    extracted = parse_json_array(response.choices[0].message.content or "")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category_id", "title", "dataset_or_benchmark", "task_type", "notes"],
        )
        writer.writeheader()
        writer.writerows(extracted)

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# C01/C07/C08 论文实验数据集整理\n\n")
        f.write("来源：对应 28 篇结构化报告的 `Experiment` 段。\n\n")
        f.write("| 类别 | 论文 | 数据集 / Benchmark / 环境 | 任务类型 | 备注 |\n")
        f.write("|---|---|---|---|---|\n")
        for row in extracted:
            f.write(
                f"| {row['category_id']} | {row['title']} | {row['dataset_or_benchmark']} | "
                f"{row['task_type']} | {row['notes']} |\n"
            )

    print(OUT_MD)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
