from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


ROOT = Path("llm_deepseek_analysis/direct_read_reports_structural")
EXTRACT_CSV = ROOT / "method_extracts.csv"
OUT_CSV = ROOT / "method_improvement_taxonomy.csv"
OUT_MD = ROOT / "METHOD_IMPROVEMENT_TAXONOMY.md"


CATEGORIES = [
    ("C01", "规划-搜索-控制类", "通过显式规划器、搜索、MPC、任务分解、层级计划或行动控制改进长程推理/执行。"),
    ("C02", "记忆-上下文管理类", "通过长期记忆、上下文压缩、检索、摘要、状态缓存或自适应上下文选择改进智能体。"),
    ("C03", "多智能体协作-组织演化类", "通过多角色、多智能体拓扑、协同协议、辩论、层级组织或agent结构演化改进系统。"),
    ("C04", "验证-审计-鲁棒性类", "通过自验证、假设检验、一致性检查、风险检测、规则遵循或安全防护降低错误和幻觉。"),
    ("C05", "工具使用-环境交互类", "通过工具调用、MCP/文件/浏览器/实验环境交互、可验证轨迹或行动生成改进agent执行。"),
    ("C06", "强化学习-奖励优化类", "通过RL、偏好目标、过程奖励、信用分配、策略优化或轨迹级奖励改进推理/行动。"),
    ("C07", "表示编辑-激活/潜变量控制类", "通过激活 steering、表示编辑、潜在轨迹、latent CoT、控制向量或内部状态调制改进模型行为。"),
    ("C08", "世界模型-信念状态类", "通过显式世界模型、信念更新、贝叶斯过滤、状态空间建模或规则环境模型改进推理。"),
    ("C09", "基准-诊断-能力测量类", "主要贡献是构造benchmark、评测协议、失败诊断、能力边界或度量体系。"),
    ("C10", "多模态-具身-空间/视频类", "通过视觉、视频、空间信念、导航或具身任务结构改进多模态/实体环境推理。"),
    ("C11", "人物一致性-对话承诺类", "通过persona、角色一致性、长期陪伴、叙事承诺或多轮对话状态改进交互一致性。"),
    ("C12", "扩散/生成路径控制类", "围绕扩散模型、Schrodinger bridge、生成路径崩塌或能量控制的结构改进。"),
]


PROMPT = """你要基于论文结构化报告中的 Method 部分，对论文的“改进方法”做分类。

请只依据 Method 内容判断主类；标题只作为辅助。每篇必须选择一个 primary_category_id，可给 0-2 个 secondary_category_ids。

可选类别：
{categories}

判别原则：
- 按方法结构归类，不按数据集、任务领域或会议归类。
- 如果论文主要是 benchmark/诊断/测量，而不是提出改进算法，归 C09。
- 如果方法同时包含多类，primary 选择最核心的结构改进，secondary 填辅助结构。
- 输出严格 JSON 数组，不要 Markdown，不要解释。

每个对象格式：
{{
  "file": "...",
  "primary_category_id": "Cxx",
  "secondary_category_ids": ["Cxx"],
  "method_keywords": ["3-6个中文关键词"],
  "reason": "一句中文说明为什么归入该类"
}}
"""


def extract_method(text: str) -> str:
    m = re.search(r"(?is)^##\s*Method\s*(.*?)(?=^##\s*Experiment\s*|^##\s*Analysis\s*|\Z)", text, re.M)
    return m.group(1).strip() if m else ""


def build_extracts() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.glob("*.md")):
        if path.name == "INDEX.md":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        method = re.sub(r"\s+", " ", extract_method(text)).strip()
        rows.append({"file": path.name, "title": title, "method_chars": str(len(method)), "method": method[:2600]})
    with EXTRACT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "title", "method_chars", "method"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_json_array(text: str) -> list[dict[str, object]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def classify_batch(client: OpenAI, model: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    categories = "\n".join(f"- {cid}: {name} - {desc}" for cid, name, desc in CATEGORIES)
    items = [
        {
            "file": r["file"],
            "title": r["title"],
            "method": r["method"],
        }
        for r in rows
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT.format(categories=categories)},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        max_tokens=6000,
        temperature=0,
    )
    return parse_json_array(response.choices[0].message.content or "")


def write_outputs(rows: list[dict[str, str]], classifications: list[dict[str, object]]) -> None:
    by_file = {r["file"]: r for r in rows}
    category_name = {cid: name for cid, name, _ in CATEGORIES}
    merged = []
    for item in classifications:
        file = str(item["file"])
        base = by_file[file]
        cid = str(item["primary_category_id"])
        merged.append(
            {
                "file": file,
                "title": base["title"],
                "primary_category_id": cid,
                "primary_category": category_name.get(cid, cid),
                "secondary_category_ids": ";".join(item.get("secondary_category_ids", []) or []),
                "method_keywords": ";".join(item.get("method_keywords", []) or []),
                "reason": str(item.get("reason", "")),
            }
        )
    merged.sort(key=lambda r: (r["primary_category_id"], r["title"]))
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "primary_category_id",
                "primary_category",
                "secondary_category_ids",
                "title",
                "file",
                "method_keywords",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(merged)

    counts = Counter(r["primary_category_id"] for r in merged)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in merged:
        grouped[row["primary_category_id"]].append(row)

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# 方法改进类型聚类\n\n")
        f.write(f"输入：`{ROOT}` 中 103 篇结构化报告的 `Method` 部分。\n\n")
        f.write(f"结论：按主类统计，共归纳出 **{len(counts)} 类改进方法**。\n\n")
        f.write("| 类别 | 名称 | 数量 | 结构特征 |\n|---|---|---:|---|\n")
        for cid, name, desc in CATEGORIES:
            if counts[cid]:
                f.write(f"| {cid} | {name} | {counts[cid]} | {desc} |\n")
        f.write("\n## 各类论文\n\n")
        for cid, name, desc in CATEGORIES:
            papers = grouped.get(cid, [])
            if not papers:
                continue
            f.write(f"### {cid} {name}（{len(papers)}篇）\n\n")
            f.write(f"结构特征：{desc}\n\n")
            for row in papers:
                f.write(f"- {row['title']}  \n  关键词：{row['method_keywords']}  \n  判定：{row['reason']}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    rows = build_extracts()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")
    client = OpenAI(base_url=args.base_url)

    classifications: list[dict[str, object]] = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        print(f"classifying {i + 1}-{i + len(batch)} / {len(rows)}", flush=True)
        classifications.extend(classify_batch(client, args.model, batch))

    if len(classifications) != len(rows):
        raise RuntimeError(f"expected {len(rows)} classifications, got {len(classifications)}")
    write_outputs(rows, classifications)
    print(OUT_MD)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
