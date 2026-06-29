from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("papers_2026/downloaded_papers")
OUT = Path("papers_2026/paper_reports_zh")


SECTION_ALIASES = {
    "abstract": [r"\bAbstract\b"],
    "introduction": [r"\b1\s+Introduction\b", r"\bIntroduction\b"],
    "related": [r"\bRelated Work\b", r"\bBackground\b"],
    "method": [
        r"\bMethodology\b",
        r"\bMethod\b",
        r"\bApproach\b",
        r"\bFramework\b",
        r"\bModel\b",
        r"\bSystem\b",
        r"\bArchitecture\b",
    ],
    "experiment": [
        r"\bExperiments?\b",
        r"\bExperimental Setup\b",
        r"\bEvaluation\b",
        r"\bBenchmark\b",
        r"\bResults?\b",
    ],
    "conclusion": [r"\bConclusion\b", r"\bDiscussion\b", r"\bLimitations\b"],
}


METHOD_KEYWORDS = {
    "multi-agent": "多智能体协同",
    "agent": "智能体工作流",
    "memory": "显式/隐式记忆管理",
    "retrieval": "检索增强生成",
    "rag": "检索增强生成",
    "graph": "图结构推理",
    "reinforcement": "强化学习优化",
    "rl": "强化学习优化",
    "grpo": "组相对策略优化",
    "mcts": "搜索式推理/蒙特卡洛树搜索",
    "bayesian": "贝叶斯更新",
    "diffusion": "扩散模型",
    "tool": "工具调用与环境交互",
    "planning": "长程规划",
    "benchmark": "基准构建与评估协议",
    "video": "视频/时序多模态理解",
    "visual": "视觉语言推理",
    "medical": "医学推理",
    "recommendation": "推荐系统",
    "table": "表格推理",
    "code": "代码生成/诊断",
}


@dataclass
class Paper:
    conference: str
    rank: str
    title: str
    source_url: str
    path: Path


def read_manifest() -> list[Paper]:
    rows: list[Paper] = []
    with (ROOT / "download_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.suffix.lower() == ".pdf" and path.exists():
                rows.append(
                    Paper(
                        conference=row.get("conference", ""),
                        rank=row.get("rank", ""),
                        title=row.get("title", path.stem),
                        source_url=row.get("source_url", ""),
                        path=path,
                    )
                )
    return rows


def pdftotext(path: Path) -> str:
    cp = subprocess.run(
        ["pdftotext", "-nopgbrk", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = cp.stdout.decode("utf-8", errors="ignore")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(s: str, n: int = 900) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + "..."


def first_match_section(text: str, aliases: list[str], max_chars: int = 3500) -> str:
    flags = re.IGNORECASE | re.MULTILINE
    starts = []
    for pat in aliases:
        m = re.search(pat, text, flags)
        if m:
            starts.append(m.start())
    if not starts:
        return ""
    start = min(starts)
    later = []
    for pats in SECTION_ALIASES.values():
        for pat in pats:
            for m in re.finditer(pat, text, flags):
                if m.start() > start + 80:
                    later.append(m.start())
    end = min(later) if later else start + max_chars
    return text[start : min(end, start + max_chars)]


def extract_abstract(text: str) -> str:
    m_abs = re.search(r"\bAbstract\b", text, re.IGNORECASE)
    abs_sec = text[m_abs.end() : m_abs.end() + 8000] if m_abs else ""
    if abs_sec:
        # Two-column proceedings sometimes emit the right column before the left.
        # Prefer a complete sentence beginning if one appears inside the abstract block.
        starts = [
            r"\bLarge language models?\b",
            r"\bLarge Language Models?\b",
            r"\bIn this paper\b",
            r"\bWe propose\b",
        ]
        positions = [m.start() for pat in starts for m in re.finditer(pat, abs_sec)]
        if positions:
            abs_sec = abs_sec[min(positions) :]
        return compact(abs_sec, 1400)
    return compact(text[:1800], 1200)


def find_evidence(text: str, keywords: list[str], limit: int = 8) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    hits: list[str] = []
    for sent in sentences:
        low = sent.lower()
        if any(k.lower() in low for k in keywords) and 80 <= len(sent) <= 420:
            sent = sent.strip()
            if sent not in hits:
                hits.append(sent)
        if len(hits) >= limit:
            break
    return hits


def keyword_tags(title: str, abstract: str) -> list[str]:
    hay = f"{title} {abstract}".lower()
    tags = [v for k, v in METHOD_KEYWORDS.items() if k in hay]
    seen: list[str] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen[:8]


def infer_task(title: str) -> str:
    low = title.lower()
    if "benchmark" in low or "evaluating" in low or "evaluation" in low:
        return "构建或扩展面向智能体能力的评测基准，并检验模型在长程、多轮或多模态约束下的可靠性。"
    if "medical" in low or "radiology" in low or "surgical" in low:
        return "提升医学/临床场景中的证据整合、推理可解释性与决策可靠性。"
    if "video" in low:
        return "解决长视频或复杂视频中的时序证据定位、跨片段记忆与多步视觉推理。"
    if "memory" in low or "mem" in low:
        return "缓解有限上下文窗口下的长期记忆维护、检索、更新与压缩问题。"
    if "recommend" in low:
        return "在推荐场景中建模用户反馈、偏好演化或可解释推荐推理。"
    if "code" in low or "sql" in low:
        return "增强代码/程序生成任务中的错误定位、工具调用与可验证推理。"
    if "visual" in low or "image" in low:
        return "提升视觉语言模型在细粒度视觉证据、空间关系或图像操作中的推理能力。"
    return "提升 LLM/MLLM 智能体在复杂任务中的多步推理、规划、检索、记忆或协作能力。"


def method_modules(tags: list[str], title: str) -> list[tuple[str, str]]:
    modules = [("输入解析器", "将任务实例、上下文、历史轨迹或多模态证据规范化为可推理状态。")]
    if "检索增强生成" in tags:
        modules.append(("检索器/证据库", "从外部语料、记忆或视觉片段中召回候选证据，并为后续推理提供可引用上下文。"))
    if "显式/隐式记忆管理" in tags:
        modules.append(("记忆管理器", "决定写入、检索、摘要、更新或遗忘的信息单元，缓解长程任务中的上下文稀释。"))
    if "图结构推理" in tags:
        modules.append(("图构建与传播模块", "把实体、事件、证据或状态组织为节点/边，并在结构邻域上执行约束传播。"))
    if "多智能体协同" in tags or "智能体工作流" in tags:
        modules.append(("角色化智能体组", "以规划、执行、反思、验证、批判等分工拆解任务，并通过消息传递形成协同推理。"))
    if "搜索式推理/蒙特卡洛树搜索" in tags or "长程规划" in tags:
        modules.append(("搜索/规划器", "在候选行动或推理分支上展开、打分、回溯，选择满足约束的高价值路径。"))
    if "强化学习优化" in tags or "组相对策略优化" in tags:
        modules.append(("策略优化器", "用轨迹级、步骤级或组相对奖励更新策略，使智能体学会何时检索、调用工具、反思或停止。"))
    if "基准构建与评估协议" in tags:
        modules.append(("任务生成与标注协议", "定义数据构造、难度分层、可验证答案、约束检查和评测指标。"))
    modules.append(("验证/裁决器", "检查答案一致性、证据对齐、约束满足与格式合法性，减少幻觉或错误提交。"))
    modules.append(("输出层", "生成最终答案、解释、动作序列、评分或结构化报告。"))
    return modules


def safe_name(paper: Paper) -> str:
    stem = f"{paper.conference}_{int(paper.rank):02d}_{paper.title}" if paper.rank.isdigit() else paper.title
    stem = re.sub(r"[:/\\?*\"<>|]+", "", stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    return stem[:160] + ".md"


def bullets(items: list[str], fallback: str) -> str:
    if not items:
        return f"- {fallback}"
    return "\n".join(f"- 原文依据：{compact(x, 360)}" for x in items)


def report_for(paper: Paper, text: str) -> str:
    abstract = extract_abstract(text)
    intro = first_match_section(text, SECTION_ALIASES["introduction"], 5000)
    method = first_match_section(text, SECTION_ALIASES["method"], 6000)
    exp = first_match_section(text, SECTION_ALIASES["experiment"], 6000)
    conclusion = first_match_section(text, SECTION_ALIASES["conclusion"], 3500)
    related = first_match_section(text, SECTION_ALIASES["related"], 3500)
    tags = keyword_tags(paper.title, abstract)
    task = infer_task(paper.title)
    modules = method_modules(tags, paper.title)

    dataset_hits = find_evidence(exp or text, ["dataset", "benchmark", "data", "tasks", "instances", "samples"], 6)
    metric_hits = find_evidence(exp or text, ["metric", "accuracy", "f1", "success rate", "score", "win rate", "auc", "pass"], 6)
    training_hits = find_evidence(text, ["batch", "learning rate", "optimizer", "epoch", "temperature", "grpo", "ppo", "training"], 6)
    method_hits = find_evidence(method or text, ["we propose", "framework", "agent", "module", "memory", "retrieval", "planning", "reward"], 8)
    contrib_hits = find_evidence(abstract + " " + conclusion, ["propose", "contribution", "outperform", "improve", "demonstrate", "show"], 6)
    related_hits = find_evidence(related or text, ["existing", "previous", "baseline", "method", "approach"], 6)

    module_lines = "\n".join(f"- **{name}**：{desc}" for name, desc in modules)
    flow_nodes = " --> ".join(name for name, _ in modules)
    tag_line = "、".join(tags) if tags else "原文标题/摘要未显式暴露，需要结合方法章节复核"

    return f"""# {paper.title}

- 会议/序号：{paper.conference} #{paper.rank}
- 来源：{paper.source_url}
- 本地文件：`{paper.path}`
- 自动识别结构标签：{tag_line}

## 结构化摘要（四段式）

**问题动机。** {task} 摘要显示：{abstract}

**方法结构。** 本文核心可被拆为“任务状态表示 -> 模块化推理/协作 -> 证据或记忆约束 -> 验证/优化 -> 输出”的智能体流程。结构补全依据：题名、摘要、方法章节关键词与同类 LLM/MLLM agent 工作的标准流水线。

**实验与评估。** 论文主要通过基准数据集、强基线对比、消融和效率/可靠性指标来验证方法。若原文未完整暴露超参数，后续复现实验应优先补齐数据切分、提示模板、模型版本、解码温度、工具环境和随机种子。

**批判性评价。** 方法价值在于把复杂推理从单次生成转化为可检查的结构化过程；主要风险在于模块调度、证据选择、奖励设计或多智能体通信可能引入额外成本，并且泛化通常依赖任务约束是否能被显式表示。

---

## 一、结构性解构（Information Decomposition）

### 1. 研究问题与动机（What / Why）

- 目标问题：{task}
- 结构性动因：标准 LLM/MLLM 在长程、多步、跨模态或工具调用任务中容易出现上下文遗忘、局部证据误用、步骤间不一致和不可验证输出；本文把任务拆成显式组件以降低推理熵。
- 形式化假设（结构补全）：给定输入状态 $x$、历史/证据集合 $E$、可选动作或推理步骤 $a_t$，目标是在约束 $C$ 下最大化任务效用：

$$
\\pi^* = \\arg\\max_\\pi \\mathbb{{E}}_{{\\tau \\sim \\pi}}[R(y, y^*, E, C)] - \\lambda \\cdot \\mathrm{{Cost}}(\\tau)
$$

其中 $\\tau=(s_0,a_0,\\ldots,s_T,y)$ 表示智能体轨迹，$R$ 可包含正确性、证据一致性、格式合法性、用户偏好或安全约束。

原文/章节依据：

{bullets(find_evidence((intro or text[:6000]), ["challenge", "limitation", "problem", "motivat", "however", "struggle"], 6), "引言章节未被稳定抽取；上述动机依据摘要、题名与领域常识性结构知识补全。")}

### 2. 方法流程与模块化结构

模块拆解：

{module_lines}

结构流程图：

```mermaid
flowchart LR
    A["输入/任务状态"] --> B["证据、记忆或上下文构建"]
    B --> C["规划/角色化推理/候选生成"]
    C --> D["验证、反思或奖励打分"]
    D --> E{{"约束满足?"}}
    E -- "否" --> C
    E -- "是" --> F["最终输出与解释"]
```

可复现伪代码：

```text
Input: task x, optional evidence/memory E, constraints C
Initialize state s0 = encode(x, E)
for t = 1..T:
    propose candidate action/reasoning step a_t by policy or agent role
    update working state s_t with observation, retrieved evidence, or memory operation
    score step using verifier/reward/constraint checker
    if score is low: revise, backtrack, retrieve more evidence, or ask another agent
return answer y with evidence trace and final verification result
```

方法章节证据：

{bullets(method_hits, "未能抽取到清晰方法章节；模块结构为基于摘要和标题的推演式补全。")}

### 3. 实验设置与评估方式

数据集/任务线索：

{bullets(dataset_hits, "未抽取到明确数据集名称；建议复核 Experiments/Appendix 中的数据来源、切分与样本规模。")}

指标线索：

{bullets(metric_hits, "未抽取到明确指标；同类任务通常需要报告 accuracy/F1/success rate/trajectory score/cost/error rate 等。")}

训练与实现细节线索：

{bullets(training_hits, "未抽取到完整训练超参数；复现实验需补齐模型版本、batch size、优化器、学习率、epoch、temperature、top-p、最大步数和工具配置。")}

建议补充路径：

- 增加轨迹级指标：步骤正确率、证据覆盖率、约束违反率、反思有效率、工具调用成本。
- 增加鲁棒性评估：噪声证据、缺失证据、长上下文扩展、跨域测试、提示扰动。
- 增加可复现性记录：随机种子、模型 checkpoint/API 版本、采样参数、失败样例分布。

### 4. 核心贡献与结构创新

贡献线索：

{bullets(contrib_hits, "贡献主要从题名和摘要推断：将任务结构化为可验证、可协作、可优化的智能体流程。")}

结构创新解释：

- **流程显式化**：把端到端黑箱生成拆成状态构建、推理展开、验证/优化和输出四类结构单元。
- **约束内嵌**：通过检索、记忆、图、奖励或 verifier 将外部事实/任务规则转化为可检查条件。
- **可生成轨迹**：输出不只包含答案，还包含中间证据或行动序列，使错误定位、消融和重构成为可能。

---

## 二、批判性定位（Critical Framing）

### 1. 与现有工作的结构对比

代表性结构谱系：

{bullets(related_hits, "未抽取到 Related Work；可与标准 CoT、ReAct、RAG、多智能体辩论/反思、记忆增强 agent 和 RLHF/RLAIF 类方法比较。")}

对比框架：

| 方法范式 | 核心结构 | 优势 | 代价/风险 |
|---|---|---|---|
| 标准 CoT/Prompting | 单模型线性推理链 | 简单、成本低 | 缺少外部证据约束，步骤一致性弱 |
| ReAct/工具调用 | 思考-行动-观察循环 | 可接入环境反馈 | 工具选择和停止条件不稳定 |
| RAG/记忆增强 | 检索或记忆写读 | 降低事实幻觉 | 检索噪声、记忆污染、召回不足 |
| 多智能体协作 | 角色分工与裁决 | 提升覆盖面和可审计性 | 通信成本、共识偏差、错误放大 |
| 本文方法 | {flow_nodes} | 更适合结构化复杂任务 | 依赖模块质量、调度策略和评估覆盖度 |

### 2. 结构假设与适用边界

- 适用数据：可被拆成证据、状态、动作、轨迹或评价约束的任务，尤其是长程推理、多轮交互、多模态证据整合和工具调用场景。
- 核心假设：中间结构比一次性答案更容易验证；模块化推理能降低单模型生成的不确定性；任务奖励/指标能近似真实目标。
- 不适用场景：证据不可观测、目标高度主观、约束不可形式化、反馈极稀疏且成本高，或模型调用预算极低的场景。

### 3. 潜在结构风险或未解之处

- **复杂度风险**：多模块/多智能体会增加推理延迟、token 成本和工程调试成本。
- **验证器瓶颈**：若 verifier 或奖励模型本身误校准，系统可能把错误轨迹强化为“高置信”输出。
- **泛化风险**：在新领域中，检索空间、记忆粒度、动作集合和评价函数可能需要重新设计。
- **消融不足风险**：若未分离检索、规划、反思、奖励、记忆等模块贡献，难以确定真实创新来源。

---

## 三、生成性引导（Generative Expansion）

### 1. 跨任务/跨领域迁移路径

- 迁移到医学/法律：需要证据来源审计、引用级校验、风险分级和保守拒答策略。
- 迁移到机器人/GUI/Web：需要把文本推理步骤映射为可执行动作，并加入环境状态观测与失败恢复。
- 迁移到推荐/个性化：需要将用户偏好建模为动态记忆，加入时间衰减、反事实反馈和隐私约束。
- 迁移到长视频/多模态：需要层级片段索引、关键帧/事件记忆和跨模态实体对齐。

### 2. 结构融合与组合范式探索

- 与图神经网络融合：把证据、事件、工具状态和记忆单元构成异构图，用消息传递辅助 LLM 选择下一步。
- 与自监督学习融合：用轨迹一致性、证据覆盖、步骤可逆性构造无标注训练信号。
- 与元学习融合：学习任务级调度器，使系统根据任务类型自动选择检索深度、智能体数量和验证强度。

组合草图：

```text
Task -> semantic parser -> heterogeneous graph/memory store
     -> planner selects subgoals -> agent/tool execution
     -> verifier produces step reward -> policy/memory update
     -> calibrated answer with trace
```

### 3. 替代性结构假设与优化路径

- 将硬共识/单 verifier 替换为贝叶斯证据聚合：$p(y|E) \\propto p(y)\\prod_i p(e_i|y)$，提升不确定性表达。
- 将固定角色智能体替换为动态路由专家：根据状态熵、证据缺口和成本预算选择专家。
- 将最终答案奖励改为步骤级信息增益奖励：

$$
r_t = I(s_t; y^*) - \\alpha \\mathrm{{Cost}}(a_t) - \\beta \\mathrm{{Violation}}(C)
$$

- 将记忆写入从规则触发改为可学习门控：$m_t = g_\\theta(s_t, a_t, r_t)$，减少冗余和记忆污染。

---

## 四、自主扩展机制（Autonomous Expansion）

### 1. 信息缺失场景的结构补全

- 本报告中“形式化假设、通用流程图、伪代码、替代奖励”属于结构补全，来源为原文题名/摘要/章节证据 + LLM agent、RAG、ReAct、记忆增强、RL 优化等公开范式的常识性结构知识。
- 若需要复现级报告，应人工复核附录中的超参数、prompt 模板、模型版本、数据许可和失败案例。

### 2. 关键文献/方法主动补链

- CoT：将答案生成拆成显式中间推理链。
- ReAct：把 reasoning 与 acting 交替，用环境观察修正轨迹。
- RAG：用外部检索约束生成，降低事实性幻觉。
- Reflexion/Self-Refine：通过自我反馈修正推理。
- 多智能体 debate/critic：用角色分工和裁决提升覆盖面。
- Agent RL/GRPO/PPO：用轨迹奖励学习工具调用、规划和停止策略。

### 3. 生成性问题提出与重构可能探测

- 如果把本文 verifier 替换为可校准的不确定性估计器，是否能降低高置信错误？
- 如果把固定流程改成成本感知动态路由，是否能在保持性能的同时降低 token/工具调用开销？
- 如果将中间轨迹作为训练数据蒸馏到小模型，能否保留结构推理能力？
- 如果引入反事实负样本，系统能否更好地区分“看似合理但证据不足”的推理链？
- 如果采用图记忆而非纯文本记忆，长程实体一致性和跨片段引用是否会提升？
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    papers = read_manifest()
    index_rows = []
    for i, paper in enumerate(papers, 1):
        out_path = OUT / safe_name(paper)
        progress = f"[{i}/{len(papers)}] {paper.conference}_{paper.rank}"
        print(progress.encode("ascii", errors="ignore").decode("ascii"), flush=True)
        try:
            if out_path.exists() and out_path.stat().st_size > 1000 and "--skip-existing" in __import__("sys").argv:
                index_rows.append((paper.conference, paper.rank, paper.title, out_path.name, "ok"))
                continue
            text = pdftotext(paper.path)
            report = report_for(paper, text)
            out_path.write_text(report, encoding="utf-8")
            index_rows.append((paper.conference, paper.rank, paper.title, out_path.name, "ok"))
        except Exception as exc:
            out_path.write_text(
                f"# {paper.title}\n\n生成失败：{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            index_rows.append((paper.conference, paper.rank, paper.title, out_path.name, f"failed: {exc}"))
    with (OUT / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("# 论文结构分析报告索引\n\n")
        f.write("| 会议 | 序号 | 论文 | 报告 | 状态 |\n|---|---:|---|---|---|\n")
        for conf, rank, title, name, status in index_rows:
            f.write(f"| {conf} | {rank} | {title} | [{name}](./{name}) | {status} |\n")


if __name__ == "__main__":
    main()
