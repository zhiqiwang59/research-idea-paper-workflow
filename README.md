# Research Idea Paper Workflow

This repository contains a reproducible research workflow for collecting 2026 AI conference paper metadata, triaging LLM-related work, selecting papers relevant to a long-horizon-agent research idea, and generating structured deep-read reports.

The released version intentionally excludes private local paths, API keys, caches, downloaded PDFs, and large raw model-output JSONL files.

## Workflow

1. Fetch public paper metadata:

```bash
python fetch_2026_papers.py AAAI ICLR ICML ACL CVPR
```

2. Run abstract-level LLM triage:

```bash
python tools/deepseek_paper_analysis.py --papers-dir papers_2026 --idea-file examples/idea.txt --out-dir llm_deepseek_analysis --conferences all
```

3. Download selected direct-read PDFs:

```bash
python tools/download_direct_read_pdfs.py --input llm_deepseek_analysis/idea_deep_read_papers.csv
```

4. Generate structured direct-read reports:

```bash
python generate_direct_pdf_structural_reports.py --model gpt-5.4
```

5. Classify method directions and extract datasets/hardware:

```bash
python classify_direct_report_methods.py
python extract_c01_c07_c08_datasets.py
```

6. Run joint analysis over selected method categories:

```bash
python joint_analyze_research_idea.py --idea-file examples/idea.txt
```

## Inputs And Outputs

- `papers_2026/`: fetched title, author, abstract, and URL files by conference.
- `llm_deepseek_analysis/`: model judgments, hotspot summaries, direct-read lists, and structured reports.
- `examples/`: small public example outputs included for repository readers.

## Secrets

Use environment variables instead of hard-coding credentials:

```bash
set OPENAI_API_KEY=your_key
set DEEPSEEK_API_KEY=your_key
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key"
$env:DEEPSEEK_API_KEY = "your_key"
```

Copy `.env.example` only for local use. Do not commit a filled `.env` file.

## System Dependencies

PDF report generation uses `pdftotext`, which is provided by Poppler. Install Poppler and ensure `pdftotext` is available on `PATH`.

## Privacy Notes

Before publishing, run:

```bash
rg -n -uu "(PRIVATE_LOCAL_PATH|sk-[A-Za-z0-9]{20,}|Bearer\\s+[A-Za-z0-9]{20,}|password|secret)" .
```

The included `.gitignore` excludes local caches, downloaded papers, generated PDFs, Python caches, and raw JSONL model logs.
