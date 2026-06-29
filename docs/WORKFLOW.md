# Workflow Notes

## Stage 1: Metadata Collection

`fetch_2026_papers.py` collects public paper metadata from OpenReview, ACL Anthology, CVF Open Access, and AAAI OJS pages. It writes one Markdown file per conference under `papers_2026/`.

## Stage 2: Abstract Triage

`tools/deepseek_paper_analysis.py` reads conference Markdown files, evaluates each title and abstract against the research idea, and writes resumable JSONL plus CSV/Markdown summaries.

Primary outputs:
- `conference_hotspot_summary.md`
- `idea_deep_read_papers.md`
- per-conference `*_llm_judgments.csv`

## Stage 3: Direct Reading

`tools/download_direct_read_pdfs.py` downloads PDFs for selected papers. `generate_direct_pdf_structural_reports.py` converts PDFs to text and asks an LLM to produce structured method reports.

## Stage 4: Method Synthesis

`classify_direct_report_methods.py` groups reports into method-improvement categories. `extract_c01_c07_c08_datasets.py` extracts dataset and hardware signals. `joint_analyze_research_idea.py` synthesizes category-level implications for the research idea.

## Reproducibility

The workflow is resumable at the model-triage stage through JSONL append logs. Large raw logs and PDFs are not included in the GitHub-ready folder; regenerate them locally when needed.
