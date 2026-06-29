# Privacy And Release Checklist

This folder was prepared for public release with the following rules:

- No API keys or bearer tokens are stored in source files.
- Private local paths were replaced with command-line arguments and `examples/idea.txt`.
- Downloaded PDFs, caches, Python bytecode, and raw JSONL model logs are excluded.
- Example outputs are lightweight summaries derived from public paper metadata.

Recommended final checks before pushing:

```bash
rg -n -uu "(PRIVATE_LOCAL_PATH|sk-[A-Za-z0-9]{20,}|Bearer\\s+[A-Za-z0-9]{20,}|password|secret)" .
git status --short
```

If the scan returns a real credential or private file path, remove it before publishing.
