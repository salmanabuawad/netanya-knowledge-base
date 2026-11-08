# Netanya knowledge base

A local RAG knowledge base built from the public website of Netanya Municipality
(www.netanya.muni.il): crawl → clean → chunk → label → embed → hybrid retrieval for a local LLM.
Everything runs on your own machine; embeddings and answers go only to a loopback Ollama.

## Pipeline

| Step | Script | Output (in `rag-data/`, git-ignored) |
| --- | --- | --- |
| 1. Crawl the site and its documents | `scripts/crawl_netanya.py` | `netanya/pages.jsonl`, `documents.jsonl`, `docs/`, `raw/` |
| 2. Move documents that are mainly private individuals' data aside | `scripts/remove_pii.py` | `netanya-private/` |
| 3. Clean, extract documents, chunk, label, index | `scripts/build_netanya_kb.py` | `netanya/kb/kb.sqlite`, `sources.jsonl`, `chunks.jsonl`, `REPORT.md`, `netanya_knowledge_base.jsonl` |
| 4. Embed chunks (bge-m3 via Ollama) | `scripts/embed_netanya_kb.py` | `netanya/kb/vectors.sqlite` |
| 5. Retrieve / answer | `scripts/netanya_rag.py`, `scripts/search_netanya_kb.py` | — |

```bash
pip install -r requirements.txt
python -P scripts/crawl_netanya.py                 # resumable; hours for the full site
python -P scripts/remove_pii.py --dry-run          # review, then run without --dry-run
python -P scripts/build_netanya_kb.py
ollama pull bge-m3
python -P scripts/embed_netanya_kb.py              # GPU: minutes; CPU only: ~18 h
python -P scripts/netanya_rag.py "איך מקבלים הנחה בארנונה?"
```

## What is in it (crawl of 2026-11-08)

- 4,236 sources (3,364 web pages, 872 documents) → 24,286 chunks (median ~1,150 chars).
- Hebrew 97%, plus English, Russian and French pages.
- Each chunk: `url`, `title`, `heading`, `domain` (20 municipal domains), `kind`, `category`,
  `review_status`, `topics`, `lang`, and `text` (title + breadcrumb + heading + content, ready to embed).
- 81 documents are scanned images or use legacy font encoding and are left out (would need OCR).
- Labelled Israeli ID numbers are replaced with `[הוסר]`.

## Attaching a local LLM

```python
import sys; sys.path.insert(0, "scripts")
from netanya_rag import KnowledgeBase

kb = KnowledgeBase()                                   # reads rag-data/netanya/kb/
hits = kb.retrieve("שעות קבלת קהל בארנונה", k=6, no_archives=True)
messages = kb.build_messages("שעות קבלת קהל בארנונה", hits)   # send to any local chat model
# or: kb.ask(question, llm="<ollama model>")
```

Retrieval fuses BM25 (with Hebrew prefix folding: `בארנונה` matches `ארנונה`) and bge-m3 vector
similarity by reciprocal-rank fusion, keeps at most two chunks per source, and falls back to keyword
search if vectors or Ollama are missing. The prompt tells the model to answer only from the
numbered sources and to cite them; `ask()` flags answers that cite sources they were not given.

## Limitations

- News, events and tenders go stale; `fetched_at` is crawl time, not publication date.
  Use `no_archives=True` / `--no-archives` for resident-facing answers, and re-crawl regularly.
- Labels (`domain`, `review_status`, `topics`) are heuristics. `service_candidate_review` marks
  pages to check by hand, not verified service procedures.
- `rag-data/netanya/kb/doc_text.jsonl` is an extraction cache that still holds unredacted
  document text: keep it private.
- Generated answers are not verified municipal advice; check the cited official URLs.
