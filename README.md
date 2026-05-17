# Research Agent

A small AI agent that takes a high-level research goal, decomposes it into a TODO plan, executes each task using web search and a vector store as working memory, and produces a markdown report with inline citations.

Users can optionally upload their own documents (PDF, DOCX, TXT, MD) as additional research sources. The agent picks tools intelligently — uploaded documents first, then session memory, then fresh web search.

![architecture](docs/architecture.svg)

```
goal → planner (gpt-4o, forced fn-call) → 3-7 sub-questions
            ↓
   for each task: executor loop (gpt-4o-mini)
       │
       ├─ tool calls (web_search, fetch_url, search_memory, search_documents)
       └─ finish_task (result_summary + sources, minItems=1)
            ↓
   synthesizer (gpt-4o) → markdown report w/ inline [1][2] citations
            ↓
   verifier (gpt-4o-mini) → unsupported-claims list (informational)
```

---

## Run it

```bash
cp .env.example .env
# add OPENAI_API_KEY and TAVILY_API_KEY
docker compose up --build
```

Open <http://localhost:8000>. Type a goal, optionally attach files, hit Send.

The CLI is also available:

```bash
python -m src.cli run "Survey the trade-offs of long-context attention in LLMs."
```

---

## How the agent loop works

`src/agent.py`. Five phases, one entrypoint (`run_session`).

**Phase 0 — Input guardrail.** Regex check against empty/oversized goals and an injection blocklist (conservative — research about LLM safety must still pass).

**Phase 1 — Planning.** One `gpt-4o` call with a forced `create_plan` function call. JSON Schema enforces `tasks: list[{description, rationale}]` with `minItems: 3, maxItems: 7`. Tasks persist with `status=pending`; the UI's TODO panel populates from a `plan.ready` event.

**Phase 2 — Execution.** For each task, an inner tool loop on `gpt-4o-mini` (max 6 iterations). The model picks tools with `tool_choice="auto"`, parallel tool calls run via `asyncio.gather`, results route back as `tool` messages. The loop exits when `finish_task` is called with non-empty sources. **On the final iteration the runtime forces `tool_choice=finish_task` and removes the other tools** so the model can't search forever.

**Phase 3 — Synthesis.** One `gpt-4o` call. Input: goal + structured task summaries with sources. Output: markdown report with inline `[n]` citations and a Limitations section.

**Phase 4 — Verification.** `gpt-4o-mini` with a forced `report_verification` function call. Returns unsupported-claims list. **Informational only** — never blocks publication. Surfaced as an expandable "⚠ Verification notes" block.

**Resume.** Same `run_session` body with `resume=True`. Reuses an existing plan; skips `done` tasks; restarts `in_progress` tasks; picks up at the first `pending`.

**Citation enforcement.** Three layers: `finish_task.sources` has `min_length=1` (Pydantic) → `minItems: 1` in the JSON Schema → the model is structurally pushed to include sources. Empty calls raise `ValidationError` → returned as a tool message → model self-corrects. After 2 corrections the task fails.

---

## Tools

Five tools registered in `src/tools/`. Each has a Pydantic input/output model; OpenAI function schemas are derived from the input model.

| Tool | Backend | Purpose |
|------|---------|---------|
| `web_search(query, max_results)` | Tavily | Broad discovery. Returns `{title, url, snippet}`. |
| `fetch_url(url)` | httpx + readability-lxml | Fetches a page, extracts main content, summarizes with `gpt-4o-mini`. **The LLM never sees raw page text** — only a 150-token summary + a `memory_id`. Full text → Chroma. |
| `search_memory(query, k)` | ChromaDB | Semantic search over pages fetched earlier this session. |
| `search_documents(query, k)` | ChromaDB | Semantic search over uploaded documents (session-scoped + globals via `$in` filter). |
| `finish_task(result_summary, sources)` | terminal | Commits findings. `sources` schema-enforced non-empty. |

---

## Context strategy

Five rules, enforced in code (`src/agent.py`):

1. **Full page contents never enter LLM context.** `fetch_url` returns only a 150-token summary + `memory_id`. The full text goes to Chroma; the LLM retrieves chunks via `search_memory` if it needs them.
2. **Same for uploaded documents.** The LLM sees retrieved chunks via `search_documents`, never raw file content.
3. **Completed-task context is summary-only**, truncated to 200 tokens per task. No inner tool exchanges cross over.
4. **Cross-task memory lives in Chroma**, retrieved per-task via semantic search.
5. **8K input-token budget per LLM call**, counted with `tiktoken cl100k_base`. The inner loop keeps the last 3 tool exchanges verbatim and folds older ones into a scratchpad assistant message. When the total still exceeds the budget, `_build_context_window` drops oldest items — **always as whole assistant + tool-message exchanges**, never an assistant alone, so the OpenAI API's tool-pairing rule stays intact.

Today's date is injected into every planner/executor/synthesizer call so words like "recent" anchor to the current year, not the model's training cutoff.

---

## Evaluation

Two layers: automated (unit tests) and scenario-based (system tests).

**Unit tests:** `pytest -v` — 157 tests, ~3.8s, fully offline. Mocks at every external boundary (OpenAI, Tavily, Chroma, Postgres). Three regression tests are named after real bugs:

- `test_build_context_window_preserves_tool_pairing_under_truncation` — OpenAI 400 from orphan tool messages.
- `test_fetch_url_rejects_duplicate_within_task` — model loop on same URL.
- `test_last_iteration_restricts_to_finish_task_only` — iteration cap as advisory.

**System-level scenarios.** Each isolates one or two pipeline components — prompts, schema enforcement, or tool wiring — and defines success as observable behavior of that component, not just "the report looks good."

| # | Scenario (component isolated) | What "working" means |
|---|---|---|
| 1 | **Planner prompt** — give a goal with temporal language: *"What's the current latency floor for WebRTC voicebots?"* | Plan has 3-5 tasks. No task starts with *"Understand…"*, *"Explore…"*, *"Research…"* (planner's counter-pattern list held). Tasks reference the current year (planner saw the injected `Current date:` line). Each task has a non-empty `rationale`. `plan.ready` event fires within ~10s. |
| 2 | **Executor prompt + citation enforcement + forced-finish** — any researchable goal | Every completed task ends with `finish_task` carrying ≥ 1 source (schema layer enforced). At least one task's `result_summary` contains ≥ 3 numeric facts (executor's density rule). No task hits `iterations_exhausted` — if the model would loop, the runtime's forced `tool_choice=finish_task` on the last iteration produces a clean commit. If the model attempts empty sources, the activity feed shows a `tool.failed` then a retry, and the retry succeeds within 2 attempts. |
| 3 | **Synthesizer structural contract** — any completed run | Report contains all required sections: `# Title`, `## Summary`, ≥ 2 body sections, `## Limitations`, `## Sources`. Every inline `[n]` resolves to a numbered Sources entry — no orphans. Body section headings are content-derived (not *"Findings"*, *"Analysis"*, *"Section 1"*). Limitations names ≥ 2 categories from the prompt's list (Data Recency, Coverage Gaps, Source Quality, Unresolved Disagreements, Goal Facets). |
| 4 | **Verifier prompt calibration** — same run as #3 | Verifier returns 0-5 unsupported claims (within the prompt's calibration band). Each flagged claim is a quoted or paraphrased fragment ≤ 200 chars, not a generic comment like *"section 2 has issues."* If the synthesizer escalated wording (e.g. *"associated with"* → *"causes"*), the verifier catches it. Verification notes surface in the UI; the report still publishes (informational, not blocking). |
| 5 | **Document RAG + page metadata** — upload one PDF, ask *"Using the uploaded document, identify the author's three main claims and the evidence cited for each."* | At least one task description begins with *"Using the uploaded documents…"* or *"From the uploaded documents…"* (planner's document-grounding rule). The executor invokes `search_documents` (visible in the activity feed). At least one citation in the final report references the uploaded filename. If the PDF had real pages, page numbers appear in the chunk metadata. |

---

## Environment variables

`OPENAI_API_KEY`, `TAVILY_API_KEY` required. Defaults for everything else; see `.env.example`. Notable knobs: `MAX_TOOL_ITERATIONS_PER_TASK=6`, `TOKEN_BUDGET_PER_CALL=8000`, `MAX_CONCURRENT_SESSIONS=4`, `MAX_DOCS_PER_SESSION=10`, `MAX_UPLOAD_BYTES=10485760`.

---

## Repository

```
src/
  agent.py          # the loop
  llm.py            # AsyncOpenAI wrapper, model routing, retries, batched DB logging
  memory.py         # MemoryStore Protocol + Chroma backend + chunking
  ingest.py         # PDF/DOCX/TXT/MD → chunks → embeddings → Chroma + Postgres row
  guardrails.py     # check_goal (regex) + verify_report (LLM with forced fn-call)
  events.py         # asyncio.Queue per-subscriber event bus
  logging_setup.py  # structlog + contextvars
  prompts/          # 4 markdown prompts, eagerly loaded
  tools/            # web_search, fetch_url, search_memory, search_documents, finish_task
  api/              # FastAPI app, routes, SSE, SPA static
  cli.py            # typer CLI
tests/              # 157 tests, ~3.8s
```
