# HANDOFF — SEC Filings RAG Pipeline ("secrag")

Written mid-session because the previous agent's shell lost filesystem access to
this directory (see "Known issue" below) before it could keep working. This
document exists so a fresh agent/session can resume without re-deriving context.
It intentionally does not narrate every line of code — read the files.

## 0. First things to do in a new session

1. **Check basic shell access works**: `ls ~/Desktop/code/sec`. If that still
   fails with "Operation not permitted" while other paths work fine, the
   Bash-tool sandbox is still broken — see "Known issue" below before doing
   anything else. (Write/Read/Edit tools were unaffected by this even while
   Bash was blocked — this file was written that way.)
2. Run `git status` and `git log --oneline` to confirm exact current state —
   this doc describes state as of commit `27220cf` plus **uncommitted** M2
   work described below; don't trust the commit hashes here blindly, verify.
3. Check `docker compose ps` — Qdrant should be running (was healthy before
   the incident; machine was not rebooted, so it's likely still up).
4. Read `/Users/n/.claude/plans/read-convo-md-to-resume-transient-mochi.md` —
   this is the full approved project plan (milestones M0–M5 + stretch M/C).
   This handoff assumes you'll read that; it isn't duplicated here.

## 1. Where things stand

- **M0 (scaffold)** — done, committed.
- **M1 (ingestion)** — done, committed, reviewed by a review subagent,
  fixes applied and re-verified. Solid and working.
- **M2 (Tier A hybrid RAG)** — chunking, indexing, and retrieval modules are
  **written but UNCOMMITTED and NOT YET VERIFIED end-to-end**. The last
  attempt to index chunks into Qdrant never completed cleanly (see incident
  below) — the collection was created with the right schema but had **0
  points** at last check. Do not assume `index_chunks` works until you've
  personally run it once, successfully, start to finish.
- **Generation step (`secrag/generate.py`)** — not started. Decision already
  made (see §3): shell out to the `claude` CLI, not the Anthropic API/SDK.

## 2. Problems hit and how they were resolved (M0–M1)

These are already fixed and committed — listed so you don't re-discover/re-fix
them, not because they're still open:

- `fastembed` has no ONNX conversion of `bge-m3` → substituted
  `BAAI/bge-large-en-v1.5` (dense, 1024-dim) + Qdrant's native `Qdrant/bm25`
  (sparse) as two models instead of bge-m3's single multi-vector output.
  Same "hybrid for free" outcome.
- XBRL `by_concept()` does **partial string matching** and includes
  dimensioned (segment/member) breakdowns — you must filter
  `df["concept"] == exact_qualified_tag` AND `is_dimensioned == False` to get
  the single consolidated value. Verified against live Apple/MSFT/COST/NVDA/
  GOOGL data.
- pandas `NaN`/`NaT` are **truthy** in Python (`nan or x` → `nan`, not `x`) —
  a real bug in period-coalescing logic, fixed with `pd.isna()`.
- `unit_ref` is filer-specific and inconsistent (`usd` vs `U_USD`) — switched
  to the `currency` column, which is consistently `"USD"`.
- The filing's SIGNATURES section has no Item number — filtered out at
  extraction time rather than left as an ungrounded chunk.
- Added a `CACHE_VERSION` marker to the on-disk JSON cache
  (`data/raw/{ticker}/{accession}/*.json`) so logic changes don't silently
  serve stale cached extractions forever.
- A code-review pass caught and fixed: an overly-broad `except Exception`
  around the XBRL query (it doesn't actually raise for missing concepts —
  removed, don't re-add it), and dead-code-dressed-as-validation in the CLI
  (removed).

## 3. Generation approach — use the `claude` CLI, not an API key

Explicit user decision: for now, avoid the Anthropic API/SDK (which would
cost money against an API key) and instead shell out to the `claude` CLI,
which uses the user's Claude Max subscription auth. Validated working
invocation:

```
claude -p "<prompt>" \
  --system-prompt "<full system prompt>" \
  --tools "" \
  --disable-slash-commands \
  --model sonnet \
  --output-format json
```

- `--tools ""` and `--disable-slash-commands` keep it a lean, non-agentic,
  single-shot completion (no CLAUDE.md discovery, no tool schemas bloating
  the prompt). Confirmed this drops cache-creation overhead from ~7670 to
  ~1989 tokens for a trivial prompt.
- `--system-prompt` **replaces** the default system prompt entirely (not
  `--append-system-prompt`, which adds to it).
- Parse the JSON output's `"result"` field for the answer text.
- The JSON also reports `total_cost_usd` / `modelUsage` — this is a
  subscription-usage-equivalent estimate for accounting, not separate
  billing under Claude Max.
- `--model` accepts aliases (`sonnet`, `opus`, etc.) or full model IDs.

This still needs `generate.py` written and wired into `secrag ask`.

## 4. Incident: local compute caused severe lag, and a sandbox side-effect

While testing `index_chunks` (embeds 175 chunks via `bge-large-en-v1.5`
locally on CPU), the same job was accidentally launched **twice
concurrently**, doubling CPU/thread contention from two ONNX runtime thread
pools and causing severe system-wide lag ("python" visible in Activity
Monitor). This was diagnosed and both processes were killed by PID — confirm
before continuing that no local job is grinding the machine to run again.

**Standing user instruction (also saved to memory at
`/Users/n/.claude/projects/-Users-n-Desktop-code-sec/memory/feedback_no_local_compute.md`):**
do not run compute-heavy work locally without asking first — flag it and
propose moving to a remote compute node instead. This applies especially to
bulk indexing (many companies) and the M3 eval harness (embeds/reranks/
queries at real scale across FinanceBench's corpus).

**Side effect, separate issue:** an earlier attempt to clean up the two
stuck processes used a broad `pkill -f "<pattern>"`, which appears to have
also killed whatever process mediates this shell's macOS access grant to
`~/Desktop` — every Bash-tool filesystem operation under
`~/Desktop/code/sec` started failing with "Operation not permitted" (even
plain `ls`), while paths outside `~/Desktop` (e.g. `/private/tmp/...`)
remained fine, and — importantly — the Write/Read/Edit tools kept working
throughout. This points to it being specific to the Bash tool's per-session
sandbox profile, not a real macOS Full Disk Access / TCC problem (the user
confirmed Terminal.app already has Full Disk Access, which ruled that out).
**Fix: restart the Claude Code session** (exit and relaunch `claude` in the
project directory) — do not try to fix this by running more shell commands
or killing more processes; that's how it happened the first time. **Never
use broad `pkill -f "<pattern>"` to clean up stuck jobs — kill by specific
PID.**

## 5. Codebase orientation (brief — read the files for the rest)

- `secrag/ingest.py` — M1, done. EDGAR fetch, per-Item text sections + core
  XBRL facts, `Provenance` dataclass on everything, disk cache with version
  marker.
- `secrag/chunk.py` — M2, written, untested-at-scale. Sentence-aware chunking
  of `TextSection`s into `Chunk`s (~450 token target, 2-sentence overlap).
  Manually verified once on AAPL (175 chunks from 23 sections, reasonable
  length distribution) — that check was fine.
- `secrag/index.py` — M2, written, **not yet verified working end-to-end**.
  Embeds via `get_dense_model()`/`get_sparse_model()` (bge-large-en-v1.5 +
  Qdrant/bm25, both already cached locally from M0's smoke test — no fresh
  download needed) and upserts into Qdrant collection `secrag_chunks`
  (named vectors `dense`/`sparse`).
- `secrag/retrieve.py` — M2, written, untested. Hybrid search via Qdrant's
  native RRF fusion (`FusionQuery`), then reranks with `BAAI/bge-reranker-base`
  (fastembed doesn't have `bge-reranker-v2-m3` either — same substitution
  pattern as the embedding model). **Not yet downloaded/verified.**
- `secrag/generate.py` — does not exist yet. See §3.
- `secrag/cli.py` — `smoke-test` and `ingest` commands work. `ask` and `eval`
  are still stubs raising `NotImplementedError`.
- `.env` (gitignored, already created) has a real `SEC_IDENTITY` set;
  `ANTHROPIC_API_KEY` is intentionally blank since generation goes through
  the `claude` CLI instead.
- `data/raw/` (gitignored) has cached extractions for AAPL, MSFT, COST, NVDA,
  GOOGL as of last known state.
- Qdrant runs via `docker-compose.yml`, pinned to `qdrant/qdrant:v1.18.2`
  (note: the container that's actually running may still be tagged `latest`
  from before the pin — same image, cosmetic mismatch only, not worth fixing
  unless it becomes a real problem).

## 6. Working-style notes for this project (also in CLAUDE.md / memory)

- Ask before spending money (Claude API calls) or running anything
  compute-heavy locally — flag it, propose a compute node if it's real work.
- Only commit when explicitly asked; commit in logical, clearly-described
  batches, not one giant commit.
- After a milestone, spawn a review subagent against CLAUDE.md standards
  (simplicity/KISS, correctness, maintainability, performance,
  over-engineering), then personally triage findings — fix what's real, and
  explicitly justify (out loud, to the user) any finding you choose not to
  act on rather than silently applying or silently ignoring it.
- Verify against real data/live APIs rather than trusting secondhand summaries
  when correctness matters (this caught several real bugs in M1 — see §2).

## 7. Immediate next steps

1. Confirm shell access is restored; confirm git/docker state.
2. Re-attempt `index_chunks` for **one** company as a single, deliberate,
   supervised run — not backgrounded blindly, no duplicate concurrent
   invocations of the same job. If it looks like it'll be heavy, stop and
   ask the user about a compute node first, per §4.
3. Once indexing genuinely succeeds, verify `retrieve.hybrid_search` +
   `rerank` return sensible results for a real question (e.g. "What was
   Apple's net income in FY2024?").
4. Write `generate.py` per §3, wire `secrag ask`.
5. Verify against ~10 hand-picked questions across the 5 ingested companies
   (the M2 milestone's own success criterion).
6. Commit M2.
