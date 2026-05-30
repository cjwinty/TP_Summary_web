# AGENTS.md — TP Summary Web

## Required Reading

**At the start of each session, read AGENTS.md and CONTEXT.md in full.** This file (AGENTS.md) covers technical reference; CONTEXT.md covers the "why" behind the architecture, data sources, and how context is assembled. Both files must be in context before making any changes.

## What this is

FastAPI web application that fetches Targetprocess ticket comments, caches them in PostgreSQL, and uses LLMs (Ollama, LM Studio, OpenAI-compatible, AWS Bedrock) to generate structured summaries, answer questions via RAG chatbot, and browse cached entity data.

## Entrypoint

```bash
pip install -r requirements.txt
python main.py
```

Runs uvicorn on `http://localhost:8000` by default. PostgreSQL must be running locally (config via `.env`).

## Architecture

```
main.py
├── VERSION               — Major.minor version base; combined with git commit count at runtime
├── routes/
│   ├── summarise.py      — /summarise, /cache/update, /cached-ids
│   ├── browse.py         — /browse, /browse/projects, /browse/entity
│   ├── comments.py       — /comments/{id}, /entity/{id}/relations, /entity/{id}/data
│   ├── settings.py       — /settings (LLM, prompts, cache mgmt, backfills, cache-range)
│   ├── search.py         — /search
│   ├── chat.py           — /chat (RAG Q&A)
│   └── rag.py            — /rag (reindex, find-fixes)
├── shared/
│   ├── api.py            — TP API v1/v2 calls (comments, entity data, relations, projects)
│   ├── analysis.py       — LLM prompt templates and summarisation logic
│   ├── config.py         — .env config loader, LLM provider config, dynamic version from VERSION file + git commit count
│   ├── llm_providers.py  — BaseLLMProvider, LocalLLMProvider, CloudLLMProvider, LLMClient
│   └── prompt_chain_executor.py — multi-step prompt chains
├── database.py           — PostgreSQL CRUD (psycopg2, all tables)
├── jinja_env.py          — Jinja2 template environment
└── templates/            — Jinja2 HTML templates (base, index, browse, comments, settings, search, chat)
```

## Versioning

Version shown in the sidebar is computed at startup:
```
v{MAJOR}.{MINOR}.{COMMIT_COUNT}
```
- `MAJOR.MINOR` read from `VERSION` file at repo root (manually managed)
- `COMMIT_COUNT` from `git rev-list --count HEAD` (auto, deterministic across clones)
- Falls back to `0.0.0-dev` if `VERSION` file or git repo is unavailable

## Database (PostgreSQL + pgvector)

All tables under `public` schema:

| Table | Purpose | Vector-enabled |
|-------|---------|---------------|
| `comments` | Raw TP API comments per entity | No |
| `summaries` | LLM-generated summary blobs | No |
| `entity_data` | Entity metadata (description, state, project, 9 custom field columns + JSONB) | No |
| `entity_relations` | Links between entities (one level deep) | No |
| `request_custom_fields` | Legacy table (no longer queried — use `entity_data` instead) | No |
| `embeddings` | Vector embeddings for RAG, vector(1536) column. Three `chunk_type` values: `comment`, `summary`, `metadata`. HNSW index (`idx_embeddings_hnsw_cosine`) for sub-linear search. | Yes (pgvector `<=>` operator) |
| `chat_history` | Chat session messages | No |
| `prompts` | Prompt templates | No |
| `prompt_chains` | Multi-step prompt chain definitions | No |
| `prompt_chain_steps` | Individual steps within prompt chains | No |
| `prompt_chain_runs` | Execution records for prompt chains | No |
| `prompt_chain_run_steps` | Per-step execution results | No |

Embedding dimension normalised to 1536. Cosine distance via pgvector `<=>` operator. HNSW index (`idx_embeddings_hnsw_cosine`) for sub-linear vector search performance.

### Embedding structure

Every embedding chunk (comment, summary, metadata) has a **metadata prefix** prepended:
```
[Request #69650 | State: Resolved | Client: Acme Corp | Product: Widget Pro]
```
This makes every chunk self-describing — the LLM always knows which entity, state, client, and product it came from.

A standalone **metadata blob** (`chunk_type='metadata'`) is created per entity containing the full ticket profile: state, project, client, product, version, all custom fields, description (truncated to 2000 chars), and relations. This makes the holistic ticket profile semantically searchable.

## Routes

All routes are defined under their respective router in `routes/`. The `main.py` prefixes them with empty string (i.e. paths are as listed).

### Summarise (`/`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Summarise page |
| POST | `/summarise` | LLM summarisation for list of entity IDs |
| POST | `/cache/update` | Update comment cache for specific IDs (smart/force) |
| GET | `/cached-ids` | All cached request IDs |

### Browse (`/browse`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/browse` | Browse page with project → type → entity drill-down |
| GET | `/browse/projects` | Distinct cached projects (id + name) |
| GET | `/browse/projects/{project_id}/types` | Entity types in a project |
| GET | `/browse/projects/{project_id}/{entity_type}` | Entities list for a project+type |
| GET | `/browse/entity/{entity_id}` | Entity detail JSON (metadata + relations) |

### Comments (`/comments`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/comments/{id}` | Comments page for an entity |
| GET | `/entity/{id}/relations` | JSON: relations for entity |
| GET | `/entity/{id}/data` | JSON: entity_data for entity |

### Settings (`/settings`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/settings` | Settings page |
| POST | `/settings/llm` | Save LLM provider config |
| POST | `/settings/test-connection` | Test LLM connectivity |
| GET | `/settings/prompts` | Prompt list (HTML) |
| GET | `/settings/prompts/{name}` | Prompt content (JSON) |
| POST | `/settings/prompts` | Save prompt |
| POST | `/settings/prompts/reset/{name}` | Reset prompt to default |
| GET | `/settings/cache-stats` | Row counts for all data tables |
| GET | `/settings/health` | DB health (rows, orphans, indexes, size) |
| POST | `/settings/optimise` | VACUUM ANALYZE + orphan cleanup |
| POST | `/settings/clear-summaries` | Delete all summaries |
| POST | `/settings/clear-entity-data` | Delete all entity_data |
| POST | `/settings/clear-entity-relations` | Delete all entity_relations |
| POST | `/settings/clear-chat-history` | Delete all chat history |
| POST | `/settings/clear-all-cache` | Delete ALL cached data (FK-safe order) |
| POST | `/settings/backfill-metadata` | SSE: backfill entity_data for entities missing it |
| GET | `/settings/backfill-status` | Backfill progress |
| POST | `/settings/backfill-stop` | Stop metadata backfill |
| POST | `/settings/backfill-project-names` | SSE: fetch all project names, backfill NULLs |
| GET | `/settings/backfill-project-names-status` | Project name backfill progress |
| POST | `/settings/backfill-project-names-stop` | Stop project name backfill |
| POST | `/settings/cache-range` | SSE (bg thread): cache a range of entity IDs (smart/force), two-phase parallel with ThreadPoolExecutor (20/8 workers) |
| POST | `/settings/cache-range-stop` | Stop cache range operation |
| GET | `/settings/cache-range-status` | Cache range progress (running, current, total, message, phase, skipped, metadata_only, unchanged) |

### Search (`/search`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/search` | Search page |
| POST | `/search` | Search cached comments with filters |

### Chat (`/chat`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/chat` | Chatbot page |
| POST | `/chat/send` | RAG Q&A endpoint (stateful, 6-turn context, company filter). Groups results by entity, enriches with entity_data + relations. Returns structured sources `{id, type, state}`. |

### RAG (`/rag`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/rag/reindex-missing` | SSE: generate embeddings for entities missing them |
| POST | `/rag/reindex-all` | SSE: regenerate all embeddings |
| GET | `/rag/reindex-status` | Reindex progress |
| POST | `/rag/reindex-stop` | Stop reindex |
| POST | `/rag/ask` | RAG Q&A (grouped-by-entity context, client filter) |
| POST | `/rag/search` | Vector search returning raw embedding matches |
| POST | `/rag/index` | Index a single entity's comments + summary + metadata blob |
| POST | `/rag/find-fixes` | Find similar resolved tickets and synthesise fix instructions |

## Key Modules

### `shared/api.py` — TP API interaction
| Function | Description |
|----------|-------------|
| `get_comments(entity_id, use_cache=True)` | Fetch comments from TP v2 API or local cache |
| `get_entity_data(entity_id, entity_type)` | Fetch entity metadata from TP v1 API (XML, ElementTree, namespace-agnostic). Extracts description, state, project, dates, custom fields (9 mapped to SQL columns). |
| `get_entity_type(entity_id)` | Discover entity type via v2 `/Assignables` |
| `get_all_projects()` | Fetch all projects via v2 `/Project` |
| `get_relations(entity_id)` | Fetch relations via v2 `/Relation` with pagination |
| `refresh_entity_metadata(entity_id, depth=0, seen=set, force=False)` | Standalone fetch+save of entity_data and relations. Recurses one level into direct relations with cycle detection. `force=True` bypasses DB cache and re-fetches from API. Skips unsupported types (Period, Timesheet) without API calls. |

### `database.py` — PostgreSQL CRUD
| Function | Description |
|----------|-------------|
| `save_entity_data(entity_id, entity_type, ...)` | Upsert entity_data row with all columns + custom_fields JSONB |
| `get_entity_data(entity_id)` | Fetch entity_data row (parsed JSONB) |
| `save_relations(entity_id, entity_type, relations)` | Replace relations for entity (DELETE + INSERT) |
| `get_relations(entity_id)` | Fetch relations for entity from DB |
| `get_cached_projects()` | Distinct project_id/project_name pairs |
| `get_entities_by_project_and_type(project_id, entity_type)` | Entity list for project+type |
| `get_cache_counts()` | Row counts for all data tables |
| `check_database_health()` | Row counts, orphans, indexes, DB size |
| `optimise_database()` | VACUUM ANALYZE + orphan cleanup |
| `clear_entity_data()` / `clear_entity_relations()` / `clear_all_chat_history()` / `clear_all_cached_data()` | Targeted deletion |
| `_build_embedding_prefix(entity_id, entity_type, entity_data)` | Build compact metadata prefix for embedding chunks |
| `_build_metadata_blob(entity_id, entity_type, entity_data)` | Build full ticket profile for standalone metadata embedding |
| `_get_write_conn()` | Per-thread psycopg2 connection (via `threading.local()`) for concurrent embedding writes |
| `auto_index_request_web(request_id, index_summary=True)` | Generate embeddings for entity: collects all texts, calls `LLMClient.generate_embeddings_list()` once (batch API), deletes existing, inserts all chunks |
| `get_pending_entity_ids_for_metadata()` | Find entities with comments but no entity_data row |
| `backfill_metadata_for_ids(ids, workers=20)` | Parallel metadata fetch for many IDs using thread pool |

### `shared/llm_providers.py` — LLM abstraction
- `BaseLLMProvider` → `LocalLLMProvider` (Ollama, LM Studio) / `CloudLLMProvider` (OpenAI-compat, AWS Bedrock)
- `LLMClient` singleton manages provider selection at runtime
- `generate_embedding()` uses the provider's native embedding API; dimension normalised to 1536
- `generate_embeddings_list(texts)` — batch embedding via single API call (native batch for OpenAI-compat/Ollama, sequential fallback for Bedrock). Reduces API calls per entity from N to 1.
- `LLMClient` holds class lock only for `_ensure_provider()` validation; HTTP API calls run outside the lock, enabling true concurrent embedding across threads

### `shared/analysis.py` — Prompt templates
- `summarise_comments()` — main summarisation prompt
- `summarise_batch()` — batch summarisation
- `summarise_search_results()` — search result synthesis

## Settings: Cache Management

The Settings page provides unified cache management:

| Feature | Description |
|---------|-------------|
| **Cache Stats** | Live row counts for all 7 data tables |
| **Health Check** | Row counts, orphan detection, index listing, DB size |
| **Optimise** | VACUUM ANALYZE + orphan cleanup across all tables |
| **Clear buttons** | Individual clear for Entity Data, Entity Relations, Summaries, Chat History; Clear All Cache (FK-safe order) |
| **Entity Metadata Backfill** | SSE: two-phase parallel (ThreadPoolExecutor 20 workers). Phase 1: metadata + relations fetch. Phase 2: embedding generation. Reports phase transitions to UI. |
| **Resolve Project Names** | SSE: single TP API call to get all projects; backfill NULL project_name values |
| **Cache Range** | SSE (bg thread): two-phase parallel (ThreadPoolExecutor 20/8 workers). Reports progress, phase, skipped, metadata_only, and unchanged counts. Smart = missing only + metadata refresh; Force = full re-fetch + diff-based embedding regeneration |

## Cache Range Behaviour

- **Smart mode**: Queries which IDs are already cached, skips them, refreshes entity_data for cached entities missing metadata. Reports separate counts for new fetches vs metadata-only refreshes.
- **Force mode**: Re-fetches comments and entity metadata for every entity in range, compares old vs new data (JSON diff), and only regenerates embeddings for entities whose data actually changed. Unchanged entities preserve their existing embeddings.

## Unsupported Entity Types

Entity types not in `V2_TO_V1_ENDPOINT` (e.g. Period, Timesheet) are skipped in `refresh_entity_metadata()` — no relations API call, logged at DEBUG level only.

## Conventions

- **UK English** (`-ise` not `-ize`): `summarise`, `optimise`, `analyse`, `initialise`, `normalise` across all code identifiers, route paths, templates, and DB prompt names
- **PostgreSQL + pgvector** for vector storage and search
- **Python 3.8+**, targeting 3.12
- **No tests, no linter, no type checker, no CI** — utility app without a test harness
- **Docs in sync with code**: Every commit that changes architecture, routes, key functions, config schema, or data flow must also update AGENTS.md and/or CONTEXT.md in the same commit

## Important Constraints

- Config via `.env` file (copy from `.env.example`); API keys stored base64 in `secure_config.json`
- Changes to `.env.example` must stay in sync with `shared/config.py` env var names
- No formal DB migration system — schema changes applied inline in `database.py:init_db()` with `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ADD COLUMN IF NOT EXISTS`
- SSE streaming pattern used for all long-running operations (reindex, backfills, cache range)
