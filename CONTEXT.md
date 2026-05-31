# CONTEXT.md — TP Summary Web

> **Conceptual knowledge base. For technical reference (routes, key functions, DB schema, entrypoint) see AGENTS.md.**

## Goal

The application helps users analyse support tickets by retrieving structured information from multiple sources: **Targetprocess (TP) API Comments**, **Structured Entity Data**, and **Ticket Summaries**. By integrating these sources, we provide comprehensive context for LLM summarisation, RAG Q&A, and data browsing.

## Core Concepts

1. **Entity** — A core record in the system (e.g. Request, Bug, Feature). Managed by a unique `entity_id` and belongs to an `entity_type`.
2. **Project** — A grouping for entities (`project_id`, `project_name`). Scopes searches and context gathering.
3. **Entity Data** — Structured metadata attached to an entity: description, state, project, client, product, and custom fields. Stored in `entity_data` with both dedicated SQL columns (queryability) and a `custom_fields` JSONB column (LLM context + unmapped fields).
4. **Relations** — Links between entities (one level deep). Stored in `entity_relations`. Populated via `refresh_entity_metadata()` which recurses into direct relations with cycle detection.

## Data Sources & How They Flow

### 1. Comments (TP v2 API → `comments` table)
- Fetched via `GET /Comments` with `General.Id` and `CreateDate` queries.
- Checked against local cache first; `use_cache=False` forces a fresh fetch.
- Stored as JSON with `request_id`, `comment_data`, `entity_type`, `fetched_at`.

### 2. Entity Metadata (TP v1 API → `entity_data` table)
- Fetched via v1 XML API using `V2_TO_V1_ENDPOINT` mapping (Request→requests, Bug→bugs, etc.).
- Parsed with `xml.etree.ElementTree` (namespace-agnostic).
- Uses `include=[id,EntityState,Description,CreateDate,Project,CustomFields]` — `CustomFields` was previously omitted, causing all custom field columns to remain empty across all entities. Fixed by adding it to the `include` parameter.
- Extracts: `entity_state`, `description`, `create_date`, `project_id`, `project_name`, custom fields.
- **9 custom field columns**: `customer_ref`, `internal_priority`, `support_level`, `next_action`, `paid_work`, `downtime`, `out_of_hours`, `customer_chased_date`, `stop_feedback_request`.
- Unsupported entity types (Period, Timesheet) are skipped before any API call — logged at DEBUG level.

### 3. Relations (TP v2 API → `entity_relations` table)
- Fetched via `GET /Relation` with pagination. Follows master/slave direction to find the "other" entity.
- Stored with `entity_id`, `related_entity_id`, `related_entity_type`, `related_entity_name`, `related_entity_state`.

### 4. Project Names (TP v2 API)
- Fetched via single `GET /Project?take=200` call.
- Returns `{"id": ..., "name": ...}` items. Backfilled into `entity_data.project_name` for any NULL values.

## How Context is Assembled

| Use Case | Data Sources | Method |
|----------|-------------|--------|
| **RAG / Chatbot** | Comments + Summaries + Entity Data + Relations | Vector search against `embeddings` (pgvector `<=>` cosine distance, HNSW index), results grouped by entity, enriched with full `entity_data` (entire metadata blob including description, project, all custom fields, site) + `entity_relations` from DB. Every chunk carries a metadata prefix (state, client, product). A standalone metadata blob per entity makes the holistic ticket profile searchable. Client filter joins `entity_data` (not deprecated `request_custom_fields`). |
| **Summarisation** | Comments + existing summary | All text concatenated into one block; entity type + ID injected in the prompt header |
| **Browse** | `entity_data` + `entity_relations` | Direct SQL queries, rendered in Alpine.js-driven Jinja2 templates |
| **Search** | Comments + custom fields | Text search across comment data with optional custom field and date filters |

## Embedding Structure

Every embedding chunk (all three `chunk_type` values: `comment`, `summary`, `metadata`) has a **metadata prefix** prepended so the LLM always knows which entity, state, client, and product it came from:

```
[Request #69650 | State: Resolved | Client: Acme Corp | Product: Widget Pro] <chunk text>
```

A standalone **metadata blob** (`chunk_type='metadata'`) is created per entity containing the full ticket profile: state, project, client, product, version, all custom fields, description (truncated to 2k chars), and relations. This makes the holistic ticket profile semantically searchable — users can find tickets by description content, custom field values, or relationship patterns.

## Chatbot Retrieval Pipeline

The chatbot and `/rag/ask` endpoint share a common retrieval pipeline in `shared/retrieval.py:vector_search()`. It operates in stages:

1. **Query top 80 chunks** by cosine distance from the embeddings table, with optional `INNER JOIN entity_data` filters
2. **Group by entity** (`request_id`) — each entity's chunks are collected together
3. **Apply exclusion set** — entities seen in the last 3 conversation turns are excluded unless explicitly referenced via `#ID` or `id NNNNN` in the user's message
4. **Sort entities by best-chunk distance** — the entity with the closest-matching chunk comes first
5. **Take top 10 entities** as candidates
6. **Dynamic token budget allocation** (~30k tokens): for each entity, add the metadata blob first, then distance-sorted chunks (up to 1200 chars each) until the budget is exhausted. This adapts naturally: specific queries dive deep on 1-2 entities, broad queries spread across all 10.

### Multi-turn Re-query

On turn 2+ in the chatbot, the user's question is rewritten into a standalone search query before embedding:

1. A prompt containing the last 3 messages (2 user + 1 assistant) and the current question is sent to the LLM
2. The LLM returns a concise standalone query with pronouns resolved — rule #5 instructs it to preserve any ticket IDs (`#12345` or `id 12345`)
3. If the rewrite call fails, the last 3 messages are concatenated as a fallback embedding input
4. The rewritten query is embedded and sent to `vector_search()`

### Entity Exclusion Sliding Window

- Each conversation turn stores the set of entity IDs returned as sources
- On the next turn, entities from the last 3 turns are excluded from vector search
- If the user types `#12345` or `id 12345` in their message, that entity is exempted from exclusion
- If exclusion empties the result set, the exclusion is cleared and the search retried (edge case: exhaustive query on a small cache)

### Direct-ID Fallback

After vector search (or keyword fallback) runs, both `/chat/send` and `/rag/ask` check the original message for mentions of entity IDs via `_parse_mentioned_ids()`. Any mentioned IDs not already present in the vector search results are fetched directly from the database via `_fetch_direct_entity_context()` — which retrieves the entity's metadata blob (state, project, client, product, custom fields, description) and up to 10 cached comments. This guarantees that explicitly referenced entities always appear in context, regardless of their vector similarity score.

### Focus Tracking

On follow-up turns where the user does not re-mention an entity ID (e.g. "what about its relations?"), the primary entity from the previous turn is automatically injected into context. Each turn stores the first source's ID as `focus_id` in the session state. On the next turn, if `focus_id` is not already in the current sources, its context is fetched directly and prepended. This keeps the conversation topic alive across natural follow-up questions. When the user explicitly mentions a new ID, that ID becomes the new focus. Clicking "Change Direction" clears `focus_id` along with the exclusion window.

### Chatbot Scoping

The chatbot supports scoping the RAG knowledge base by five filter dimensions, all sourced from `entity_data`:
- **Client** (`entity_data.client`)
- **Product** (`entity_data.product`)
- **Project** (`entity_data.project_name`)
- **Entity Type** (`entity_data.entity_type`)
- **Entity State** (`entity_data.entity_state`)

These filters narrow the vector search and/or keyword fallback to only consider entities matching the selected dimension values. This prevents answer bleed across unrelated products, projects, or states.

Each dimension is **single-select** — one value at a time. Leaving all dimensions blank means no scoping (search across all cached entities).

Matching is **exact equality** (`=`) rather than partial ILIKE, since dropdown values are known exact strings. NULL/empty values are excluded from dropdown options. Filters combine via **AND** logic — each additional filter narrows the result set. Filters apply to both the vector search path (via `INNER JOIN entity_data` + WHERE clause) and the keyword fallback path (post-filtered via `entity_data` lookup).

Dropdown options are populated from a dedicated `GET /chat/filter-options` endpoint that queries `SELECT DISTINCT ... FROM entity_data` for each of the five columns, excluding NULL/empty values.

## Key Variables (Prompt Injection)

Prompts receive these variables automatically:

| Variable | Source |
|----------|--------|
| `{{entity_type}}` | `entity_data.entity_type` |
| `{{entity_id}}` | user-provided ID |
| `{{project_name}}` | `entity_data.project_name` |
| `{{description}}` | `entity_data.description` |
| `{{entity_state}}` | `entity_data.entity_state` |
| `{{custom_fields}}` | `entity_data.custom_fields` JSONB |

## Backfill & Caching Strategies

| Operation | Pattern | Behaviour |
|-----------|---------|-----------|
| **Smart cache range** | SSE + bg thread pool | Two-phase parallel (20/8 workers): Phase 1 fetches comments + metadata for missing and stale entities, Phase 2 generates embeddings. Reports phase transitions, skipped, and metadata_only counts. |
| **Force cache range** | SSE + bg thread pool | Two-phase parallel (20/8 workers): Phase 1 re-fetches all comments + metadata for **every ID in the range** (not just un-cached ones — force mode was skipping existing entities, fixed), compares old vs new data (JSON diff), tracks changed vs unchanged. Phase 2 regenerates embeddings only for changed entities. Unchanged entities preserve existing embeddings. |
| **Entity Metadata Backfill** | SSE + bg thread pool | Two-phase parallel (20 workers): Phase 1 fetches metadata + relations, Phase 2 generates embeddings. SSE reports phase transitions. |
| **Project Name Backfill** | SSE (inline) | Single API call for all projects; updates all NULL `project_name` rows |
| **Reindex (missing)** | SSE + bg thread pool | Generates embeddings only for entities that lack them; 8-worker parallel batch system |
| **Reindex (full)** | SSE + bg thread pool | TRUNCATEs then regenerates all embeddings; 8 workers calling `auto_index_request_web()` with per-thread DB connections and batch LLM embedding API |

## Vector Search Indexing

The `embeddings` table uses an **HNSW (Hierarchical Navigable Small World)** index on the `embedding` column using the `vector_cosine_ops` operator class. This provides sub-linear vector search performance — queries against ~35k embeddings return in single-digit milliseconds vs hundreds of milliseconds for sequential scan.

- **Distance operator**: `<=>` (cosine distance). Lower values = more semantically similar. Range: 0 (identical direction) to 2 (opposite).
- **Index type**: HNSW (supported by pgvector 0.8.2+). Chosen over IVFFlat for better recall without tuning, low memory overhead at current scale.
- **Creation**: `database.py:init_db()` with `CREATE INDEX IF NOT EXISTS ... USING hnsw (embedding vector_cosine_ops)`. Gracefully skipped if pgvector version lacks HNSW support.

## Deprecated Tables

`request_custom_fields` is a dead data source — no new rows written since SQLite migration. All queries (client filter, search) use `entity_data` instead. The sole remaining query against this table (`search_cached_issues_by_product_keyword`) was migrated to `entity_data`. The table and its indexes are preserved for backward compatibility but not queried by any route.

All long-running operations use SSE streaming with a progress bar. The same pattern is reused across reindex, backfills, and cache range. The settings page checks all four status endpoints (`/rag/reindex-status`, `/settings/backfill-status`, `/settings/cache-range-status`, `/settings/backfill-project-names-status`) on page load via Alpine.js `init()` — if any operation is still running server-side, the UI re-enters running state and polls every 2s until completion. This ensures running jobs survive navigation away and back.