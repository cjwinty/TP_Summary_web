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
| **RAG / Chatbot** | Comments + Summaries + Entity Data + Relations | Vector search against `embeddings` (pgvector `<->`), results grouped by entity, enriched with full `entity_data` + `entity_relations` from DB. Every chunk carries a metadata prefix (state, client, product). A standalone metadata blob per entity makes the holistic ticket profile searchable. |
| **Summarisation** | Comments + existing summary | All text concatenated into one block; entity type + ID injected in the prompt header |
| **Browse** | `entity_data` + `entity_relations` | Direct SQL queries, rendered in Alpine.js-driven Jinja2 templates |
| **Search** | Comments + custom fields | Text search across comment data with optional custom field and date filters |

## Embedding Structure

Every embedding chunk (all three `chunk_type` values: `comment`, `summary`, `metadata`) has a **metadata prefix** prepended so the LLM always knows which entity, state, client, and product it came from:

```
[Request #69650 | State: Resolved | Client: Acme Corp | Product: Widget Pro] <chunk text>
```

A standalone **metadata blob** (`chunk_type='metadata'`) is created per entity containing the full ticket profile: state, project, client, product, version, all custom fields, description (truncated to 2k chars), and relations. This makes the holistic ticket profile semantically searchable — users can find tickets by description content, custom field values, or relationship patterns.

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
| **Smart cache range** | SSE | Skips already-cached entities; refreshes entity_data for entities missing it |
| **Force cache range** | SSE | Deletes embeddings, re-fetches comments + metadata for every entity in range |
| **Entity Metadata Backfill** | SSE + bg thread | Finds entities with comments but no entity_data row; fetches + saves |
| **Project Name Backfill** | SSE (inline) | Single API call for all projects; updates all NULL `project_name` rows |
| **Reindex (missing)** | SSE | Generates embeddings (with metadata prefix + blob) only for entities that lack them |
| **Reindex (full)** | SSE | Regenerates all embeddings with metadata prefix + blob for every entity |

All long-running operations use SSE streaming with a progress bar. The same pattern is reused across reindex, backfills, and cache range.