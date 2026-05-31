# Plan: Chatbot Filter Dropdowns

## Problem

The chatbot's "Company" filter is a free-text input — undiscoverable, error-prone, and limited to one dimension. Users can't scope the RAG knowledge base by product, project, type, or state, causing answers to bleed across unrelated tickets.

## Goal

Replace the single free-text input with 5 single-select dropdowns (Client, Product, Project, Type, State) that all narrow the RAG context via AND logic. Filters apply to both vector search and keyword fallback paths.

## Decisions (resolved)

| Decision | Choice |
|----------|--------|
| Filter dimensions | client, product, project_name, entity_type, entity_state |
| Select mode | Single-select per dimension |
| Matching | Exact equality (`=`); `(empty)` option for NULL/blank |
| Logic across dimensions | AND (each filter narrows) |
| Keyword fallback | Also filtered |
| UI layout | Single horizontal row of 5 dropdowns |
| `/rag/ask` | Also gets all 5 filters |
| Filter reset | Separate "Reset Filters" button |
| Data source | Single `GET /chat/filter-options` endpoint (distinct values from DB) |
| Option counts | Omitted |

## Steps

### 1. `database.py` — Add `get_distinct_filter_options()`

New function that runs 5 `SELECT DISTINCT ... WHERE IS NOT NULL AND != '' ORDER BY ...` queries against `entity_data` and returns a dict like `{"clients": [...], "products": [...], "projects": [...], "types": [...], "states": [...]}`.

### 2. `routes/chat.py` — New endpoint + updated handler

- **Add** `GET /chat/filter-options` calling `get_distinct_filter_options()`
- **Extend** `ChatSendRequest`: add `product`, `project`, `entity_type`, `entity_state` (`Optional[str]`, all default `None`)
- **Modify** `chat_send()`: build dynamic AND WHERE clause from non-null filters. Apply to both:
  - Vector search path (WHERE clause on embeddings query with JOIN entity_data)
  - Keyword fallback path (add JOIN entity_data + WHERE to `search_and_fetch_full`)

### 3. `routes/rag.py` — Same pattern

- **Extend** `AskRequest`: add same 4 optional fields
- **Modify** `ask_rag()`: same AND filter builder on both paths

### 4. Shared filter-builder helper (optional but recommended)

A small inline helper — or just repeat the pattern in both routes — to keep the SQL construction in sync:

```python
def _build_filter_clause(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    for col, val in filters.items():
        if val:
            clauses.append(f"ed.{col} = %s")
            params.append(val)
    return " AND " + " AND ".join(clauses) if clauses else "", params
```

### 5. `templates/chat.html` — UI

- **Add** `<select>` dropdowns for all 5 dimensions in a single row, replacing the current `<input>`
- **Populate** from `GET /chat/filter-options` fetched in Alpine.js `init()`
- **Add** Alpine state: `filters: {client: '', product: '', project: '', entity_type: '', entity_state: ''}` and `filterOptions: {}`
- **Add** "Reset Filters" button
- **Update** `sendMessage()` to include all 5 filter values in the JSON body
- **Update** `clearChat()` to reset filters to `''`

### 6. `shared/analysis.py` (if needed)

No changes expected — the prompt templates don't need to know about filters. They just receive filtered context.

### 7. `CONTEXT.md` — Update glossary

Already partially updated; finalize the `Chatbot Scoping` section.

### 8. `AGENTS.md` — Add new route

Document `GET /chat/filter-options` under the Chat section.

## Files changed

- `database.py` — new function
- `routes/chat.py` — new endpoint, updated model + handler
- `routes/rag.py` — updated model + handler
- `templates/chat.html` — new UI
- `CONTEXT.md` — glossary update
- `AGENTS.md` — route table update
