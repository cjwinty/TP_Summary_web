# Entity Metadata Backfill Enhancement

## Changes Needed

### File: `routes/settings.py`

#### 1. Add `phase` and `phase_message` to `_backfill_state`

```python
_backfill_state = {
    "running": False,
    "stop": False,
    "current": 0,
    "total": 0,
    "message": "",
    "error": None,
    "phase": "",
    "phase_message": "",
}
```

#### 2. Replace `_backfill_work()` with two-phase version

Replace the sequential `_backfill_work()` function with a version that uses `ThreadPoolExecutor(max_workers=20)` for both phases:

**Phase 1 — Metadata fetch** (parallel):
```python
with ThreadPoolExecutor(max_workers=20) as executor:
    fut_map = {executor.submit(refresh_entity_metadata, eid): eid for eid in ids}
    for fut in as_completed(fut_map):
        if _backfill_state["stop"]:
            _backfill_state["running"] = False
            _backfill_state["message"] = f"Stopped after {_backfill_state['current']} entities (metadata phase)."
            return
        eid = fut_map[fut]
        try:
            fut.result()
            successful_ids.append(eid)
        except Exception:
            pass
        _backfill_state["current"] += 1
```

**Phase 2 — Embedding generation** (parallel):
```python
emb_total = len(successful_ids)
_backfill_state["phase"] = "embeddings"
_backfill_state["current"] = 0
_backfill_state["total"] = emb_total
_backfill_state["phase_message"] = "Phase 2/2: Generating embeddings for RAG search..."

with ThreadPoolExecutor(max_workers=20) as executor:
    fut_map = {executor.submit(auto_index_request_web, eid): eid for eid in successful_ids}
    for fut in as_completed(fut_map):
        if _backfill_state["stop"]:
            _backfill_state["running"] = False
            _backfill_state["message"] = f"Stopped after {_backfill_state['current']}/{emb_total} (embedding phase)."
            return
        try:
            fut.result()
        except Exception:
            pass
        _backfill_state["current"] += 1

_backfill_state["running"] = False
_backfill_state["message"] = f"Done. Backfilled metadata + embeddings for {emb_total} entities."
```

**Add import at top of function**:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

#### 3. Update `_backfill_sse()` to handle phase transitions

In the SSE generator, after yielding the initial status message, track `last_phase` and send phase transition status events:

- Check `_backfill_state.get("phase_message", "")` each iteration — if non-empty, send as status event and clear it
- Include `phase` field in progress events: `data.phase`
- Add phase label to initial status: `Phase 1/2: Backfilling metadata for {len(ids)} entities...`

### File: `templates/settings.html`

#### Update `startBackfill()` JavaScript

In the `data.type === 'progress'` handler, show the phase name:

```javascript
if (data.type === 'progress') {
    const p = data.percent || 0;
    const phase = data.phase || 'metadata';
    if (barFill) barFill.style.width = p + '%';
    if (pct) pct.textContent = p + '%';
    const phaseLabel = phase === 'embeddings' ? 'Generating embeddings' : 'Backfilling metadata';
    if (status) status.textContent = phaseLabel + ' ' + data.count + '/' + data.total;
}
```

When `data.type === 'status'`, the status text already gets set — no change needed since the server sends descriptive messages like "Phase 1/2: Backfilling metadata...".

## Verification

1. Open Settings page, click "Backfill Missing"
2. Observe progress bar filling during Phase 1 ("Backfilling metadata X/Y")
3. Observe transition message "Phase 2/2: Generating embeddings..."
4. Observe final message "Done. Backfilled metadata + embeddings for N entities."
5. Check DB: `SELECT COUNT(*) FROM embeddings WHERE request_id IN (backfilled_ids)` should show entries
6. Check chatbot finds entities from backfilled IDs via vector search
