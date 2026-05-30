# Embedding Throughput Benchmark

## Goal
Find the optimal thread count for `auto_index_request_web()` concurrency against LM Studio (`text-embedding-nomic-embed-text-v1.5`). Avoid crippling the system while maximising reindex speed.

## Test Setup

### Entity Selection
Pick **500 entities** with full data (both comments AND entity_data with non-null state):

```sql
SELECT DISTINCT e.entity_id FROM entity_data e 
INNER JOIN comments c ON c.request_id = e.entity_id
WHERE e.entity_state IS NOT NULL
ORDER BY e.entity_id LIMIT 500
```

### Thread Counts to Test
`[1, 2, 4, 8, 12, 16, 20, 24, 32]`

### Script Structure (pseudocode)

```python
# For each thread count:
#   1. Clear embeddings for test IDs
#   2. Start timer
#   3. Submit all 500 entities to ThreadPoolExecutor(max_workers=N)
#   4. Wait max 60 seconds (via timeout on as_completed)
#   5. Record: entities_done, errors, chunks_created, elapsed
#   6. Calculate: entities/sec, chunks/sec, error_rate
```

### Key Details

**Each worker thread** calls `auto_index_request_web(eid)` which handles:
- Own DB connection (internal to the function)
- Prefix building
- Embedding generation via `LLMClient.generate_embedding()`
- DELETE existing + INSERT new rows
- Commit

**Thread-safe progress tracking** — use a `threading.Lock` around shared counters.

**Reset between runs** — clear all embeddings for test IDs before each thread count test so each run starts from zero (not accumulating from previous runs).

**Timeout** — 60 seconds max per thread count. If not all 500 entities complete, report the partial count.

### Metrics Collected per Thread Count

| Metric | Source |
|--------|--------|
| Threads | Test parameter |
| Entities completed | Counter |
| Errors | Counter (exception in worker) |
| Embedding chunks created | Sum of return values from `auto_index_request_web` |
| Elapsed time | `time.time()` diff |
| Entities/sec | Entities / elapsed |
| Chunks/sec | Chunks / elapsed |
| Error rate | Errors / (entities + errors) |

## Output

A table printed to console:

```
Threads  Entities  Errors  Chunks   Elapsed(s)  Ent/s   Chunks/s  Err%
1        120       0       840      60.0        2.00    14.00     0.0%
2        240       0       1680     60.0        4.00    28.00     0.0%
4        ...       ...     ...      ...          ...     ...       ...
...
```

## Optimal Selection

The optimal thread count is the **elbow point** before:
1. Throughput (entities/sec) stops increasing linearly
2. Errors start appearing (>0%)
3. Diminishing returns set in (<5% improvement over previous count)

## System Monitoring

During the benchmark, optionally monitor:
- **LM Studio** — watch for timeout errors in response, CPU/memory usage
- **System** — CPU, RAM, disk I/O (not expected to be bottlenecks)

## After Benchmark

The chosen thread count will be used to replace the sequential loop in `routes/rag.py:_reindex_work()` — swapping the `for i, rid in enumerate(ids):` loop for a `ThreadPoolExecutor(max_workers=N)` calling `auto_index_request_web(rid)` in parallel.
