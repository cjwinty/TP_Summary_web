import asyncio
import json
import logging
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from database import (
    get_cached_comments,
    get_summary,
    save_comments,
    get_conn,
)

from shared import config as cfg

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared state for reindex progress/stop
_reindex_state = {
    "running": False,
    "stop": False,
    "current": 0,
    "total": 0,
    "message": "",
}


class IndexRequest(BaseModel):
    request_id: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    client: Optional[str] = None


class FindFixesRequest(BaseModel):
    request_id: int
    top_k: int = 10


migrated_embedding_schema = False


def _ensure_chunk_type_column():
    global migrated_embedding_schema
    if migrated_embedding_schema:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'chunk_type'
    """)
    if not c.fetchone():
        c.execute("ALTER TABLE embeddings ADD COLUMN chunk_type TEXT DEFAULT 'comment'")
        conn.commit()
        logger.info("Added chunk_type column to embeddings table")
    migrated_embedding_schema = True


def get_embedding(text: str) -> list[float]:
    try:
        cfg.initialise_llm()
        from llm_providers import LLMClient
        return LLMClient.generate_embedding(text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return [0.0] * 1536


@router.post("/rag/index")
async def index_comments(req: IndexRequest):
    _ensure_chunk_type_column()
    from database import _build_embedding_prefix, _build_metadata_blob, get_entity_data
    comments, fetched_at = get_cached_comments(req.request_id)
    if not comments:
        return JSONResponse({"error": "Request ID not in cache."}, status_code=404)

    entity_type = "Request"
    entity_data = get_entity_data(req.request_id)
    if entity_data:
        entity_type = entity_data.get("entity_type") or "Request"
    else:
        conn = get_conn()
        c2 = conn.cursor()
        c2.execute("SELECT entity_type FROM comments WHERE request_id = %s", (req.request_id,))
        row = c2.fetchone()
        if row:
            entity_type = row[0]

    prefix = _build_embedding_prefix(req.request_id, entity_type, entity_data)

    conn = get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("DELETE FROM embeddings WHERE request_id = %s", (req.request_id,))

    count = 0
    for comment in comments:
        text = comment.get("text", "").strip()
        if not text or len(text) < 20:
            continue
        prefixed = f"{prefix} {text[:5000]}"
        embedding = get_embedding(prefixed)
        if embedding and any(v != 0.0 for v in embedding):
            c.execute(
                "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                (req.request_id, prefixed, json.dumps(embedding), "comment", now, entity_type),
            )
            count += 1

    from database import get_summary
    summary, _ = get_summary(req.request_id)
    if summary and summary.strip() and len(summary) >= 20:
        prefixed = f"{prefix} {summary[:5000]}"
        embedding = get_embedding(prefixed)
        if embedding and any(v != 0.0 for v in embedding):
            c.execute(
                "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                (req.request_id, prefixed, json.dumps(embedding), "summary", now, entity_type),
            )
            count += 1

    # Metadata blob
    blob = _build_metadata_blob(req.request_id, entity_type, entity_data)
    if blob:
        embedding = get_embedding(blob[:5000])
        if embedding and any(v != 0.0 for v in embedding):
            c.execute(
                "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                (req.request_id, blob[:5000], json.dumps(embedding), "metadata", now, entity_type),
            )
            count += 1

    conn.commit()
    return JSONResponse({"message": f"Indexed {count} chunks for request #{req.request_id}."})


def _reindex_work(mode: str, ids: list[int]):
    """Run reindex work in a background thread. Updates _reindex_state for progress."""
    global _reindex_state
    import psycopg2
    from database import DB_CONFIG
    from database import _build_embedding_prefix, _build_metadata_blob, get_entity_data
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    c = conn.cursor()
    _ensure_chunk_type_column()
    total = 0
    n = len(ids)

    if mode == "full":
        c.execute("TRUNCATE embeddings")
        conn.commit()

    for i, rid in enumerate(ids):
        if _reindex_state["stop"]:
            _reindex_state["message"] = "Stopped by user."
            break

        _reindex_state["current"] = i + 1
        _reindex_state["message"] = f"Processing request #{rid} ({i + 1}/{n})"

        entity_type = "Request"
        entity_data = get_entity_data(rid)
        # fallback: guess type from comments table if entity_data missing
        if not entity_data:
            from database import get_cached_comments as _gcc
            c2 = conn.cursor()
            c2.execute("SELECT entity_type FROM comments WHERE request_id = %s", (rid,))
            row = c2.fetchone()
            if row:
                entity_type = row[0]
        else:
            entity_type = entity_data.get("entity_type") or "Request"

        prefix = _build_embedding_prefix(rid, entity_type, entity_data)

        comments, fetched_at = get_cached_comments(rid)
        if comments:
            for comment in comments:
                if _reindex_state["stop"]:
                    break
                text = comment.get("text", "").strip()
                if not text or len(text) < 20:
                    continue
                prefixed = f"{prefix} {text[:5000]}"
                embedding = get_embedding(prefixed)
                if embedding and any(v != 0.0 for v in embedding):
                    try:
                        c.execute(
                            "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                            (rid, prefixed, json.dumps(embedding), "comment", datetime.now().isoformat(), entity_type),
                        )
                        total += 1
                    except Exception:
                        continue
        if _reindex_state["stop"]:
            _reindex_state["message"] = "Stopped by user."
            break
        summary, _ = get_summary(rid)
        if summary and summary.strip() and len(summary) >= 20:
            prefixed = f"{prefix} {summary[:5000]}"
            embedding = get_embedding(prefixed)
            if embedding and any(v != 0.0 for v in embedding):
                try:
                    c.execute(
                        "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                        (rid, prefixed, json.dumps(embedding), "summary", datetime.now().isoformat(), entity_type),
                    )
                    total += 1
                except Exception:
                    continue
        # Metadata blob
        blob = _build_metadata_blob(rid, entity_type, entity_data)
        if blob:
            embedding = get_embedding(blob[:5000])
            if embedding and any(v != 0.0 for v in embedding):
                try:
                    c.execute(
                        "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                        (rid, blob[:5000], json.dumps(embedding), "metadata", datetime.now().isoformat(), entity_type),
                    )
                    total += 1
                except Exception:
                    continue
        conn.commit()

    label = "Reindexed" if mode == "full" else "Indexed"
    _reindex_state["running"] = False
    if not _reindex_state["stop"]:
        _reindex_state["message"] = f"{label} {total} chunks across {n} requests."


async def _reindex_sse(mode: str):
    """SSE generator that polls a background thread for progress."""
    global _reindex_state
    import threading

    _reindex_state["running"] = True
    _reindex_state["stop"] = False
    _reindex_state["current"] = 0
    _reindex_state["total"] = 0
    _reindex_state["message"] = ""
    _reindex_state["error"] = None

    # Gather IDs synchronously first
    conn = get_conn()
    c = conn.cursor()
    if mode == "full":
        c.execute("SELECT request_id FROM comments ORDER BY request_id")
    else:
        c.execute("""
            SELECT DISTINCT c.request_id FROM comments c
            WHERE NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.request_id = c.request_id
            )
            ORDER BY c.request_id
        """)
    ids = [r[0] for r in c.fetchall()]
    _reindex_state["total"] = len(ids)

    yield f"data: {json.dumps({'type': 'status', 'message': f'Starting reindex of {len(ids)} requests...'})}\n\n"

    # Spawn background thread for the actual blocking work
    thread = threading.Thread(target=_reindex_work, args=(mode, ids), daemon=True)
    thread.start()

    # Poll for progress and forward as SSE events
    last = 0
    while thread.is_alive():
        if _reindex_state["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': _reindex_state['error']})}\n\n"
            return
        cur = _reindex_state["current"]
        if cur != last:
            pct = int(cur / _reindex_state["total"] * 100) if _reindex_state["total"] > 0 else 0
            yield f"data: {json.dumps({'type': 'progress', 'percent': pct, 'count': cur, 'total': _reindex_state['total'], 'request_id': 0})}\n\n"
            last = cur
        await asyncio.sleep(0.5)

    yield f"data: {json.dumps({'type': 'done', 'message': _reindex_state['message']})}\n\n"


@router.post("/rag/reindex-all")
async def reindex_all():
    if _reindex_state["running"]:
        return JSONResponse({"error": "Reindex already running."}, status_code=400)
    return StreamingResponse(_reindex_sse("full"), media_type="text/event-stream")


@router.post("/rag/reindex-missing")
async def reindex_missing():
    if _reindex_state["running"]:
        return JSONResponse({"error": "Reindex already running."}, status_code=400)
    return StreamingResponse(_reindex_sse("missing"), media_type="text/event-stream")


@router.post("/rag/reindex-stop")
async def reindex_stop():
    _reindex_state["stop"] = True
    return JSONResponse({"message": "Reindex stop requested."})


@router.get("/rag/reindex-status")
async def reindex_status():
    return JSONResponse({
        "running": _reindex_state["running"],
        "current": _reindex_state["current"],
        "total": _reindex_state["total"],
        "message": _reindex_state["message"],
        "stop": _reindex_state["stop"],
    })


@router.post("/rag/search")
async def search_rag(req: SearchRequest):
    query_embedding = get_embedding(req.query)
    if not query_embedding or all(v == 0.0 for v in query_embedding):
        return JSONResponse({"error": "Failed to generate query embedding."}, status_code=500)

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT e.request_id, e.chunk_text, e.entity_type, e.embedding <-> %s::vector AS distance "
        "FROM embeddings e "
        "ORDER BY distance "
        "LIMIT %s",
        (json.dumps(query_embedding), req.top_k),
    )

    results = []
    for row in c.fetchall():
        results.append({
            "request_id": row[0],
            "chunk_text": row[1][:500],
            "entity_type": row[2] or "Request",
            "distance": float(row[3]),
        })

    return JSONResponse({"results": results, "query": req.query})


@router.post("/rag/ask")
async def ask_rag(req: AskRequest):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM embeddings")
    has_embeddings = c.fetchone()[0] > 0

    sources = []

    if has_embeddings:
        query_embedding = get_embedding(req.query)
        if not query_embedding or all(v == 0.0 for v in query_embedding):
            return JSONResponse({"error": "Failed to generate query embedding."}, status_code=500)

        if req.client:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e "
                "INNER JOIN request_custom_fields rcf ON e.request_id = rcf.request_id "
                "WHERE rcf.client ILIKE %s "
                "ORDER BY distance "
                "LIMIT %s",
                (json.dumps(query_embedding), f"%{req.client}%", req.top_k * 2),
            )
        else:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e "
                "ORDER BY distance "
                "LIMIT %s",
                (json.dumps(query_embedding), req.top_k * 2),
            )

        rows = c.fetchall()
        # Group by request_id
        chunks_by_id: dict[int, list] = {}
        for row in rows:
            rid = row[0]
            if rid not in chunks_by_id:
                chunks_by_id[rid] = []
            chunks_by_id[rid].append({
                "chunk_text": row[1],
                "entity_type": row[2] or "Request",
                "chunk_type": row[3],
                "distance": float(row[4]),
            })

        from database import get_entity_data as _get_ed, get_relations as _get_rel
        context_parts = []
        for rid, chunks in list(chunks_by_id.items())[:req.top_k]:
            ed = _get_ed(rid)
            if ed:
                et = ed.get("entity_type") or chunks[0]["entity_type"]
                state = ed.get("entity_state") or ""
                client = ed.get("client") or ""
                product = ed.get("product") or ""
                meta_line = f"[{et} #{rid}]"
                details = []
                if state:
                    details.append(f"State: {state}")
                if client:
                    details.append(f"Client: {client}")
                if product:
                    details.append(f"Product: {product}")
                if details:
                    meta_line += " | " + " | ".join(details)

                rels = _get_rel(rid)
                if rels:
                    rel_texts = []
                    for rel in rels[:5]:
                        rt = rel.get("related_entity_type") or "?"
                        rn = rel.get("related_entity_name") or rel.get("related_entity_id") or ""
                        rel_texts.append(f"{rt} #{rn}")
                    meta_line += " | Related: " + ", ".join(rel_texts)

                context_parts.append(meta_line)
                sources.append({"id": rid, "type": et, "state": state})
            else:
                et = chunks[0]["entity_type"]
                context_parts.append(f"[{et} #{rid}]")
                sources.append({"id": rid, "type": et, "state": ""})

            # Group chunks under the entity heading
            comment_chunks = [c for c in chunks if c["chunk_type"] in ("comment", "summary")]
            if comment_chunks:
                context_parts.append("")
                for cc in comment_chunks[:5]:
                    label = "Summary" if cc["chunk_type"] == "summary" else "Comment"
                    context_parts.append(f"  {label}: {cc['chunk_text'][:600]}")
                context_parts.append("")

        context = "\n".join(context_parts)
    else:
        from database import search_and_fetch_full
        kw_results = search_and_fetch_full(req.query, limit=5)
        context_parts = []
        for r in kw_results:
            et = r.get('entity_type', 'Request')
            context_parts.append(f"[{et} #{r['request_id']}]")
            sources.append({"id": r["request_id"], "type": et, "state": ""})
            comments_text = " ".join(
                cm.get("text", "") for cm in (r.get("comments") or [])
            )[:1000]
            if comments_text:
                context_parts.append(f"  Comment: {comments_text}")
            context_parts.append("")
        context = "\n".join(context_parts)

    prompt = (
        "You are a support ticket knowledge base. Each ticket shows its metadata "
        "(state, client, product) followed by relevant comments. Use this as evidence "
        "for your answer. If the available information is insufficient, say what you "
        "know and what's missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\n"
        "Answer concisely based on the context."
    )

    try:
        cfg.initialise_llm()
        from llm_providers import LLMClient
        answer = LLMClient.generate(prompt, temperature=0.3)
        return JSONResponse({
            "answer": answer,
            "sources": sources[:req.top_k],
            "mode": "rag" if has_embeddings else "keyword",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/rag/find-fixes")
async def find_fixes(req: FindFixesRequest):
    _ensure_chunk_type_column()
    comments, fetched_at = get_cached_comments(req.request_id)
    if not comments:
        return JSONResponse({"error": "Request ID not in cache."}, status_code=404)

    from shared.analysis import deduplicate_comment_dicts
    texts = deduplicate_comment_dicts(comments)
    from database import get_summary
    summary, _ = get_summary(req.request_id)
    if summary:
        texts.append(summary)
    combined = "\n".join(texts)[:5000]

    query_embedding = get_embedding(combined)
    if not query_embedding or all(v == 0.0 for v in query_embedding):
        return JSONResponse({"error": "Failed to generate query embedding."}, status_code=500)

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT e.request_id, e.chunk_text, e.entity_type, e.embedding <-> %s::vector AS distance, e.chunk_type "
        "FROM embeddings e "
        "WHERE e.request_id != %s "
        "ORDER BY distance "
        "LIMIT %s",
        (json.dumps(query_embedding), req.request_id, req.top_k),
    )
    results = []
    for row in c.fetchall():
        results.append({
            "request_id": row[0],
            "chunk_text": row[1][:500],
            "entity_type": row[2] or "Request",
            "distance": float(row[3]),
            "chunk_type": row[4],
        })

    synthesis = ""
    if results:
        context_parts = []
        for r in results[:5]:
            et_r = r.get('entity_type', 'Request')
            context_parts.append(f"[{et_r} #{r['request_id']}]: {r['chunk_text'][:1000]}")
        context = "\n".join(context_parts)
        synth_prompt = (
            "These are resolved support tickets similar to a current issue. "
            "Extract the resolution steps, workarounds, or fixes from each. "
            "If the same fix appears in multiple tickets, note it. "
            "Format as clear instructions.\n\n"
            f"Similar tickets:\n{context}"
        )
        try:
            cfg.initialise_llm()
            from shared.llm_providers import LLMClient
            synthesis = LLMClient.generate(synth_prompt, temperature=0.2)
        except Exception as e:
            synthesis = f"Could not generate synthesis: {e}"

    return JSONResponse({
        "synthesis": synthesis,
        "source_tickets": results,
    })
