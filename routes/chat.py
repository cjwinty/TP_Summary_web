import json
import logging
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from database import (
    save_chat_message,
    get_chat_history,
    get_conn,
)

from shared import config as cfg

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatSendRequest(BaseModel):
    session_id: str
    message: str
    client: Optional[str] = None


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    from jinja_env import templates
    session_id = str(uuid.uuid4())
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "session_id": session_id},
    )


def _format_grouped_context(rows: list, top_k: int) -> tuple[str, list]:
    """Group embedding rows by request_id, enrich with entity_data + relations, return (context_str, sources)."""
    from database import get_entity_data as _get_ed, get_relations as _get_rel

    chunks_by_id: dict[int, list] = {}
    for row in rows:
        rid = row[0]
        if rid not in chunks_by_id:
            chunks_by_id[rid] = []
        chunks_by_id[rid].append({
            "chunk_text": row[1],
            "entity_type": row[2] or "Request",
            "chunk_type": row[3] if len(row) > 3 else "comment",
            "distance": float(row[4]) if len(row) > 4 else 0.0,
        })

    context_parts = []
    sources = []

    for rid, chunks in list(chunks_by_id.items())[:top_k]:
        ed = _get_ed(rid)

        if ed:
            et = ed.get("entity_type") or chunks[0]["entity_type"]
            state = ed.get("entity_state", "")
            client = ed.get("client", "")
            product = ed.get("product", "")
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
                    rn = rel.get("related_entity_name") or str(rel.get("related_entity_id", ""))
                    rel_texts.append(f"{rt} #{rn}")
                meta_line += " | Related: " + ", ".join(rel_texts)

            context_parts.append(meta_line)
            sources.append({"id": rid, "type": et, "state": state})
        else:
            et = chunks[0]["entity_type"]
            context_parts.append(f"[{et} #{rid}]")
            sources.append({"id": rid, "type": et, "state": ""})

        comment_chunks = [c for c in chunks if c["chunk_type"] in ("comment", "summary")]
        if comment_chunks:
            context_parts.append("")
            for cc in comment_chunks[:5]:
                label = "Summary" if cc["chunk_type"] == "summary" else "Comment"
                context_parts.append(f"  {label}: {cc['chunk_text'][:600]}")
            context_parts.append("")

    return "\n".join(context_parts), sources


@router.post("/chat/send")
async def chat_send(req: ChatSendRequest):
    cfg.initialise_llm()
    history = get_chat_history(req.session_id, limit=12)

    query_embedding = None
    try:
        from shared.llm_providers import LLMClient
        query_embedding = LLMClient.generate_embedding(req.message)
    except Exception:
        pass

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM embeddings")
    has_embeddings = c.fetchone()[0] > 0

    context = ""
    sources = []
    top_k = 10

    if has_embeddings and query_embedding and any(v != 0.0 for v in query_embedding):
        if req.client:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e "
                "INNER JOIN request_custom_fields rcf ON e.request_id = rcf.request_id "
                "WHERE rcf.client ILIKE %s "
                "ORDER BY distance LIMIT %s",
                (json.dumps(query_embedding), f"%{req.client}%", top_k),
            )
        else:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e ORDER BY distance LIMIT %s",
                (json.dumps(query_embedding), top_k),
            )
        context, sources = _format_grouped_context(c.fetchall(), 5)
    else:
        from database import search_and_fetch_full
        kw_results = search_and_fetch_full(req.message, limit=5)
        kw_rows = []
        for r in kw_results:
            kw_rows.append((
                r["request_id"],
                " ".join(cm.get("text", "") for cm in (r.get("comments") or []))[:1000],
                r.get("entity_type", "Request"),
                "comment",
                0.0,
            ))
        if kw_rows:
            context, sources = _format_grouped_context(kw_rows, 5)

    history_text = ""
    for h in history[-6:]:
        role = "User" if h["role"] == "user" else "Assistant"
        history_text += f"{role}: {h['content']}\n"

    prompt = (
        "You are a support knowledge base assistant. Each ticket shows its metadata "
        "(state, client, product, custom fields) followed by relevant comments. "
        "Use this as evidence for your answer. If the available information is "
        "insufficient, say what you know and what's missing.\n\n"
    )
    if history_text:
        prompt += f"Conversation history:\n{history_text}\n\n"
    if context:
        prompt += f"Relevant ticket context:\n{context}\n"
    prompt += f"Question: {req.message}"

    try:
        from shared.llm_providers import LLMClient
        answer = LLMClient.generate(prompt, temperature=0.3)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    save_chat_message(req.session_id, "user", req.message)
    save_chat_message(req.session_id, "assistant", answer)

    return JSONResponse({
        "reply": answer,
        "sources": sources[:5],
        "mode": "rag" if has_embeddings else "keyword",
    })
