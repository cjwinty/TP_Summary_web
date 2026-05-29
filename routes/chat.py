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


@router.post("/chat/send")
async def chat_send(req: ChatSendRequest):
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

    if has_embeddings and query_embedding and any(v != 0.0 for v in query_embedding):
        if req.client:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e "
                "INNER JOIN request_custom_fields rcf ON e.request_id = rcf.request_id "
                "WHERE rcf.client ILIKE %s "
                "ORDER BY distance LIMIT %s",
                (json.dumps(query_embedding), f"%{req.client}%", 5),
            )
        else:
            c.execute(
                "SELECT e.request_id, e.chunk_text, e.entity_type, e.embedding <-> %s::vector AS distance "
                "FROM embeddings e ORDER BY distance LIMIT %s",
                (json.dumps(query_embedding), 5),
            )
        for row in c.fetchall():
            et = row[2] or "Request"
            context += f"[{et} #{row[0]}] {row[1][:1000]}\n\n"
            sources.append(row[0])
    else:
        from database import search_and_fetch_full
        kw_results = search_and_fetch_full(req.message, limit=5)
        for r in kw_results:
            et = r.get('entity_type', 'Request')
            text = " ".join(cm.get("text", "") for cm in (r.get("comments") or []))[:1000]
            context += f"[{et} #{r['request_id']}] {text}\n\n"
            sources.append(r["request_id"])

    history_text = ""
    for h in history[-6:]:
        role = "User" if h["role"] == "user" else "Assistant"
        history_text += f"{role}: {h['content']}\n"

    prompt = (
        "You are a support knowledge base assistant. Answer the user's question "
        "based ONLY on the provided ticket context. If the context doesn't contain "
        "enough information to answer, say so clearly. Be concise and professional.\n\n"
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
