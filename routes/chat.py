import json
import logging
import re
import threading
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from database import (
    save_chat_message,
    get_chat_history,
    get_conn,
    get_distinct_filter_options,
    get_entity_data,
    get_cached_comments,
    _build_metadata_blob,
)

from shared import config as cfg

logger = logging.getLogger(__name__)
router = APIRouter()

# Per-session state for entity exclusion across turns
# Structure: {session_id: {"turn": int, "seen_ids_per_turn": {int: set[int]}}}
chat_session_state: dict[str, dict] = {}
_session_lock = threading.Lock()

EXCLUSION_WINDOW_TURNS = 3

REQUERY_PROMPT = (
    "You are a search query rewriter. Your job is to convert a "
    "follow-up question into a standalone search query for a "
    "support ticket database.\n\n"
    "Rules:\n"
    "1. Replace pronouns (it, that, this, they) with the specific "
    "terms they refer to in the conversation\n"
    "2. Keep it concise \u2014 10 to 30 words, a single sentence\n"
    "3. Return ONLY the rewritten query, nothing else\n"
    "4. Never add information not present in the history or question\n"
    "5. Preserve any ticket IDs (e.g. #12345 or id 12345) in the rewritten query\n\n"
    "Conversation:\n"
    "User: {prev_q}\n"
    "Assistant: {prev_a}\n\n"
    "Follow-up: {current_q}\n\n"
    "Rewritten query:"
)


class ChatSendRequest(BaseModel):
    session_id: str
    message: str
    client: Optional[str] = None
    product: Optional[str] = None
    project: Optional[str] = None
    entity_type: Optional[str] = None
    entity_state: Optional[str] = None


class ChatSessionState(BaseModel):
    session_id: str


@router.get("/chat/filter-options")
async def chat_filter_options():
    return JSONResponse(get_distinct_filter_options())


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    from jinja_env import templates
    session_id = str(uuid.uuid4())
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "session_id": session_id},
    )


@router.post("/chat/reset-direction")
async def reset_direction(req: ChatSessionState):
    with _session_lock:
        state = chat_session_state.get(req.session_id)
        if state:
            state["turn"] = 0
            state["seen_ids_per_turn"] = {}
            state["focus_id"] = None
    return JSONResponse({"ok": True})


def _get_session_state(session_id: str) -> dict:
    with _session_lock:
        if session_id not in chat_session_state:
            chat_session_state[session_id] = {
                "turn": 0,
                "seen_ids_per_turn": {},
                "focus_id": None,
            }
        return chat_session_state[session_id]


def _get_exclude_ids(state: dict) -> set[int]:
    seen_ids_per_turn = state.get("seen_ids_per_turn", {})
    current_turn = state.get("turn", 0)
    start_turn = max(0, current_turn - EXCLUSION_WINDOW_TURNS)
    excluded = set()
    for t in range(start_turn, current_turn):
        excluded.update(seen_ids_per_turn.get(t, set()))
    return excluded


def _parse_mentioned_ids(text: str) -> set[int]:
    ids = set()
    ids.update(int(m) for m in re.findall(r'#(\d+)', text))
    ids.update(int(m) for m in re.findall(r'(?i)id\s+(\d+)', text))
    return ids


def _fetch_direct_entity_context(conn, entity_ids: set[int]) -> tuple[str, list[dict]]:
    if not entity_ids:
        return "", []
    context_parts = []
    sources = []
    for rid in sorted(entity_ids):
        ed = get_entity_data(rid)
        if not ed:
            continue
        et = ed.get("entity_type", "Request")
        state = ed.get("entity_state", "")
        blob = _build_metadata_blob(rid, et, ed)
        header = blob or f"[{et} #{rid}]"
        entity_lines = [header]
        comments_data = get_cached_comments(rid)
        if comments_data:
            comments, _ = comments_data
            if comments:
                for cm in comments[:10]:
                    text = (cm.get("text") or "").strip()[:1200]
                    if text:
                        entity_lines.append(f"  Comment: {text}")
        if context_parts:
            context_parts.append("")
        context_parts.extend(entity_lines)
        sources.append({"id": rid, "type": et, "state": state})
    return "\n".join(context_parts), sources


def _build_requery_text(last_msgs: list[dict]) -> str:
    if len(last_msgs) < 3:
        return ""
    prev_q = last_msgs[-3]["content"]
    prev_a = last_msgs[-2]["content"]
    current_q = last_msgs[-1]["content"]
    return REQUERY_PROMPT.format(prev_q=prev_q, prev_a=prev_a, current_q=current_q)


def _get_last_three(session_id: str) -> list[dict]:
    history = get_chat_history(session_id, limit=12)
    return history[-3:] if len(history) >= 3 else history


def _rewrite_query(session_id: str, message: str, history: list[dict]) -> str:
    last = history[-3:] if len(history) >= 3 else history
    if len(last) < 3:
        return message

    prompt = _build_requery_text(last)
    if not prompt:
        return message

    try:
        from shared.llm_providers import LLMClient
        rewritten = LLMClient.generate(prompt, temperature=0.0, max_tokens=60)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten:
            logger.info("Re-query rewrite: %r -> %r", message, rewritten)
            return rewritten
    except Exception as e:
        logger.warning("Re-query rewrite failed: %s", e)

    fallback_texts = []
    for m in last:
        role = "User" if m["role"] == "user" else "Assistant"
        fallback_texts.append(f"{role}: {m['content']}")
    return " | ".join(fallback_texts)


@router.post("/chat/send")
async def chat_send(req: ChatSendRequest):
    cfg.initialise_llm()
    state = _get_session_state(req.session_id)
    history = get_chat_history(req.session_id, limit=12)

    mentioned_ids = _parse_mentioned_ids(req.message)
    exclude_ids = _get_exclude_ids(state)
    exclude_ids -= mentioned_ids

    if mentioned_ids:
        logger.info("Mentioned IDs in message: %s", mentioned_ids)

    is_first_turn = len(history) < 2

    if is_first_turn:
        search_query = req.message
    else:
        search_query = _rewrite_query(req.session_id, req.message, history)

    query_embedding = None
    try:
        from shared.llm_providers import LLMClient
        query_embedding = LLMClient.generate_embedding(search_query)
    except Exception:
        pass

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM embeddings")
    has_embeddings = c.fetchone()[0] > 0

    context = ""
    sources = []

    filter_map = {
        "client": "ed.client",
        "product": "ed.product",
        "project": "ed.project_name",
        "entity_type": "ed.entity_type",
        "entity_state": "ed.entity_state",
    }
    filter_clauses = []
    filter_params = []
    for field, db_col in filter_map.items():
        val = getattr(req, field, None)
        if val:
            filter_clauses.append(f"{db_col} = %s")
            filter_params.append(val)

    if has_embeddings and query_embedding and any(v != 0.0 for v in query_embedding):
        from shared.retrieval import vector_search
        context, sources = vector_search(
            query_embedding=query_embedding,
            max_entities=10,
            chunk_char_limit=1200,
            token_budget=30000,
            exclude_ids=exclude_ids,
            filter_clauses=filter_clauses,
            filter_params=filter_params,
        )
    else:
        from database import search_and_fetch_full, get_entity_data as _get_ed
        kw_results = search_and_fetch_full(req.message, limit=25)
        if filter_clauses:
            filtered = []
            for r in kw_results:
                ed = _get_ed(r["request_id"])
                if not ed:
                    continue
                if all(
                    (ed.get(col) or "") == val
                    for col, val in [
                        ("client", req.client),
                        ("product", req.product),
                        ("project_name", req.project),
                        ("entity_type", req.entity_type),
                        ("entity_state", req.entity_state),
                    ]
                    if val
                ):
                    filtered.append(r)
            kw_results = filtered
        context_parts = []
        for r in kw_results[:10]:
            rid = r["request_id"]
            if rid in exclude_ids:
                continue
            et = r.get("entity_type", "Request")
            context_parts.append(f"[{et} #{rid}]")
            sources.append({"id": rid, "type": et, "state": ""})
            comments_text = " ".join(
                cm.get("text", "") for cm in (r.get("comments") or [])
            )[:1200]
            if comments_text:
                context_parts.append(f"  Comment: {comments_text}")
            context_parts.append("")
        context = "\n".join(context_parts)

    if mentioned_ids:
        existing_ids = {s["id"] for s in sources}
        missing_ids = mentioned_ids - existing_ids
        if missing_ids:
            direct_ctx, direct_src = _fetch_direct_entity_context(conn, missing_ids)
            if direct_ctx:
                context = direct_ctx + "\n\n" + context if context else direct_ctx
                sources = direct_src + sources

    focus_id = state.get("focus_id")
    if focus_id is not None:
        existing_ids = {s["id"] for s in sources}
        if focus_id not in existing_ids:
            focus_ctx, focus_src = _fetch_direct_entity_context(conn, {focus_id})
            if focus_ctx:
                context = focus_ctx + "\n\n" + context if context else focus_ctx
                sources = focus_src + sources

    history_text = ""
    for h in history[-10:]:
        role = "User" if h["role"] == "user" else "Assistant"
        history_text += f"{role}: {h['content']}\n"

    prompt = (
        "You are a support knowledge base assistant. Each ticket shows its full ticket profile "
        "(state, project, client, product, version, custom fields, description) followed by relevant comments. "
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

    with _session_lock:
        state["turn"] += 1
        seen_ids = {s["id"] for s in sources}
        state["seen_ids_per_turn"][state["turn"]] = seen_ids
        state["focus_id"] = sources[0]["id"] if sources else None

    save_chat_message(req.session_id, "user", req.message)
    save_chat_message(req.session_id, "assistant", answer)

    return JSONResponse({
        "reply": answer,
        "sources": sources[:5],
        "mode": "rag" if has_embeddings else "keyword",
    })
