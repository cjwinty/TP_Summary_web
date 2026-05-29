import json
import logging
import threading
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from shared import config as cfg
from database import (
    get_all_prompts, save_prompt, get_cache_counts, delete_all_summaries,
    get_max_min_request_id, check_database_health, optimise_database, get_conn,
    clear_entity_data, clear_entity_relations, clear_all_chat_history,
    clear_all_cached_data,
)
from shared.api import refresh_entity_metadata, get_all_projects, get_comments as api_get_comments
from database import auto_index_request_web, delete_embeddings

logger = logging.getLogger(__name__)
router = APIRouter()


class LLMConfigRequest(BaseModel):
    provider_type: str
    local_provider: str = "Ollama"
    model: str = ""
    host: str = "localhost"
    cloud_type: str = "openai"
    endpoint: str = ""
    api_key: str = ""
    aws_region: str = "us-east-1"
    embedding_endpoint: str = ""
    embedding_model: str = ""


class PromptSaveRequest(BaseModel):
    name: str
    content: str


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "llm_provider_type": cfg.LLM_PROVIDER_TYPE,
        "local_provider": cfg.LOCAL_PROVIDER,
        "local_host": cfg.LOCAL_LLM_HOST,
        "local_model": cfg.OLLAMA_MODEL,
        "cloud_type": cfg.CLOUD_CONFIG["provider"],
        "cloud_endpoint": cfg.CLOUD_CONFIG["endpoint"],
        "cloud_model": cfg.CLOUD_CONFIG["model"],
        "aws_region": cfg.CLOUD_CONFIG["aws_region"],
        "embedding_endpoint": cfg._config.get("llm_embedding_endpoint", ""),
        "embedding_model": cfg.EMBEDDING_MODEL,
    })


@router.get("/settings/prompts")
async def list_prompts():
    prompts = get_all_prompts()
    html = ""
    for p in prompts:
        active = " (Active)" if p["is_active"] else ""
        html += f'<div class="prompt-item" data-name="{p["name"]}" hx-get="/settings/prompts/{p["name"]}" hx-target="#prompt-content" hx-trigger="click">{p["name"]}{active}</div>'
    if not html:
        html = "<p class='status-text'>No prompts configured.</p>"
    return HTMLResponse(html)


@router.get("/settings/prompts/{name}")
async def get_prompt_content(name: str):
    from database import get_prompt
    p = get_prompt(name)
    if p:
        return JSONResponse({"name": name, "content": p["content"]})
    return JSONResponse({"name": name, "content": ""})


@router.post("/settings/prompts")
async def save_prompt_route(req: PromptSaveRequest):
    save_prompt(req.name, req.content)
    return JSONResponse({"message": f"Prompt '{req.name}' saved."})


@router.post("/settings/prompts/reset/{name}")
async def reset_prompt(name: str):
    from database import DEFAULT_PROMPTS
    content = DEFAULT_PROMPTS.get(name, "")
    if content:
        save_prompt(name, content)
        return JSONResponse({"message": f"Prompt '{name}' reset to default."})
    return JSONResponse({"error": "No default found."}, status_code=404)


@router.post("/settings/llm")
async def save_llm_config(req: LLMConfigRequest):
    try:
        cfg.set_llm_provider_type(req.provider_type)
        if req.provider_type == "cloud":
            api_key = req.api_key or None
            if req.cloud_type == "bedrock":
                cfg.set_bedrock_config(req.aws_region, api_key, req.model)
            else:
                cfg.set_cloud_config(req.cloud_type, req.endpoint, api_key, req.model, embedding_endpoint=req.embedding_endpoint)
        else:
            cfg.set_local_provider(req.local_provider)
            cfg.set_local_host(req.host)
            cfg.set_ollama_model(req.model)
        cfg.set_embedding_model(req.embedding_model)
        return JSONResponse({"message": "LLM config saved."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/settings/test-connection")
async def test_llm_connection(req: LLMConfigRequest):
    from shared.llm_providers import LLMClient, LocalLLMProvider, CloudLLMProvider, LOCAL_PROVIDERS
    try:
        if req.provider_type == "cloud":
            api_key = req.api_key or cfg.CLOUD_CONFIG.get("api_key", "")
            if req.cloud_type == "bedrock":
                cloud_config = {"provider": "bedrock", "api_key": api_key, "aws_region": req.aws_region, "model": req.model}
            else:
                cloud_config = {"provider": "openai", "endpoint": req.endpoint, "api_key": api_key, "model": req.model, "verify": cfg.VERIFY_SSL, "embedding_endpoint": req.embedding_endpoint}
            provider = CloudLLMProvider(cloud_config)
        else:
            provider_config = LOCAL_PROVIDERS.get(req.local_provider, LOCAL_PROVIDERS["Ollama"])
            local_config = {"host": req.host, "port": provider_config["port"], "model": req.model, "timeout": 30, "provider_name": req.local_provider}
            provider = LocalLLMProvider(local_config)
        success, message = provider.test_connection()
        return JSONResponse({"message": message})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/settings/cache-stats")
async def cache_stats():
    counts = get_cache_counts()
    id_range = get_max_min_request_id()
    html = (
        f'<p class="status-text">'
        f'Comments: {counts["comments"]} | '
        f'Summaries: {counts["summaries"]} | '
        f'Entity Data: {counts["entity_data"]} | '
        f'Entity Relations: {counts["entity_relations"]} | '
        f'Custom Fields: {counts["custom_fields"]} | '
        f'Embeddings: {counts["embeddings"]} | '
        f'Chat History: {counts["chat_history"]}'
        f'</p>'
    )
    if id_range["min"] is not None:
        html += f'<p class="status-text">ID Range: {id_range["min"]} - {id_range["max"]}</p>'
    return HTMLResponse(html)


@router.post("/settings/clear-summaries")
async def clear_summaries():
    delete_all_summaries()
    return JSONResponse({"message": "All summaries cleared."})


@router.post("/settings/clear-entity-data")
async def clear_entity_data_route():
    clear_entity_data()
    return JSONResponse({"message": "All entity data cleared."})


@router.post("/settings/clear-entity-relations")
async def clear_entity_relations_route():
    clear_entity_relations()
    return JSONResponse({"message": "All entity relations cleared."})


@router.post("/settings/clear-chat-history")
async def clear_chat_history_route():
    clear_all_chat_history()
    return JSONResponse({"message": "All chat history cleared."})


@router.post("/settings/clear-all-cache")
async def clear_all_cache_route():
    clear_all_cached_data()
    return JSONResponse({"message": "All cached data cleared."})


@router.get("/settings/health")
async def health_check():
    health = check_database_health()
    return JSONResponse(health)


@router.post("/settings/optimise")
async def optimise():
    result = optimise_database()
    return JSONResponse(result)


# ── Metadata backfill ──

_backfill_state = {
    "running": False,
    "stop": False,
    "current": 0,
    "total": 0,
    "message": "",
    "error": None,
}


def _backfill_work(ids: list[int]):
    global _backfill_state
    count = 0
    for rid in ids:
        if _backfill_state["stop"]:
            _backfill_state["message"] = f"Stopped after {count} entities."
            break
        try:
            refresh_entity_metadata(rid)
            count += 1
        except Exception:
            pass
        _backfill_state["current"] = count
    _backfill_state["running"] = False
    if not _backfill_state["stop"]:
        _backfill_state["message"] = f"Backfilled metadata for {count} entities."


async def _backfill_sse():
    global _backfill_state
    _backfill_state["running"] = True
    _backfill_state["stop"] = False
    _backfill_state["current"] = 0
    _backfill_state["total"] = 0
    _backfill_state["message"] = ""
    _backfill_state["error"] = None

    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT c.request_id FROM comments c
        WHERE NOT EXISTS (
            SELECT 1 FROM entity_data e WHERE e.entity_id = c.request_id
        )
        ORDER BY c.request_id
    """)
    ids = [r[0] for r in c.fetchall()]
    _backfill_state["total"] = len(ids)

    if not ids:
        _backfill_state["running"] = False
        yield f"data: {json.dumps({'type': 'done', 'message': 'All cached entities already have metadata.'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'status', 'message': f'Backfilling metadata for {len(ids)} entities...'})}\n\n"

    thread = threading.Thread(target=_backfill_work, args=(ids,), daemon=True)
    thread.start()

    last = 0
    while thread.is_alive():
        if _backfill_state["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': _backfill_state['error']})}\n\n"
            return
        cur = _backfill_state["current"]
        if cur != last:
            pct = int(cur / _backfill_state["total"] * 100) if _backfill_state["total"] > 0 else 0
            yield f"data: {json.dumps({'type': 'progress', 'percent': pct, 'count': cur, 'total': _backfill_state['total']})}\n\n"
            last = cur
        import asyncio
        await asyncio.sleep(0.5)

    yield f"data: {json.dumps({'type': 'done', 'message': _backfill_state['message']})}\n\n"


@router.post("/settings/backfill-metadata")
async def backfill_metadata():
    if _backfill_state["running"]:
        return JSONResponse({"error": "Backfill already running."}, status_code=400)
    return StreamingResponse(_backfill_sse(), media_type="text/event-stream")


@router.get("/settings/backfill-status")
async def backfill_status():
    return JSONResponse({
        "running": _backfill_state["running"],
        "current": _backfill_state["current"],
        "total": _backfill_state["total"],
        "message": _backfill_state["message"],
    })


@router.post("/settings/backfill-stop")
async def backfill_stop():
    _backfill_state["stop"] = True
    return JSONResponse({"message": "Backfill stop requested."})


# ── Project name backfill ──

_project_name_state = {
    "running": False,
    "stop": False,
    "current": 0,
    "total": 0,
    "message": "",
    "error": None,
}


async def _project_name_sse():
    global _project_name_state
    _project_name_state["running"] = True
    _project_name_state["stop"] = False
    _project_name_state["current"] = 0
    _project_name_state["total"] = 0
    _project_name_state["message"] = ""
    _project_name_state["error"] = None

    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching projects from TP API...'})}\n\n"

        projects = get_all_projects()
        if not projects:
            _project_name_state["running"] = False
            yield f"data: {json.dumps({'type': 'done', 'message': 'No projects returned from API.'})}\n\n"
            return

        name_map = {}
        for p in projects:
            pid = p.get("id")
            name = p.get("name")
            if pid and name:
                try:
                    name_map[int(pid)] = name
                except ValueError:
                    pass

        if not name_map:
            _project_name_state["running"] = False
            yield f"data: {json.dumps({'type': 'done', 'message': 'No project names found.'})}\n\n"
            return

        _project_name_state["total"] = len(name_map)
        yield f"data: {json.dumps({'type': 'status', 'message': f'Resolving {len(name_map)} project names...'})}\n\n"

        conn = get_conn()
        c = conn.cursor()
        count = 0
        for pid, pname in name_map.items():
            if _project_name_state["stop"]:
                _project_name_state["running"] = False
                yield f"data: {json.dumps({'type': 'done', 'message': f'Stopped after {count} projects.'})}\n\n"
                return

            c.execute(
                "UPDATE entity_data SET project_name = %s WHERE project_id = %s AND (project_name IS NULL OR project_name = '')",
                (pname, pid),
            )
            conn.commit()
            count += 1
            _project_name_state["current"] = count
            pct = int(count / len(name_map) * 100)
            yield f"data: {json.dumps({'type': 'progress', 'percent': pct, 'count': count, 'total': len(name_map)})}\n\n"

        _project_name_state["running"] = False
        yield f"data: {json.dumps({'type': 'done', 'message': f'Resolved {count} project names.'})}\n\n"
    except Exception as e:
        logger.exception("Project name backfill error")
        _project_name_state["error"] = str(e)
        _project_name_state["running"] = False
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post("/settings/backfill-project-names")
async def backfill_project_names():
    if _project_name_state["running"]:
        return JSONResponse({"error": "Backfill already running."}, status_code=400)
    return StreamingResponse(_project_name_sse(), media_type="text/event-stream")


@router.get("/settings/backfill-project-names-status")
async def backfill_project_names_status():
    return JSONResponse({
        "running": _project_name_state["running"],
        "current": _project_name_state["current"],
        "total": _project_name_state["total"],
        "message": _project_name_state["message"],
    })


@router.post("/settings/backfill-project-names-stop")
async def backfill_project_names_stop():
    _project_name_state["stop"] = True
    return JSONResponse({"message": "Project name backfill stop requested."})


# ── Cache Range ──

_cache_running = False
_cache_stop = False


class CacheRangeRequest(BaseModel):
    start: int
    end: int
    mode: str = "smart"


@router.post("/settings/cache-range")
async def cache_range(req: CacheRangeRequest):
    global _cache_running, _cache_stop
    if _cache_running:
        return JSONResponse({"error": "Cache already running."}, status_code=400)

    _cache_running = True
    _cache_stop = False
    start, end = req.start, req.end
    mode = (req.mode or "smart").lower()

    async def event_stream():
        global _cache_running, _cache_stop
        total = end - start + 1
        count = 0
        skipped = 0

        if mode == "smart":
            conn = get_conn()
            c = conn.cursor()
            c.execute("SELECT request_id FROM comments WHERE request_id BETWEEN %s AND %s", (start, end))
            existing = {r[0] for r in c.fetchall()}
            missing = sorted(set(range(start, end + 1)) - existing)
            skipped = total - len(missing)
            c.execute("""
                SELECT c.request_id FROM comments c
                LEFT JOIN entity_data e ON c.request_id = e.entity_id
                WHERE c.request_id BETWEEN %s AND %s AND e.entity_id IS NULL
            """, (start, end))
            stale = {r[0] for r in c.fetchall()} - set(missing)
            stale_metadata = sorted(stale)
            if not missing and not stale_metadata:
                _cache_running = False
                yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'count': 0, 'total': total, 'skipped': skipped})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message': f'All {total} IDs already cached'})}\n\n"
                return
            for rid in missing:
                if _cache_stop:
                    break
                api_get_comments(rid, use_cache=False)
                refresh_entity_metadata(rid)
                try:
                    auto_index_request_web(rid)
                except Exception:
                    pass
                count += 1
                percent = int((count + skipped) / total * 100)
                yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'count': count, 'total': total, 'skipped': skipped})}\n\n"
            metadata_only_count = 0
            for rid in stale_metadata:
                if _cache_stop:
                    break
                refresh_entity_metadata(rid)
                metadata_only_count += 1
                count += 1
                percent = int((count + skipped) / total * 100)
                yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'count': count, 'total': total, 'skipped': skipped, 'metadata_only': metadata_only_count})}\n\n"
            _cache_running = False
            yield f"data: {json.dumps({'type': 'done', 'message': f'Cached {len(missing)} new, refreshed {metadata_only_count} metadata, skipped {skipped} existing ({total} total)'})}\n\n"
        else:  # force
            for rid in range(start, end + 1):
                if _cache_stop:
                    break
                delete_embeddings(rid)
                api_get_comments(rid, use_cache=False)
                refresh_entity_metadata(rid)
                try:
                    auto_index_request_web(rid)
                except Exception:
                    pass
                count += 1
                percent = int((count / total) * 100)
                yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'count': count, 'total': total})}\n\n"
            _cache_running = False
            yield f"data: {json.dumps({'type': 'done', 'message': f'Cached {count} requests'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/settings/cache-range-stop")
async def cache_range_stop():
    global _cache_stop
    _cache_stop = True
    return JSONResponse({"message": "Cache stopping..."})
