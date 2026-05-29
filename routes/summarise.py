import json
import logging
import threading
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from shared.config import PROJECT_NAME, initialise_llm, logger

from shared.api import get_comments as api_get_comments, refresh_entity_metadata

from shared.analysis import summarise_batch, deduplicate_comment_dicts

from database import (
    save_summary, get_summary_with_cache_time,
    delete_comments, get_all_cached_ids, get_conn,
    auto_index_request_web, delete_embeddings,
    get_cached_entity_type,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_cache_running = False
_cache_stop = False


class SummariseRequest(BaseModel):
    ids: str
    refresh: bool = False


class CacheUpdateRequest(BaseModel):
    ids: str
    mode: str = "smart"


class CacheRangeRequest(BaseModel):
    start: int
    end: int
    mode: str = "smart"


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "project_name": PROJECT_NAME or "Not Configured"},
    )


@router.post("/summarise")
async def summarise(req: SummariseRequest):
    ids_text = req.ids.strip()
    if not ids_text:
        return JSONResponse({"error": "Please enter request IDs."}, status_code=400)

    try:
        request_ids = [int(x.strip()) for x in ids_text.split(",")]
    except ValueError:
        return JSONResponse({"error": "Invalid ID format. Use comma-separated integers."}, status_code=400)

    if len(request_ids) > 500:
        return JSONResponse({"error": "Too many IDs. Maximum is 500."}, status_code=400)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(request_ids)} requests...'})}\n\n"

        all_comments = []
        request_ids_to_summarise = []
        cached_count = fetched_count = 0
        error_ids = []
        empty_ids = []

        for i, request_id in enumerate(request_ids):
            if i % 10 == 0 and i > 0:
                yield f"data: {json.dumps({'type': 'status', 'message': f'Fetching comments... {i}/{len(request_ids)}'})}\n\n"

            use_cache = not req.refresh
            comments, fetched_at, fresh = api_get_comments(request_id, use_cache=use_cache)
            refresh_entity_metadata(request_id)
            if fresh:
                fetched_count += 1
            else:
                cached_count += 1

            if comments is None:
                error_ids.append(request_id)
                continue
            if not comments:
                empty_ids.append(request_id)
                continue

            source = f" (cached from {fetched_at[:10]})" if fetched_at else " (fresh)"
            summary_data = get_summary_with_cache_time(request_id)
            cached_summary = summary_data.get("summary", "") if summary_data else ""
            has_error = cached_summary.startswith("Error") or "API error" in cached_summary

            et = get_cached_entity_type(request_id) or "Request"
            if summary_data and not fresh and summary_data.get("fetched_at") == fetched_at and not has_error:
                yield f"data: {json.dumps({'type': 'result', 'text': f'[{et} #{request_id}]{source} (cached summary)\n{cached_summary}\n\n'})}\n\n"
            else:
                unique_lines = deduplicate_comment_dicts(comments)
                if unique_lines:
                    all_comments.append(f"[{et} #{request_id}]{source}:\n" + "\n".join(unique_lines))
                    request_ids_to_summarise.append(request_id)

        if not all_comments:
            msg = ""
            if error_ids:
                msg = "No comments could be fetched from the API.\n" + "Check tp_query_error.log for details.\n"
            if empty_ids:
                id_list = ", ".join(str(i) for i in empty_ids)
                msg += f"No comments found for request ID(s): {id_list}\n"
            yield f"data: {json.dumps({'type': 'error', 'text': msg})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'result', 'text': f'Using {cached_count} cached, {fetched_count} fresh\n\n'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': f'Summarising {len(all_comments)} requests...'})}\n\n"

        try:
            initialise_llm()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'LLM initialisation error: {e}\n'})}\n\n"
            return

        summaries = summarise_batch(all_comments)

        yield f"data: {json.dumps({'type': 'result', 'text': '=' * 60 + '\nSUPPORT CALL SUMMARY\n' + '=' * 60 + '\n\n'})}\n\n"

        for i, summary in enumerate(summaries):
            yield f"data: {json.dumps({'type': 'result', 'text': summary + '\n\n'})}\n\n"
            rid = request_ids_to_summarise[i]
            save_summary(rid, summary)
            try:
                auto_index_request_web(rid, index_summary=True)
            except Exception:
                pass

        yield f"data: {json.dumps({'type': 'status', 'message': 'Done! Summaries saved.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/cache/update")
async def update_cache(req: CacheUpdateRequest):
    ids_text = req.ids.strip()
    if not ids_text:
        return JSONResponse({"error": "No IDs provided."}, status_code=400)

    try:
        request_ids = [int(x.strip()) for x in ids_text.split(",")]
    except ValueError:
        return JSONResponse({"error": "Invalid ID format."}, status_code=400)

    mode = (req.mode or "smart").lower()

    if mode == "smart":
        conn = get_conn()
        c = conn.cursor()
        placeholders = ",".join("%s" for _ in request_ids)
        c.execute(f"SELECT request_id FROM comments WHERE request_id IN ({placeholders})", request_ids)
        existing = {r[0] for r in c.fetchall()}
        proc = [rid for rid in request_ids if rid not in existing]
    else:
        proc = request_ids

    def run():
        for rid in proc:
            delete_comments(rid)
            delete_embeddings(rid)
            api_get_comments(rid, use_cache=False)
            refresh_entity_metadata(rid)
            try:
                auto_index_request_web(rid)
            except Exception:
                pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    detail = f" ({mode} mode)" if mode == "force" else ""
    return JSONResponse({"message": f"Updating cache for {len(proc)} requests{detail}..."})


@router.post("/cache/range")
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
            if not missing:
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
            _cache_running = False
            yield f"data: {json.dumps({'type': 'done', 'message': f'Cached {count} new, skipped {skipped} existing ({total} total)'})}\n\n"
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


@router.post("/cache/stop")
async def stop_cache():
    global _cache_stop
    _cache_stop = True
    return JSONResponse({"message": "Cache stopping..."})


@router.get("/cached-ids")
async def cached_ids():
    ids = get_all_cached_ids()
    return JSONResponse({"ids": ids})
