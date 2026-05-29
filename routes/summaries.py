import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from database import (
    get_all_summaries, get_summaries_page, get_summary_count,
    get_summary, delete_summary, get_cached_entity_type,
)

logger = logging.getLogger(__name__)
router = APIRouter()
PAGE_SIZE = 50


@router.get("/summaries", response_class=HTMLResponse)
async def summaries_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("summaries.html", {"request": request})


@router.get("/summaries/list")
async def list_summaries():
    total = get_summary_count()
    page = get_summaries_page(PAGE_SIZE, 0)
    html = ""
    for s in page:
        et = s.get("entity_type", "Request")
        html += f'<div class="summary-item" hx-get="/summaries/{s["id"]}" hx-target="#summary-detail" hx-trigger="click">{et} #{s["id"]} - {s["created"][:10]}</div>\n'
    if not html:
        html = '<p class="status-text">No summaries saved.</p>'
    return HTMLResponse(html)


@router.get("/summaries/{request_id}")
async def get_summary_detail(request_id: int):
    text, created = get_summary(request_id)
    if text is None:
        return JSONResponse({"error": "Not found."}, status_code=404)
    entity_type = get_cached_entity_type(request_id) or "Request"
    html = f"<pre>{entity_type} #{request_id}\nCreated: {created}\n{'='*50}\n\n{text or '(empty)'}</pre>"
    return HTMLResponse(html)


class SearchRequest(BaseModel):
    query: str = ""


@router.post("/summaries/search")
async def search_summaries(req: SearchRequest):
    if not req.query.strip():
        return list_summaries()
    all_s = get_all_summaries()
    filtered = [s for s in all_s if str(s["id"]) == req.query.strip()]
    html = ""
    for s in filtered:
        et = s.get("entity_type", "Request")
        html += f'<div class="summary-item" hx-get="/summaries/{s["id"]}" hx-target="#summary-detail" hx-trigger="click">{et} #{s["id"]} - {s["created"][:10]}</div>\n'
    if not html:
        html = '<p class="status-text">No results found.</p>'
    return HTMLResponse(html)


@router.delete("/summaries/{request_id}")
async def delete_summary_route(request_id: int):
    delete_summary(request_id)
    return JSONResponse({"message": "Deleted."})
