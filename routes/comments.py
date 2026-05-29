import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from database import get_cached_comments, get_cached_entity_type, get_relations as db_get_relations

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/comments", response_class=HTMLResponse)
async def comments_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("comments.html", {"request": request})


@router.get("/comments/{request_id}")
async def get_comments(request_id: int, sort: str = "desc"):
    comments, fetched_at = get_cached_comments(request_id)
    if not comments:
        return JSONResponse({"error": f"ID {request_id} not in cache."}, status_code=404)

    sorted_comments = sorted(
        comments,
        key=lambda x: x.get("date") or "",
        reverse=(sort == "desc"),
    )

    from shared.windows_utils import parse_dotnet_date
    from shared.analysis import deduplicate_text

    entity_type = get_cached_entity_type(request_id) or "Request"
    html = f'<div class="comment-header">{entity_type} #{request_id} | {len(sorted_comments)} comments | Cached: {fetched_at}</div>\n'
    for i, comment in enumerate(sorted_comments, 1):
        text = comment.get("text", "")
        date = comment.get("date", "Unknown date")
        if date and date != "Unknown date":
            date = parse_dotnet_date(date)
        display_text = deduplicate_text(text)
        html += f'<div class="comment-card"><strong>[{i}] {date}</strong>\n<pre>{display_text}</pre></div>\n'

    return HTMLResponse(html)


@router.get("/entity/{entity_id}/relations")
async def entity_relations(entity_id: int):
    from database import get_relations as _get_relations
    relations = _get_relations(entity_id)
    return JSONResponse(relations)
