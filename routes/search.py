import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from database import (
    search_cached_comments, search_summaries, search_and_fetch_full,
)
from shared.analysis import refine_search_query, summarise_search_results

logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    refine: bool = True
    skip_summary: bool = False
    full_fetch: bool = False
    custom_prompt: str = ""
    date_from: str = ""
    date_to: str = ""
    filters: list = []


class FilterDef(BaseModel):
    field: str = ""
    value: str = ""
    logic: str = "AND"


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("search.html", {"request": request})


@router.post("/search")
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        return JSONResponse({"results": "", "summary": "", "status": "No query provided."})

    search_terms = [query]
    if req.refine and not req.full_fetch:
        try:
            from shared import config as cfg
            cfg.initialise_llm()
            refined = refine_search_query(query)
            if refined and len(refined) > 1:
                search_terms = refined
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")

    date_filter = {}
    if req.date_from:
        date_filter["start_date"] = req.date_from
    if req.date_to:
        date_filter["end_date"] = req.date_to
    if not date_filter:
        date_filter = None

    custom_field_filter = None
    if req.filters:
        if len(req.filters) == 1:
            custom_field_filter = {"field_name": req.filters[0]["field"], "field_value": req.filters[0]["value"]}
        else:
            custom_field_filter = {
                "filters": [{"field_name": f["field"], "field_value": f["value"]} for f in req.filters],
                "logic": req.filters[-1].get("logic", "AND"),
            }

    all_matches = []

    if req.full_fetch:
        for term in search_terms:
            matches = search_and_fetch_full(term, custom_field_filter=custom_field_filter, date_filter=date_filter)
            all_matches.extend(matches)
        seen = set()
        unique = []
        for m in all_matches:
            if m["request_id"] not in seen:
                seen.add(m["request_id"])
                unique.append(m)
        unique.sort(key=lambda x: x["request_id"])
        all_matches = unique

        result_text = f"Found {len(all_matches)} matching IDs for: {', '.join(search_terms)}\n(Full Record Retrieval)\n\n"
        for match in all_matches:
            et = match.get('entity_type', 'Request')
            result_text += "=" * 60 + "\n"
            result_text += f"{et} #{match['request_id']} | Product: {match.get('product', 'N/A')}\n"
            result_text += "=" * 60 + "\n"
            for j, c in enumerate(match.get("comments", []), 1):
                date = c.get("date", "Unknown")
                result_text += f"--- COMMENT {j} ({date}) ---\n{c.get('text', '')}\n\n"
            if match.get("summary"):
                result_text += f"EXISTING SUMMARY:\n{match['summary']}\n"
            result_text += "\n"
    else:
        for term in search_terms:
            all_matches.extend(search_cached_comments(term, custom_field_filter=custom_field_filter, date_filter=date_filter))
            all_matches.extend(search_summaries(term, custom_field_filter=custom_field_filter, date_filter=date_filter))

        seen = set()
        unique = []
        for m in all_matches:
            key = (m["request_id"], m["text"][:100])
            if key not in seen:
                seen.add(key)
                unique.append(m)
        unique.sort(key=lambda x: x["request_id"])
        all_matches = unique

        result_text = f"Found {len(all_matches)} matches for: {', '.join(search_terms)}\n\n"
        for match in all_matches:
            et = match.get('entity_type', 'Request')
            result_text += f"[{et} #{match['request_id']}] ({match['source']})\n"
            lines = match["text"].split("\n")
            for line in lines[:20]:
                prefix = "  >>> " if query.lower() in line.lower() else "  "
                result_text += prefix + line + "\n"
            result_text += "=" * 60 + "\n\n"

    summary_text = ""
    if all_matches and not req.skip_summary:
        try:
            from shared import config as cfg
            cfg.initialise_llm()
            llm_matches = []
            if req.full_fetch:
                for m in all_matches:
                    full_text = "\n\n".join(c.get("text", "") for c in m.get("comments", []))
                    llm_matches.append({"request_id": m["request_id"], "text": full_text, "source": "comments"})
            else:
                llm_matches = all_matches
            summary_text = summarise_search_results(llm_matches, query, req.custom_prompt)
        except Exception as e:
            summary_text = f"LLM summary failed: {e}"

    return JSONResponse({
        "results": result_text,
        "summary": summary_text,
        "status": f"Found {len(all_matches)} results.",
    })
