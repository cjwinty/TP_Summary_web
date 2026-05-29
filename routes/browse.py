import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from database import (
    get_cached_projects, get_entity_types_for_project,
    get_entities_by_project_and_type,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/browse", response_class=HTMLResponse)
async def browse_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("browse.html", {"request": request})


@router.get("/browse/projects")
async def list_projects():
    projects = get_cached_projects()
    return JSONResponse({"projects": projects})


@router.get("/browse/projects/{project_id}/types")
async def list_types(project_id: int):
    types = get_entity_types_for_project(project_id)
    return JSONResponse({"types": types})


@router.get("/browse/projects/{project_id}/{entity_type}")
async def list_entities(project_id: int, entity_type: str):
    entities = get_entities_by_project_and_type(project_id, entity_type)
    return JSONResponse({"entities": entities})
