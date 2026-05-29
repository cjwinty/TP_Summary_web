import json
import logging
import time
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from database import (
    list_chains, get_chain, save_chain, update_chain, delete_chain,
    list_runs, get_run, create_run as pg_create_run,
    update_run_step as pg_update_run_step, finish_run as pg_finish_run,
)
from shared.prompt_chain_executor import execute_chain

logger = logging.getLogger(__name__)
router = APIRouter()


class StepDef(BaseModel):
    step_order: int
    name: str
    prompt_template: str
    input_variable: str = ""
    output_variable: str = ""
    variables: dict = {}


class ChainSaveRequest(BaseModel):
    name: str
    description: str = ""
    steps: list[StepDef] = []


class ChainRunRequest(BaseModel):
    input: str = ""


@router.get("/chains", response_class=HTMLResponse)
async def chains_page(request: Request):
    from jinja_env import templates
    return templates.TemplateResponse("chain_builder.html", {"request": request})


@router.get("/chains.json", response_class=JSONResponse)
async def list_chains_api():
    chains = list_chains()
    return JSONResponse(chains)


@router.post("/chains")
async def create_chain(req: ChainSaveRequest):
    steps = [s.model_dump() for s in req.steps]
    chain_id = save_chain(req.name, req.description, steps)
    return JSONResponse({"id": chain_id, "message": "Chain created."})


@router.get("/chains/{chain_id}")
async def get_chain_api(chain_id: int):
    chain = get_chain(chain_id)
    if not chain:
        return JSONResponse({"error": "Not found."}, status_code=404)
    return JSONResponse(chain)


@router.put("/chains/{chain_id}")
async def update_chain_api(chain_id: int, req: ChainSaveRequest):
    steps = [s.model_dump() for s in req.steps]
    update_chain(chain_id, req.name, req.description, steps)
    return JSONResponse({"message": "Chain updated."})


@router.delete("/chains/{chain_id}")
async def delete_chain_api(chain_id: int):
    delete_chain(chain_id)
    return JSONResponse({"message": "Chain deleted."})


@router.post("/chains/{chain_id}/run")
async def run_chain(chain_id: int, req: ChainRunRequest):
    try:
        result = execute_chain(
            chain_id, req.input,
            db_get_chain=get_chain,
            db_create_run=pg_create_run,
            db_update_run_step=pg_update_run_step,
            db_finish_run=pg_finish_run,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/chains/{chain_id}/runs")
async def list_chain_runs(chain_id: int):
    runs = list_runs(chain_id)
    return JSONResponse(runs)


@router.get("/chains/{chain_id}/runs/{run_id}")
async def get_chain_run(chain_id: int, run_id: int):
    run = get_run(run_id)
    if not run:
        return JSONResponse({"error": "Not found."}, status_code=404)
    return JSONResponse(run)
