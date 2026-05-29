import json
import logging
from threading import Thread
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from shared import config as cfg
from database import (
    get_all_prompts, save_prompt, get_cache_counts, delete_all_summaries,
    get_max_min_request_id, check_database_health, optimize_database,
)

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
    html = f'<p class="status-text">Cached: {counts["comments"]} comments, {counts["summaries"]} summaries, {counts["custom_fields"]} custom fields, {counts["embeddings"]} embeddings</p>'
    if id_range["min"] is not None:
        html += f'<p class="status-text">ID Range: {id_range["min"]} - {id_range["max"]}</p>'
    return HTMLResponse(html)


@router.post("/settings/clear-summaries")
async def clear_summaries():
    delete_all_summaries()
    return JSONResponse({"message": "All summaries cleared."})


@router.get("/settings/health")
async def health_check():
    health = check_database_health()
    return JSONResponse(health)


@router.post("/settings/optimize")
async def optimize():
    result = optimize_database()
    return JSONResponse(result)
