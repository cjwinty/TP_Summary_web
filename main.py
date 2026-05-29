import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from shared.config import validate_env, logger, VERSION
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    env_valid, env_errors = validate_env()
    if not env_valid:
        logger.warning("Web app started with incomplete env config: %s", env_errors)
    logger.info("Web app starting up")
    init_db()
    yield
    logger.info("Web app shutting down")


app = FastAPI(title="TP Query Web", version=VERSION, lifespan=lifespan)


@app.exception_handler(404)
async def not_found(request, exc):
    from jinja_env import templates
    return templates.TemplateResponse("error.html", {"request": request, "code": 404, "message": "Page not found"}, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    from jinja_env import templates
    return templates.TemplateResponse("error.html", {"request": request, "code": 500, "message": "Internal server error"}, status_code=500)


static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

from routes.summarize import router as summarize_router
from routes.settings import router as settings_router
from routes.comments import router as comments_router
from routes.summaries import router as summaries_router
from routes.search import router as search_router
from routes.chains import router as chains_router
from routes.rag import router as rag_router
from routes.chat import router as chat_router
from routes.browse import router as browse_router

app.include_router(summarize_router)
app.include_router(settings_router)
app.include_router(comments_router)
app.include_router(summaries_router)
app.include_router(search_router)
app.include_router(chains_router)
app.include_router(rag_router)
app.include_router(chat_router)
app.include_router(browse_router)


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
