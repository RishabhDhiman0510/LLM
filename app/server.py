from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .engine import ChatEngine
from .settings import Settings
from .store import ChatStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = ChatStore(settings.db_path)
        app.state.engine = ChatEngine(settings)
        print(f"Model loaded on {app.state.engine.device}")
        yield

    app = FastAPI(title="WikiGPT", lifespan=lifespan)
    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
