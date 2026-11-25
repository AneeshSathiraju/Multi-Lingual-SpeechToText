import logging
from contextlib import asynccontextmanager
from api.ws import router as ws_router

from fastapi import FastAPI

from api.routes import router
from config import Settings
from recognition.transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: the model loads here, never per request.
    settings = Settings()
    app.state.settings = settings
    app.state.transcriber = Transcriber(settings.model_size, settings.device, settings.compute_type)
    yield


app = FastAPI(title="Multilingual Speech-to-Text API", version="0.1.0", lifespan=lifespan)
app.include_router(router)

app.include_router(router)
app.include_router(ws_router)