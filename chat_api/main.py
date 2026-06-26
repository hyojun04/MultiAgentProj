import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chat_api.config import get_settings
from chat_api.dependencies import agent_service
from chat_api.routers.chat import router as chat_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await agent_service.initialize()
        logger.info("[CHAT API] Agent service initialized")
    except Exception as exc:
        logger.warning("[CHAT API] Agent service initialization skipped: %s", exc)

    yield

    await agent_service.close()


app = FastAPI(title="Multi Agent Chat API", version="0.1.0", lifespan=lifespan)

# CORS 설정 - 프론트엔드에서 API 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Internal server error"}},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    settings = get_settings()
    uvicorn.run(
        "chat_api.main:app",
        host=settings.chat_api_host,
        port=settings.chat_api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
