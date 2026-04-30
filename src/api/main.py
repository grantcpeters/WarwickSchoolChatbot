"""
FastAPI backend — Chat API for Ask Warwick
"""

import os
import logging
import traceback
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.chatbot.rag_pipeline import chat

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ask Warwick API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ASSETS_DIR = STATIC_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if STATIC_ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_ASSETS_DIR), name="static")


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> Response:
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return JSONResponse({"status": "ok", "service": "Ask Warwick API"})


@app.post("/chat")
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters.",
        )

    log.info(
        "PROMPT ip=%s history_turns=%d msg=%r",
        request.client.host,
        len(req.history),
        req.message,
    )

    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def stream_response() -> AsyncIterator[str]:
        try:
            async for token in chat(req.message, history):
                yield token
        except Exception:
            log.error("Chat error:\n%s", traceback.format_exc())
            yield "\n\n[Sorry, an error occurred. Please try again.]"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.post("/chat-debug")
async def chat_debug(req: ChatRequest) -> JSONResponse:
    """Non-streaming debug endpoint — returns full response as JSON."""
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        tokens = []
        async for token in chat(req.message, history):
            tokens.append(token)
        return JSONResponse({"response": "".join(tokens), "ok": True})
    except Exception:
        log.error("Chat debug error:\n%s", traceback.format_exc())
        return JSONResponse(
            {"error": "An internal error occurred.", "ok": False}, status_code=500
        )


@app.get("/{full_path:path}")
async def frontend_routes(full_path: str) -> Response:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Not found")

    requested = STATIC_DIR / full_path
    if full_path and requested.is_file():
        return FileResponse(requested)

    return FileResponse(INDEX_FILE)
