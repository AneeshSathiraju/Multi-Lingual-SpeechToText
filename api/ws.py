"""WebSocket endpoint for live transcription."""
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.concurrency import run_in_threadpool

from realtime.chunker import UtteranceChunker
from realtime.stream import LiveTranscriber
from recognition.language import UnsupportedLanguageError, normalize_language

router = APIRouter(prefix="/api/v1")


@router.websocket("/live")
async def live(ws: WebSocket, language: str | None = None):
    await ws.accept()
    try:
        code = normalize_language(language)
    except UnsupportedLanguageError as exc:
        await ws.close(code=1008, reason=str(exc))
        return

    session = LiveTranscriber(
        ws.app.state.transcriber,
        language=code,
        chunker=UtteranceChunker(threshold=ws.app.state.settings.live_threshold),
    )
    await ws.send_json({"type": "ready", "format": "pcm_s16le", "sample_rate": 16000, "channels": 1})

    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return

        if message.get("bytes") is not None:
            data = message["bytes"]
            data = data[: len(data) // 2 * 2]  # int16 = 2 bytes; drop a stray odd byte
            block = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            # Transcription blocks, so run it in a thread to keep the event loop responsive.
            events = await run_in_threadpool(session.feed, block)
        elif message.get("text") == "end":
            for event in await run_in_threadpool(session.flush):
                await ws.send_json(asdict(event))
            await ws.close()
            return
        else:
            continue

        for event in events:
            await ws.send_json(asdict(event))