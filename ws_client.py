"""Streams your mic to the server. Run: python ws_client.py "ws://127.0.0.1:8000/api/v1/live?language=hi" """
import asyncio
import json
import sys

import sounddevice as sd
import websockets


async def main(url: str):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_audio(indata, frames, time_info, status):
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    async with websockets.connect(url) as ws:
        async def send():
            with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1600, callback=on_audio):
                while True:
                    await ws.send(await queue.get())

        async def receive():
            async for message in ws:
                event = json.loads(message)
                if event["type"] != "ready":
                    print(f"{event['type']:>7} [{event['language']}] {event['text']}")

        await asyncio.gather(send(), receive())


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/api/v1/live"))
    except KeyboardInterrupt:
        pass