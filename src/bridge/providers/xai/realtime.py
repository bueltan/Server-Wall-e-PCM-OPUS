import asyncio
import base64
import json
import os
import websockets
from bridge.config import BridgeConfig


class XaiRealtimeBridge:
    def __init__(self, config: BridgeConfig, output_pcm_queue: asyncio.Queue[bytes | None]):
        self.config = config
        self.output_pcm_queue = output_pcm_queue

        self.ws = None
        self.connected_event = asyncio.Event()
        self.send_lock = asyncio.Lock()

        self.output_audio_bytes = 0
        self.output_audio_chunks = 0

    async def connect(self) -> None:
        api_key = self.config.xai_api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set")

        print("[XAI] connecting...")
        self.ws = await websockets.connect(
            self.config.xai_realtime_url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=20,
            close_timeout=10,
            ping_interval=15,
            ping_timeout=30,
            max_queue=64,
        )
        self.connected_event.set()
        await self._send_session_update()
        print("[XAI] connected")

    async def _safe_send(self, payload: dict) -> None:
        if self.ws is None or not self.connected_event.is_set():
            raise RuntimeError("xAI websocket is not connected")
        async with self.send_lock:
            await self.ws.send(json.dumps(payload))

    def turn_detection(self):
        return None

    def expression_tools(self):
        return []  # Legacy providers retain their original lifecycle.

    async def _send_session_update(self) -> None:
        from bridge.expressions.plan import EXPRESSION_INSTRUCTIONS
        tools = self.expression_tools()
        instructions = self.config.xai_prompt
        if tools: instructions += "\n\n" + EXPRESSION_INSTRUCTIONS
        await self._safe_send(
            {
                "type": "session.update",
                "session": {
                    "voice": self.config.xai_voice,
                    "instructions": instructions,
                    "turn_detection": self.turn_detection(),
                    "tools": [{"type": "web_search"}] + tools,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.config.input_sample_rate,
                            },
                            "transcription": {
                                "model": "grok-transcribe"
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.config.output_sample_rate,
                            }
                        },
                    },
                },
            }
        )

    async def send_audio_append(self, pcm_data: bytes) -> bool:
        if not self.connected_event.is_set():
            print("[XAI] append skipped: not connected")
            return False

        try:
            await self._safe_send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm_data).decode("utf-8"),
                }
            )
            return True
        except Exception as exc:
            print(f"[XAI] append failed: {exc}")
            return False

    async def send_commit(self) -> bool:
        if not self.connected_event.is_set():
            print("[XAI] commit skipped: not connected")
            return False

        self.output_audio_bytes = 0
        self.output_audio_chunks = 0

        try:
            await self._safe_send({"type": "input_audio_buffer.commit"})
            await self._safe_send({"type": "response.create"})
            print("[XAI] commit + response.create sent")
            return True
        except Exception as exc:
            print(f"[XAI] commit failed: {exc}")
            return False

    async def receive_events_forever(self) -> None:
        if self.ws is None:
            raise RuntimeError("receive_events_forever called before connect")

        async for raw_event in self.ws:
            event = json.loads(raw_event)
            event_type = event.get("type")

            if event_type == "session.created":
                session_id = event.get("session", {}).get("id")
                print(f"[XAI] session.created id={session_id}")

            elif event_type == "session.updated":
                print("[XAI] session.updated")

            elif event_type == "input_audio_buffer.committed":
                print("[XAI] input_audio_buffer.committed")

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"[USER] {transcript}")

            elif event_type == "response.output_audio.delta":
                pcm_data = base64.b64decode(event["delta"])
                self.output_audio_chunks += 1
                self.output_audio_bytes += len(pcm_data)

                if self.output_audio_chunks <= 5 or self.output_audio_chunks % 25 == 0:
                    samples = len(pcm_data) // 2
                    approx_ms = (samples / self.config.output_sample_rate) * 1000.0
                    print(
                        "[XAI] output_audio.delta "
                        f"chunk={self.output_audio_chunks} bytes={len(pcm_data)} ~{approx_ms:.1f}ms"
                    )

                await self.output_pcm_queue.put(pcm_data)

            elif event_type == "response.output_audio_transcript.delta":
                delta_text = event.get("delta", "")
                if delta_text:
                    print(delta_text, end="", flush=True)

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"\n[ASSISTANT] {transcript}")

            elif event_type == "response.output_audio.done":
                total_samples = self.output_audio_bytes // 2
                total_ms = (total_samples / self.config.output_sample_rate) * 1000.0
                print(
                    "[XAI] response.output_audio.done "
                    f"chunks={self.output_audio_chunks} bytes={self.output_audio_bytes} ~{total_ms:.1f}ms"
                )
                await self.output_pcm_queue.put(None)

            elif event_type == "response.done":
                usage = event.get("usage", {})
                print(f"[XAI] response.done tokens={usage.get('total_tokens')}")

            elif event_type == "ping":
                pass

            elif event_type == "error":
                print(f"[XAI] error: {json.dumps(event, ensure_ascii=False)}")

            else:
                print(f"[XAI] unhandled event: {event_type}")

    async def close(self) -> None:
        self.connected_event.clear()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
