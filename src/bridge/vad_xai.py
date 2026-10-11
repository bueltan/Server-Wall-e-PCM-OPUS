"""xAI VAD adapter: provider events, response decoding and history repair."""
import asyncio
import time
from .tagged_xai import TaggedXai
from .diagnostics import event

class VadXai(TaggedXai):
    def __init__(self, config, queue, valid, events):
        super().__init__(config, queue, valid)
        self.events = events
        self.history_ready = asyncio.Event(); self.history_ready.set()
        self.done = {}
        self.provider_done = {}
        self.latest_response_id = None
        self.pause_count = 0
        self.response_ids = {}

    def turn_detection(self):
        return {'type': 'server_vad'}

    async def connect(self):
        self.done.clear(); self.response_ids.clear(); self.provider_done.clear()
        self.latest_response_id=None; self.pause_count=0; self.history_ready.set()
        await super().connect()

    def pause_input(self):
        self.pause_count += 1; self.history_ready.clear()

    def resume_input(self):
        self.pause_count = max(0, self.pause_count-1)
        if not self.pause_count: self.history_ready.set()

    async def send_audio_append(self, pcm):
        await self.history_ready.wait()
        return await super().send_audio_append(pcm)

    async def handle_event(self, message):
        kind = message.get('type')
        if kind in ('input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped'):
            self.events(kind, message)
        elif kind == 'response.created':
            response_id=message.get('response',{}).get('id')
            self.latest_response_id=response_id
            self.provider_done[response_id]=asyncio.Event()
            token = self.events(kind, message)
            self.pending = token
            self.response_requested_at = time.monotonic()
            if token:
                self.generation_done.clear()
                self.done[token] = asyncio.Event()
                self.response_ids[token] = message.get('response', {}).get('id')
        token = self.responses.get(message.get('response', {}).get('id', message.get('response_id')))
        await super().handle_event(message)
        if kind == 'response.done':
            event('XAI_USAGE',token,billable_audio_seconds=message.get('response',{}).get('usage',{}).get('billable_audio_seconds'))
            response_id=message.get('response',{}).get('id',message.get('response_id'))
            if response_id in self.provider_done: self.provider_done[response_id].set()
            if token in self.done: self.done[token].set()
            # Old completed IDs have no further transport role.
            for old in list(self.provider_done):
                if old!=self.latest_response_id and self.provider_done[old].is_set():
                    self.provider_done.pop(old,None)

    async def repair_history(self, token, played_ms, manual=False):
        """Preserve the new user audio. Never clear it on a VAD interruption."""
        if token is None: return
        self.cancel_identity = token
        done = self.done.get(token)
        if done and not done.is_set():
            if manual:
                await self._safe_send({'type': 'response.cancel', 'response_id': self.response_ids[token]})
            await asyncio.wait_for(done.wait(), 1.0)
        remaining = max(0, played_ms)
        async with self.operation_lock:
            for item_id, index, duration in self.items.pop(token, []):
                if duration > 0 and remaining >= duration:
                    remaining -= duration
                elif remaining > 0:
                    await self._confirmed({'type':'conversation.item.truncate','item_id':item_id,
                        'content_index':index,'audio_end_ms':min(remaining,duration)},
                        'conversation.item.truncated',item_id)
                    remaining = 0
                else:
                    await self._confirmed({'type':'conversation.item.delete','item_id':item_id},
                        'conversation.item.deleted',item_id)
        self.done.pop(token, None); self.response_ids.pop(token, None)
        self.output_done.discard(token); self.audio_totals.pop(token, None)
        event('HISTORY_REPAIRED', token, played_ms=played_ms, manual=manual)

    async def close(self):
        # Bound shutdown even when the peer stops answering the close handshake.
        self.ready_event.clear();self.connected_event.clear()
        ws=self.ws;self.ws=None
        if ws is None: return
        try:
            await asyncio.wait_for(ws.close(),2.0)
        except TimeoutError:
            transport=getattr(ws,'transport',None)
            if transport is not None: transport.abort()
            event('XAI_CLOSE_FORCED')
        except Exception as exc:
            event('XAI_CLOSE_ERROR',error=type(exc).__name__)
