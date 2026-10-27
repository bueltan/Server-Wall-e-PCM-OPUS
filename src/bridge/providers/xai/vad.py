"""xAI VAD adapter: provider events, response decoding and history repair."""
import asyncio
import time
from bridge.providers.xai.tagged import TaggedXai
from bridge.providers.xai.expression_tools import ExpressionTools
from bridge.expressions.plan import tool_schema
from bridge.providers.operation_errors import no_active_response
from bridge.diagnostics.logging import event

class VadXai(TaggedXai):
    def __init__(self, config, queue, valid, events):
        super().__init__(config, queue, valid)
        self.events = events
        self.expression_control = ExpressionTools(self)
        self._suppress_tool_end = False
        self.history_ready = asyncio.Event(); self.history_ready.set()
        self.done = {}
        self.provider_done = {}
        self.latest_response_id = None
        self.pause_count = 0
        self.response_ids = {}
        self.history_changed = asyncio.Event()
        self.repaired_responses = {}
        self.cancel_operations = {}

    def release_response(self, token):
        self.items.pop(token, None); self.done.pop(token, None)
        self.response_ids.pop(token, None); self.output_done.discard(token)
        self.audio_totals.pop(token, None)

    def expression_tools(self):
        return [tool_schema()] if self.config.expression_tools_enabled else []

    def turn_detection(self):
        return {'type': 'server_vad'}

    async def connect(self):
        self.expression_control.reset()
        self.done.clear(); self.response_ids.clear(); self.provider_done.clear()
        self.cancel_operations.clear()
        self.repaired_responses.clear(); self.history_changed.clear()
        self.latest_response_id=None; self.pause_count=0; self.history_ready.set()
        await super().connect()

    def pause_input(self):
        self.pause_count += 1; self.history_ready.clear()

    def resume_input(self):
        self.pause_count = max(0, self.pause_count-1)
        if not self.pause_count: self.history_ready.set()

    async def wait_input_ready(self):
        if self.history_ready.is_set(): return
        started=time.monotonic()
        event('INPUT_PAUSED',reason='history_repair')
        await self.history_ready.wait()
        event('INPUT_RESUMED',paused_ms=round((time.monotonic()-started)*1000))

    async def cancel_response(self, token):
        done=self.done.get(token)
        if done and not done.is_set():
            response_id=self.response_ids[token]
            event('XAI_RESPONSE_CANCEL',token,response_id=response_id)
            if token in self.cancel_operations: return
            payload,future=self._begin_operation({'type':'response.cancel','response_id':response_id})
            self.cancel_operations[token]=(payload['event_id'],future)
            await self._safe_send(payload)

    def _accept_operation_error(self, payload, error):
        if payload['type']=='response.cancel' and no_active_response(error):
            response_id=payload.get('response_id')
            token=self.responses.get(response_id)
            if token in self.done: self.done[token].set()
            if response_id==self.latest_response_id: self.generation_done.set()
            if response_id in self.provider_done: self.provider_done[response_id].set()
            self.history_changed.set()
            return True
        return super()._accept_operation_error(payload,error)

    async def repair_history(self, token, played_ms, manual=False, phase=None, cancel_if_missing=True):
        try:
            return await self._repair_history(token,played_ms,manual,phase,cancel_if_missing)
        finally:
            operation=self.cancel_operations.pop(token,None)
            if operation: self.pending_operations.pop(operation[0],None)

    async def handle_event(self, message):
        kind = message.get('type')
        if kind == 'response.function_call_arguments.done':
            if self.config.expression_tools_enabled: self.expression_control.collect(message)
            return
        if kind == 'input_audio_buffer.speech_started':
            self.expression_control.plans.clear()
            self.expression_control.calls.clear()
        if kind == 'response.output_item.added':
            self.expression_control.observe_item(message)

        response_id=message.get('response_id')
        retired=self.repaired_responses.get(response_id)
        if retired is not None and kind in ('response.output_item.added','response.output_audio.delta'):
            item_id=message.get('item',{}).get('id') if kind=='response.output_item.added' else message.get('item_id')
            if item_id and item_id not in retired:
                self.events('history.late_item',message)
            return  # old audio must not recreate local bookkeeping or playback
        if kind in ('input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped'):
            self.events(kind, message)
        elif kind == 'response.created':
            response_id=message.get('response',{}).get('id')
            self.latest_response_id=response_id
            self.provider_done[response_id]=asyncio.Event()
            continuation = self.expression_control.continuation
            self.expression_control.continuation = None
            if continuation is not None:
                if not self.valid(continuation) or not self.history_ready.is_set():
                    # Ambiguous response after interruption: never attach stale plans to a new turn.
                    raise RuntimeError('stale_tool_continuation')
                token = continuation
            else:
                token = self.events(kind, message)
            self.pending = token
            self.response_requested_at = time.monotonic()
            if token:
                self.generation_done.clear()
                self.done[token] = asyncio.Event()
                self.response_ids[token] = message.get('response', {}).get('id')
        token = self.responses.get(message.get('response', {}).get('id', message.get('response_id')))
        if kind == 'response.done' and token is not None:
            # The terminal snapshot may identify output whose incremental event
            # never arrived. Include it before declaring an early response empty.
            for item in message.get('response', {}).get('output', []):
                await super().handle_event({'type': 'response.output_item.added',
                    'response_id': self.response_ids[token], 'item': item})
        if (self.config.expression_tools_enabled and kind == 'response.output_audio.done'
                and not self.audio_totals.get(token, [0, 0])[0]):
            return  # Tool-only turns must not produce an empty audio END before continuation.
        tool_rid, calls = (None, {})
        if kind == 'response.done' and self.config.expression_tools_enabled:
            tool_rid, calls = self.expression_control.terminal_calls(message)
        had_audio = bool(token and self.audio_totals.get(token, [0, 0])[0])
        if kind == 'response.output_audio.delta' and token is not None and self.valid(token):
            plan = self.expression_control.take_plan(token)
            if plan is not None:
                self.events('expression.plan', {'identity': token, 'plan': plan})
        self._suppress_tool_end = bool(calls and not had_audio)
        try:
            await super().handle_event(message)
        finally:
            self._suppress_tool_end = False
        if kind in ("response.output_item.added", "response.output_audio.delta", "response.done", "error"):
            self.history_changed.set()
        if kind == 'response.done':
            event('XAI_USAGE',token,billable_audio_seconds=message.get('response',{}).get('usage',{}).get('billable_audio_seconds'))
            response_id=message.get('response',{}).get('id',message.get('response_id'))
            if response_id in self.provider_done: self.provider_done[response_id].set()
            if token in self.done: self.done[token].set()
            # Old completed IDs have no further transport role.
            for old in list(self.provider_done):
                if old!=self.latest_response_id and self.provider_done[old].is_set():
                    self.provider_done.pop(old,None)

        if kind == 'response.done' and calls:
            await self.expression_control.finish(tool_rid, token,
                message.get('response', {}).get('status'), had_audio)

    async def _repair_history(self, token, played_ms, manual=False, phase=None,
                              cancel_if_missing=True):
        """Preserve the new user audio. Never clear it on a VAD interruption."""
        if token is None: return
        self.cancel_identity = token
        self.expression_control.discard(token)
        if manual: await self.cancel_response(token)
        # A concrete item can be repaired without depending on response.done.
        # If interrupted immediately after response.created, wait for either an
        # item to identify or a terminal response proving no output remains.
        def check_cancel_error():
            operation=self.cancel_operations.get(token)
            if operation and operation[1].done(): operation[1].result()
        async def wait_item():
            check_cancel_error()
            while not self.items.get(token):
                check_cancel_error()
                done=self.done.get(token)
                if done is None or done.is_set(): return
                self.history_changed.clear()
                await self.history_changed.wait()
        if phase: phase("item_wait")
        try:
            await asyncio.wait_for(wait_item(),self.config.history_item_timeout_sec)
        except TimeoutError:
            if not cancel_if_missing or manual:
                raise
            # VAD normally cancels generation itself. An early response can stall
            # before producing an item; cancel that exact response as a fallback.
            if phase: phase("cancel_missing_output")
            event('XAI_CANCEL_FALLBACK', token,
                  response_id=self.response_ids.get(token), reason='missing_output')
            await self.cancel_response(token)
            if phase: phase("cancel_confirmation")
            await asyncio.wait_for(wait_item(),self.config.history_item_timeout_sec)
        if not self.items.get(token):
            event('HISTORY_EMPTY_CONFIRMED', token,
                  response_id=self.response_ids.get(token))
        if phase: phase("history_confirmation")
        remaining = max(0, played_ms)
        processed=set()
        async with self.operation_lock:
            while True:
                pending=[item for item in self.items.get(token,[]) if (item[0],item[1]) not in processed]
                if not pending: break
                for item_id, index, duration in pending:
                    done=self.done.get(token)
                    terminal=done is not None and done.is_set()
                    last=self.items.get(token,[])[-1]
                    # While generation is not confirmed terminal, even the last
                    # fully heard item needs a boundary: it could still grow.
                    if duration>0 and remaining>=duration and (terminal or last[:2]!=[item_id,index]):
                        remaining-=duration
                    elif remaining>0:
                        await self._confirmed({'type':'conversation.item.truncate','item_id':item_id,
                            'content_index':index,'audio_end_ms':min(remaining,duration)},
                            'conversation.item.truncated',item_id)
                        remaining=0
                    else:
                        await self._confirmed({'type':'conversation.item.delete','item_id':item_id},
                            'conversation.item.deleted',item_id)
                    processed.add((item_id,index))
                # Account for items delivered while confirmation was in flight.
        response_id=self.response_ids.get(token)
        if response_id:
            self.repaired_responses[response_id]={item[0] for item in self.items.get(token,[])}
            while len(self.repaired_responses)>64:
                self.repaired_responses.pop(next(iter(self.repaired_responses)))
            self.responses.pop(response_id,None)
        self.items.pop(token, None)
        self.done.pop(token, None); self.response_ids.pop(token, None)
        self.output_done.discard(token); self.audio_totals.pop(token, None)
        event('HISTORY_REPAIRED', token, played_ms=played_ms, manual=manual)

    async def close(self):
        self.expression_control.reset()
        # Bound shutdown even when the peer stops answering the close handshake.
        self._clear_operations();self.cancel_operations.clear()
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
