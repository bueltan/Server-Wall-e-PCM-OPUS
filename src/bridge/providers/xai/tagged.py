"""Response identities and confirmed cancellation/history truncation for xAI."""
import asyncio
import base64
import json
import time
from bridge.diagnostics.logging import event as log_event
from bridge.providers.xai.realtime import XaiRealtimeBridge
from bridge.providers.operation_errors import ProviderOperationError, already_absent

class TaggedXai(XaiRealtimeBridge):
    def __init__(self, config, queue, valid):
        super().__init__(config, queue)
        self.valid=valid
        self.ready_event=asyncio.Event()
        self.operation_lock=asyncio.Lock()
        self.generation_done=asyncio.Event();self.generation_done.set()
        self.pending=None
        self.responses={}
        self.items={}
        self.confirmations={}
        self.pending_operations={}
        self.operation_sequence=0
        self.recent_operations={}
        self.output_done=set()
        self.audio_totals={}
        self.response_requested_at=0.0
        self.cancel_identity=None

    async def connect(self):
        self._clear_operations();self.recent_operations.clear()
        self.ready_event.clear();self.pending=None;self.responses.clear()
        self.items.clear();self.output_done.clear();self.generation_done.set()
        await super().connect()

    async def commit_turn(self, token):
        async with self.operation_lock:
            if not self.ready_event.is_set() or not self.valid(token): return False
            self.pending=token;self.generation_done.clear()
            self.response_requested_at=time.monotonic()
            log_event("XAI_COMMIT_BEGIN",token)
            try:
                await self._safe_send({'type':'input_audio_buffer.commit'})
                if not self.valid(token):
                    self.pending=None;self.generation_done.set();return False
                await self._safe_send({'type':'response.create'})
                log_event('XAI_RESPONSE_REQUESTED',token)
                return True
            except BaseException:
                self.pending=None
                # Delivery can be ambiguous; reconnect rather than replay a commit.
                await self.close()
                raise

    def _clear_operations(self):
        for _,future in self.pending_operations.values():
            if not future.done(): future.cancel()
        self.pending_operations.clear();self.confirmations.clear()

    def _begin_operation(self, payload):
        self.operation_sequence+=1
        payload=dict(payload,event_id=f'piuu-op-{self.operation_sequence}')
        future=asyncio.get_running_loop().create_future()
        # Cancellation can be acknowledged after its history item arrives.
        # Retrieve exceptions even if the transaction itself is subsequently cancelled.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self.pending_operations[payload['event_id']]=(payload,future)
        self.recent_operations[payload['event_id']]=payload
        while len(self.recent_operations)>64:
            self.recent_operations.pop(next(iter(self.recent_operations)))
        return payload,future

    def _accept_operation_error(self, payload, error):
        return payload['type']=='conversation.item.delete' and already_absent(error)

    async def _confirmed(self, payload, event, item=''):
        payload,future=self._begin_operation(payload);key=(event,item)
        self.confirmations[key]=future
        started=time.monotonic()
        log_event("XAI_OPERATION",self.cancel_identity,operation=payload['type'],item=item,
                  audio_end_ms=payload.get('audio_end_ms'),event_id=payload['event_id'])
        try:
            await self._safe_send(payload)
            await asyncio.wait_for(future,self.config.history_confirmation_timeout_sec)
            log_event("XAI_CONFIRMED",self.cancel_identity,confirmation=event,item=item,
                      elapsed_ms=round((time.monotonic()-started)*1000))
        finally:
            self.confirmations.pop(key,None)
            self.pending_operations.pop(payload['event_id'],None)

    async def cancel_turn(self, token, played_ms):
        async with self.operation_lock:
            if not self.connected_event.is_set():
                log_event("XAI_CANCEL_UNAVAILABLE",token);return False
            self.cancel_identity=token
            started=time.monotonic()
            log_event("XAI_CANCEL_BEGIN",token,played_ms=played_ms)
            try:
                if not self.generation_done.is_set():
                    # No subsequent response.create can pass the operation lock.
                    log_event('XAI_RESPONSE_CANCEL',token)
                    await self._safe_send({'type':'response.cancel'})
                    await asyncio.wait_for(self.generation_done.wait(),1.0)
                remaining=max(0,played_ms)
                for item in self.items.get(token,[]):
                    item_id,index,duration=item
                    if remaining>=duration and duration>0:
                        remaining-=duration;continue
                    if remaining>0:
                        await self._confirmed({'type':'conversation.item.truncate','item_id':item_id,
                            'content_index':index,'audio_end_ms':min(remaining,duration)},
                            'conversation.item.truncated',item_id)
                        remaining=0
                    else:
                        await self._confirmed({'type':'conversation.item.delete','item_id':item_id},
                            'conversation.item.deleted',item_id)
                await self._confirmed({'type':'input_audio_buffer.clear'},'input_audio_buffer.cleared')
                self.pending=None;self.items.pop(token,None)
                log_event("XAI_CANCEL_COMPLETE",token,elapsed_ms=round((time.monotonic()-started)*1000))
                return True
            except Exception as exc:
                log_event("XAI_CANCEL_FAILED",token,error=type(exc).__name__,
                          elapsed_ms=round((time.monotonic()-started)*1000))
                # A new input must not inherit an uncertain remote buffer/history.
                await self.close();return False

    async def receive_events_forever(self):
        async for raw in self.ws:
            await self.handle_event(json.loads(raw))
        self.ready_event.clear()

    async def handle_event(self, event):
        kind=event.get('type')
        key=(kind,event.get('item_id',''))
        future=self.confirmations.get(key)
        if future is not None and not future.done(): future.set_result(event)
        if kind=='session.updated':
            self.ready_event.set()
            log_event('XAI_READY',rate=self.config.output_sample_rate)
        elif kind=='response.created':
            response_id=event.get('response',{}).get('id')
            if response_id and self.pending is not None:
                token=self.pending
                self.responses[response_id]=token;self.pending=None
                self.audio_totals={token:[0,0]}
                log_event("XAI_RESPONSE_CREATED",token,response_id=response_id)
        elif kind=='response.output_item.added':
            token=self.responses.get(event.get('response_id'))
            item=event.get('item',{})
            if token is not None and item.get('role')=='assistant' and item.get('id'):
                items=self.items.setdefault(token,[])
                if not any(i[0]==item['id'] for i in items):
                    items.append([item['id'],0,0])
        elif kind=='response.output_audio.delta':
            token=self.responses.get(event.get('response_id'))
            if token is None: return
            pcm=base64.b64decode(event['delta'],validate=True)
            totals=self.audio_totals.setdefault(token,[0,0])
            totals[0]+=1;totals[1]+=len(pcm)
            if totals[0]==1:
                log_event("XAI_FIRST_AUDIO",token,response_id=event.get('response_id'),
                          elapsed_ms=round((time.monotonic()-self.response_requested_at)*1000),
                          valid=self.valid(token))
            item_id=event.get('item_id');index=event.get('content_index',0)
            if item_id:
                items=self.items.setdefault(token,[])
                item=next((i for i in items if i[0]==item_id and i[1]==index),None)
                if item is None:
                    item=[item_id,index,0];items.append(item)
                item[2]+=len(pcm)//32
            if self.valid(token): self.output_pcm_queue.put_nowait((token,pcm))
        elif kind=='response.output_audio.done':
            token=self.responses.get(event.get('response_id'))
            if token is not None:
                totals=self.audio_totals.get(token,[0,0])
                log_event("XAI_AUDIO_DONE",token,response_id=event.get('response_id'),
                          chunks=totals[0],bytes=totals[1],audio_ms=totals[1]//32)
            if token is not None and self.valid(token) and token not in self.output_done:
                self.output_done.add(token);self.output_pcm_queue.put_nowait((token,None))
        elif kind=='response.done':
            response_id=event.get('response',{}).get('id',event.get('response_id'))
            token=self.responses.pop(response_id,None)
            if token is not None:
                log_event("XAI_RESPONSE_DONE",token,response_id=response_id,
                          status=event.get('response',{}).get('status'),valid=self.valid(token),
                          billable_audio_seconds=event.get('response',{}).get('usage',{}).get('billable_audio_seconds'))
                self.generation_done.set()
                if self.valid(token) and token not in self.output_done:
                    self.output_done.add(token);self.output_pcm_queue.put_nowait((token,None))
        elif kind=='error':
            error=event.get('error',{})
            request_id=error.get('event_id')
            operation=self.pending_operations.get(request_id)
            correlation='event_id'
            if operation is None and len(self.pending_operations)==1 and request_id not in self.recent_operations:
                # Some provider errors omit/replace the originating event ID.
                operation=next(iter(self.pending_operations.values()));correlation='single_pending'
            payload,future=operation if operation else (self.recent_operations.get(request_id),None)
            log_event('XAI_ERROR',code=error.get('code'),error_type=error.get('type'),
                      event_id=request_id,param=error.get('param'),message=str(error.get('message',''))[:512],
                      operation=payload.get('type') if payload else None,correlation=correlation)
            if payload and self._accept_operation_error(payload,error):
                log_event('XAI_OPERATION_RECOVERED',self.cancel_identity,operation=payload['type'],
                          event_id=payload['event_id'],code=error.get('code'))
                if future is not None and not future.done(): future.set_result(event)
                return
            if future is not None:
                if not future.done(): future.set_exception(ProviderOperationError(payload,error))
                return  # transaction decides recovery; keep receiving confirmations/events
            raise RuntimeError(f"xAI rejected operation code={error.get('code')}: {error.get('message','')}")

    async def close(self):
        self._clear_operations()
        self.ready_event.clear()
        await super().close()
