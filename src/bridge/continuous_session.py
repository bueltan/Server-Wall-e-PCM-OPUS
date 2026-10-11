"""UDP v3 conversation coordinator; one instance per admitted client."""
import asyncio
import secrets
import math
import struct
import time
from .audio_pipeline import AudioPipeline
from .conversation import Conversation
from .conversation_protocol import Identity, Packet, Kind, Event, encode, decode
from .reliable_events import ReliableEvents
from .vad_xai import VadXai
from .ingress import InboundSession
from .diagnostics import event
from .startup_audio import StartupAudio
from .interruption import InterruptionCoordinator

class ContinuousSession(AudioPipeline):
    def __init__(self, config):
        for name in ('core_disconnect_timeout_sec','history_repair_timeout_sec',
                     'history_item_timeout_sec','history_confirmation_timeout_sec'):
            if not math.isfinite(getattr(config,name)) or getattr(config,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(config)
        self.transport = self.remote_addr = self.client_nonce = None
        self.server_nonce = secrets.randbits(64) or 1
        self.retired = []
        self.last_conversation = 0
        self.conversation = Conversation()
        self.controls = asyncio.Queue(maxsize=8)
        self.acks = {}; self.pending = set()
        self.tasks = []
        self.closing = False
        self.peer_resetting = False
        self.last_peer_seen = time.monotonic()
        self.repairs = set()
        self.interruptions = InterruptionCoordinator(self)
        self.provider_task = None
        self.input_live = False
        self.startup = StartupAudio(config.startup_audio_frames)
        self.last_activity = time.monotonic()
        self.counters = dict(invalid=0, foreign=0, stale=0, stale_output=0, opus_frames=0)
        self.events = ReliableEvents(self._wire, self.fail)
        self.xai = VadXai(config, self.output_pcm_queue, self.output_valid, self.provider_event)
        self.input = self.new_input()

    def new_input(self):
        worker = InboundSession(self.pcm_packet_format, self.config, self._accept_audio,
            self.ignore_end, self.no_commit, nack_sink=self.nack, silence_sink=self._accept_audio, before_audio=self.xai.wait_input_ready)
        worker.log_identity = self.conversation.identity
        return worker

    async def ignore_end(self, packet): pass
    async def no_commit(self): return False
    def ready(self): return self.transport is not None and not self.closing  # admission readiness, not an idle xAI connection
    def output_valid(self, token): return token is not None and token == self.conversation.response
    def _wire(self, p):
        if self.transport and self.remote_addr: self.transport.sendto(encode(p), self.remote_addr)
    def _send(self, kind, identity, sequence=0, body=b'', flags=0):
        self._wire(Packet(kind, flags, identity, sequence, body))
    def ack(self, p, ok=True):
        kind = {Kind.START:Kind.START_ACK, Kind.CANCEL:Kind.CANCEL_ACK,
                Kind.INTERRUPT:Kind.INTERRUPT_ACK, Kind.PLAYED:Kind.PLAYED_ACK}[p.kind]
        reply = Packet(kind, int(ok), p.identity, p.sequence)
        key = (p.kind, p.identity.turn, p.sequence)
        self.pending.discard(key); self.acks[key] = reply
        while len(self.acks) > 64: self.acks.pop(next(iter(self.acks)))
        self._wire(reply)
        event('CONTROL_ACK', p.identity, kind=kind.name, request=p.sequence, ok=ok)

    def connection_made(self, transport):
        self.transport = transport
        self.events.start()
        for coro, name in [(self.run_controls(), 'conversation-controls'),
                           (self._udp_output_sender(), 'opus-output'),
                           (self.watchdog(), 'conversation-watchdog')]:
            self.tasks.append(asyncio.create_task(coro, name=name))

    def datagram_received(self, data, addr):
        try: p = decode(data)
        except (ValueError, struct.error): self.counters['invalid'] += 1; return
        if p.flags or not p.identity.client: self.counters['invalid'] += 1; return
        if p.kind == Kind.HELLO:
            if p.body or p.identity.server or p.identity.turn or p.identity.response: return
            if self.client_nonce is None:
                self.client_nonce = p.identity.client; self.remote_addr = addr
            if addr != self.remote_addr: self.counters['foreign'] += 1; return
            if p.identity.client != self.client_nonce:
                if p.identity.client in self.retired: return
                if self.conversation.identity or self.pending or not self.ready():
                    if self.conversation.identity and not self.peer_resetting:
                        self.peer_resetting=True; self.schedule(self.fail('peer_restarted'))
                    return
                self.retired = (self.retired + [self.client_nonce])[-32:]
                self.client_nonce = p.identity.client
                self.server_nonce = secrets.randbits(64) or 1
                self.last_conversation = 0; self.acks.clear()
            self.last_peer_seen=time.monotonic()
            self._send(Kind.HELLO_ACK, Identity(self.client_nonce,self.server_nonce,0,0),p.sequence,flags=int(self.ready()))
            return
        if addr != self.remote_addr or p.identity.client != self.client_nonce:
            self.counters['foreign'] += 1; return
        if p.identity.server != self.server_nonce:
            self._send(Kind.ERROR,p.identity,p.sequence); return
        if p.kind == Kind.PING:
            if p.body or p.identity.turn or p.identity.response:
                self.counters["invalid"] += 1; return
            self.last_peer_seen=time.monotonic()
            self._send(Kind.PONG,p.identity,p.sequence,flags=int(self.ready())); return
        current = self.conversation.identity
        if p.kind == Kind.EVENT_ACK:
            if self.events.ack(p): self.last_peer_seen=time.monotonic()
            return
        if p.kind in (Kind.START,Kind.CANCEL,Kind.INTERRUPT,Kind.PLAYED):
            if not p.identity.turn: return
            if (p.kind == Kind.START and (p.body or p.identity.response)) or (p.kind != Kind.START and len(p.body)!=4): return
            if p.kind == Kind.START or (current and p.identity.turn == current.turn):
                self.last_peer_seen=time.monotonic()
            key = (p.kind,p.identity.turn,p.sequence)
            if key in self.acks: self._wire(self.acks[key]); return
            if key in self.pending: return
            if self.controls.full(): self.schedule(self.fail('control_overflow')); return
            self.pending.add(key)
            # Stop old PCM/output at admission, not after waiting for xAI operations.
            if p.kind == Kind.CANCEL and current and p.identity.turn == current.turn:
                self.conversation.close(); self.input.close(); self.reset_output()
            self.controls.put_nowait(p); return
        if current is None or p.identity.client != current.client or p.identity.turn != current.turn:
            self.counters['stale'] += 1; return
        if p.identity.response: return  # uplink is independent of response identity
        if p.kind == Kind.MISSING and not p.body:
            self.last_peer_seen=time.monotonic()
            self.input.stats.cache_misses += 1
        elif p.kind == Kind.PCM:
            try: inner = self.input.validate(p.body)
            except ValueError: self.counters['invalid'] += 1; return
            if inner['seq'] != p.sequence or self.pcm_packet_format.is_end_packet(inner):
                self.counters['invalid'] += 1; return
            self.last_peer_seen=time.monotonic()
            if self.input_live: self.input.admit(p.body)
            else: self.startup.admit(p.body)

    def schedule(self, coro):
        task = asyncio.create_task(coro)
        self.repairs.add(task); task.add_done_callback(self.repairs.discard)
        return task

    def notify(self, code, token=None):
        token = token or self.conversation.identity
        return self.events.emit(token, code, self.conversation.utterance)

    def provider_event(self, kind, message):
        if self.conversation.identity is None: return None
        self.last_activity = time.monotonic()
        if kind == 'input_audio_buffer.speech_started':
            previous = self.conversation.speech_started()
            self.reset_output()
            future = self.notify(Event.SPEECH_STARTED, previous)
            if previous:
                self.interruptions.begin(previous, future)
            event('SPEECH_STARTED',self.conversation.identity,utterance=self.conversation.utterance,
                  interrupted_response=previous.response if previous else 0)
        elif kind == 'input_audio_buffer.speech_stopped':
            self.conversation.speech_stopped(); self.notify(Event.SPEECH_STOPPED)
            event('SPEECH_STOPPED',self.conversation.identity,utterance=self.conversation.utterance)
        elif kind == 'history.late_item':
            event('HISTORY_LATE_ITEM',self.conversation.identity,response_id=message.get('response_id'))
            self.schedule(self.fail('late_item_after_history_repair'))
        elif kind == 'response.created':
            if self.interruptions.pending:
                event('RESPONSE_DURING_HISTORY_REPAIR', self.conversation.identity,
                      response_id=message.get('response',{}).get('id'))
                self.schedule(self.fail('response_before_history_ready'))
                return None
            token = self.conversation.response_started()
            self.reset_output(); self.output_gate.clear()
            future = self.notify(Event.RESPONSE_STARTED, token)
            self.schedule(self.allow_output(token, future))
            event('RESPONSE_STARTED',token,utterance=self.conversation.utterance)
            return token

    async def allow_output(self, token, future):
        try:
            await future
            if self.output_valid(token): self.output_gate.set()
        except asyncio.CancelledError: pass

    async def run_controls(self):
        while True:
            p = await self.controls.get()
            try:
                if p.kind == Kind.START:
                    if self.conversation.identity or p.identity.turn <= self.last_conversation or not self.ready():
                        self.ack(p,False); continue
                    self.conversation.open(p.identity); self.last_conversation=p.identity.turn
                    self.last_activity=time.monotonic()
                    self.events.reset(); self.startup.clear(); self.input_live=False
                    self.input=self.new_input(); self.reset_output()
                    self.provider_task=asyncio.create_task(self.run_provider(p.identity),name='active-xai')
                    self.ack(p); event('CONVERSATION_OPEN',p.identity)
                elif p.kind == Kind.CANCEL:
                    if p.identity.turn<self.last_conversation:
                        self.ack(p); continue
                    if self.conversation.identity and self.conversation.identity.turn != p.identity.turn:
                        self.ack(p,p.identity.turn<=self.last_conversation); continue
                    await self.close_remote(p.identity,int.from_bytes(p.body,'big'))
                    self.ack(p)
                elif p.kind == Kind.PLAYED:
                    token=self.conversation.response
                    if token and p.identity==token:
                        self.conversation.played(token); self.last_activity=time.monotonic()
                        self.xai.items.pop(token,None); self.xai.done.pop(token,None)
                        self.xai.response_ids.pop(token,None); self.xai.output_done.discard(token)
                        self.xai.audio_totals.pop(token,None)
                        event('PLAYBACK_FINISHED',token,played_ms=int.from_bytes(p.body,'big'))
                    self.ack(p)
                else:
                    if not self.conversation.identity or p.identity.turn != self.conversation.identity.turn:
                        self.ack(p,False); continue
                    old=self.conversation.response
                    self.conversation.speech_started(); self.reset_output()
                    future=self.notify(Event.SPEECH_STARTED,old)
                    if old is None and self.interruptions.active:
                        old=next(iter(self.interruptions.active))
                    played=int.from_bytes(p.body,'big') if old and p.identity==old else 0
                    self.interruptions.begin(old, future, manual=True, played_ms=played)
                    self.ack(p)
            except asyncio.CancelledError: raise
            except Exception as exc:
                event('CONTROL_FAILED',p.identity,error=type(exc).__name__)
                self.ack(p,False); await self.fail('control_failed')

    async def stop_provider(self):
        task=self.provider_task
        self.provider_task=None
        if task and task is not asyncio.current_task():
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
        await self.xai.close()

    async def close_remote(self, token, played):
        self.closing=True
        self.conversation.close(); self.input.close(); self.input_live=False
        self.startup.clear(); self.events.reset()
        await self.interruptions.close()
        for task in list(self.repairs):
            if task is not asyncio.current_task(): task.cancel()
        if self.input.task:
            try: await self.input.task
            except asyncio.CancelledError: pass
        self.input_recorder.close(); self.inbound_started=False; self.reset_output()
        # Closing the WebSocket releases this conversation and its history.
        await self.stop_provider()
        self.closing=False
        event('CONVERSATION_CLOSED',token,played_ms=played,provider_closed=True)

    async def fail(self, cause):
        if self.closing: return
        token=self.conversation.identity
        if token: self._send(Kind.ERROR,token)
        await self.close_remote(token,0)
        self.peer_resetting=False
        event('CONVERSATION_FAILED',token,cause=cause)

    def nack(self, base, mask):
        if self.conversation.identity:
            self._send(Kind.NACK,self.conversation.identity,base,struct.pack('!IH',base,mask))

    async def run_provider(self, token):
        receiver=None
        try:
            async def handshake():
                nonlocal receiver
                await self.xai.connect()
                receiver=asyncio.create_task(self.xai.receive_events_forever(),name='xai-events')
                ready=asyncio.create_task(self.xai.ready_event.wait())
                try:
                    done,_=await asyncio.wait([ready,receiver],return_when=asyncio.FIRST_COMPLETED)
                    if receiver in done:
                        await receiver
                        raise RuntimeError('Provider closed during handshake')
                finally:
                    ready.cancel()
            await asyncio.wait_for(handshake(),self.config.provider_open_timeout_sec)
            if self.conversation.identity!=token: return
            event('PROVIDER_READY',token,startup_frames=len(self.startup.pending),startup_dropped=self.startup.dropped)
            await self.notify(Event.READY)
            self.input.start()
            # Drain without flooding the normal 16-frame ingress queue. New PCM
            # remains in startup until the transfer completes, preserving order.
            while self.startup.pending and self.conversation.identity==token:
                if self.input.audio_queued>=self.config.input_queue_frames:
                    await asyncio.sleep(.001); continue
                self.input.admit(self.startup.pending.popleft())
            self.input_live=True
            await receiver
        except asyncio.CancelledError: raise
        except Exception as exc:
            event('XAI_DISCONNECTED',token,error=type(exc).__name__,reason=str(exc)[:160])
        finally:
            if receiver:
                receiver.cancel()
                try: await receiver
                except asyncio.CancelledError: pass
                except Exception: pass
            await self.xai.close()
            if self.conversation.identity==token:
                await self.fail('provider_closed')

    async def watchdog(self):
        next_stats=time.monotonic()+5
        while True:
            await asyncio.sleep(min(1.0,self.config.core_disconnect_timeout_sec))
            now=time.monotonic()
            await self.check_timeouts(now)
            if now<next_stats: continue
            next_stats=now+5
            event('CONVERSATION_STATS',self.conversation.identity,phase=self.conversation.phase.value,
                peer_age_ms=round((now-self.last_peer_seen)*1000),
                input_queue=self.input.queue.qsize(),output_queue=self.output_pcm_queue.qsize(),startup_frames=len(self.startup.pending),startup_dropped=self.startup.dropped,
                **self.counters)
            event('INPUT_STATS',self.conversation.identity,**vars(self.input.stats))
            if self.conversation.response:
                event('OUTPUT_PACING_STATS',self.conversation.response,**self.output_pacer.stats())

    async def check_timeouts(self, now):
        age=now-self.last_peer_seen
        if self.conversation.identity and not self.closing and age>=self.config.core_disconnect_timeout_sec:
            event('CORE_DISCONNECTED',self.conversation.identity,absent_ms=round(age*1000),
                  timeout_sec=self.config.core_disconnect_timeout_sec)
            await self.fail('peer_timeout')

    def connection_lost(self, exc):
        self.conversation.close(); self.input.close(); self.events.close()
        if self.provider_task: self.provider_task.cancel()
        for task in self.tasks: task.cancel()
        for task in self.repairs: task.cancel()
        self.input_recorder.close(); self.output_recorder.close(); self.opus_encoder.close()
        if self.xai.ws: asyncio.create_task(self.xai.close())
