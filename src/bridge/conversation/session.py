"""UDP v3 conversation coordinator; one instance per admitted client."""
import asyncio
import secrets
import math
import struct
import time
from bridge.audio.pipeline import AudioPipeline
from bridge.transport.downlink_recovery import DownlinkRecovery
from bridge.conversation.state import Conversation
from bridge.transport.protocols.conversation import Identity, Packet, Kind, Event, encode, decode, CAP_EYES_PLAN, CAP_DEVICE_MAC
from bridge.expressions.delivery import EyePlanDelivery
from bridge.transport.reliable_events import ReliableEvents
from bridge.providers import create_provider
from bridge.transport.ingress import InboundSession
from bridge.audio.provider_input import ProviderInput
from bridge.diagnostics.logging import event
from bridge.audio.startup_audio import StartupAudio
from bridge.conversation.interruption import InterruptionCoordinator

class ContinuousSession(AudioPipeline):
    def __init__(self, config):
        for name in ('core_disconnect_timeout_sec','history_repair_timeout_sec',
                     'history_item_timeout_sec','history_confirmation_timeout_sec'):
            if not math.isfinite(getattr(config,name)) or getattr(config,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(config)
        self.transport = self.remote_addr = self.client_nonce = None
        self.device_mac = None
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
        # Keep the internal xai alias for existing pipeline/legacy compatibility.
        self.xai = create_provider(config, self.output_pcm_queue, self.output_valid, self.provider_event)
        self.input = self.new_input()
        self.downlink = DownlinkRecovery(config,self._send)
        self.eye_plans = EyePlanDelivery(self._wire)
        self.peer_eye_offer = None
        self.eyes_capable = False

    def new_input(self):
        self.provider_input=ProviderInput(self.config,self._accept_audio,self.xai.wait_input_ready)
        worker = InboundSession(self.pcm_packet_format, self.config, self.provider_input.admit,
            self.ignore_end, self.no_commit, nack_sink=self.nack, silence_sink=self.provider_input.admit)
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
                           (self.watchdog(), 'conversation-watchdog'),
                           (self.recover_output(), 'downlink-recovery')]:
            self.tasks.append(asyncio.create_task(coro, name=name))

    def datagram_received(self, data, addr):
        try: p = decode(data)
        except (ValueError, struct.error): self.counters['invalid'] += 1; return
        allowed_flags = CAP_EYES_PLAN | CAP_DEVICE_MAC if p.kind == Kind.HELLO else 0
        if p.flags & ~allowed_flags or not p.identity.client: self.counters['invalid'] += 1; return
        if p.kind == Kind.HELLO:
            new_peer = self.client_nonce is None or p.identity.client != self.client_nonce
            mac = p.body if p.flags & CAP_DEVICE_MAC else None
            if (p.identity.server or p.identity.turn or p.identity.response or
                (mac is None and p.body) or
                (mac is not None and (len(mac)!=6 or not any(mac) or mac[0]&1))):
                self.counters['invalid'] += 1; return
            # First admission pins the device for this server process. The MAC
            # is a declared identity, not cryptographic proof of origin.
            if self.device_mac is not None and mac != self.device_mac:
                self.counters['foreign'] += 1; return
            if self.client_nonce is None:
                self.client_nonce = p.identity.client; self.remote_addr = addr; self.device_mac = mac
                event('CORE_ADMITTED', device_mac=mac.hex(':') if mac else None,
                      peer=addr, client=f'{self.client_nonce:016x}')
            if addr != self.remote_addr and not (new_peer and mac is not None and mac == self.device_mac):
                self.counters['foreign'] += 1; return
            if p.identity.client != self.client_nonce:
                if p.identity.client in self.retired: return
                if self.conversation.identity or self.pending or not self.ready():
                    if self.conversation.identity and not self.peer_resetting:
                        self.peer_resetting=True; self.schedule(self.fail('peer_restarted'))
                    return
                old_addr = self.remote_addr
                self.retired = (self.retired + [self.client_nonce])[-32:]
                self.client_nonce = p.identity.client
                self.remote_addr = addr
                self.device_mac = mac
                self.server_nonce = secrets.randbits(64) or 1
                self.last_conversation = 0; self.acks.clear()
                self.events.reset(); self.downlink.reset()
                event('CORE_REBOUND', device_mac=mac.hex(':') if mac else None,
                      old_peer=old_addr, peer=addr, client=f'{self.client_nonce:016x}')
            offered = bool(p.flags & CAP_EYES_PLAN)
            if new_peer:
                self.peer_eye_offer = offered
                self.eyes_capable = offered and self.config.expression_udp_enabled
                self.eye_plans.reset()
            elif offered != self.peer_eye_offer:
                self.counters['invalid'] += 1; return  # Capabilities are fixed for a client nonce.
            self.last_peer_seen=time.monotonic()
            capabilities = (CAP_EYES_PLAN if self.eyes_capable else 0) | (CAP_DEVICE_MAC if mac else 0)
            self._send(Kind.HELLO_ACK, Identity(self.client_nonce,self.server_nonce,0,0),p.sequence,flags=2|capabilities|int(self.ready()))
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
                self.conversation.close(); self.input.close(); self.provider_input.close(); self.reset_output()
            self.controls.put_nowait(p); return
        if current is None or p.identity.client != current.client or p.identity.turn != current.turn:
            self.counters['stale'] += 1; return
        if p.kind == Kind.EYES_PLAN_ACK:
            if p.identity != self.conversation.response:
                self.counters['stale'] += 1; return
            if self.eye_plans.acknowledge(p): self.last_peer_seen=time.monotonic()
            else: self.counters['invalid'] += 1
            return
        if p.kind in (Kind.DOWNLINK_NACK,Kind.AUDIO_END_ACK):
            if p.identity != self.conversation.response:
                self.counters['stale'] += 1; return
            accepted = (self.downlink.request(p) if p.kind==Kind.DOWNLINK_NACK
                        else self.downlink.acknowledge(p))
            if accepted: self.last_peer_seen=time.monotonic()
            else: self.counters['invalid'] += 1
            return
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
        if kind == 'expression.plan':
            if self.output_valid(message['identity']) and self.output_sequence == 0 and self.expressions.plan is None:
                self.expressions.prepare(message['identity'], message['plan'])
            return
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

    def send_expression_plan(self, token):
        # Called at first Opus transmission, after RESPONSE_STARTED ACK. No wait,
        # per-step tasks, or dependency on the visual receipt acknowledgement.
        if (not self.eyes_capable or not self.output_valid(token) or
                self.expressions.identity != token or self.expressions.plan is None):
            return
        try:
            self.eye_plans.start(token, self.expressions.plan)
        except ValueError:
            event('EYES_PLAN_REJECTED', token, reason='invalid_wire_plan')

    async def allow_output(self, token, future):
        try:
            await future
            if self.output_valid(token): self.output_gate.set()
        except asyncio.CancelledError: pass

    async def run_controls(self):
        while True:
            p = await self.controls.get()
            try:
                if self.peer_resetting:
                    self.ack(p,False); continue
                if p.kind == Kind.START:
                    if self.conversation.identity or p.identity.turn <= self.last_conversation or not self.ready():
                        self.ack(p,False); continue
                    self.conversation.open(p.identity); self.last_conversation=p.identity.turn
                    self.last_activity=time.monotonic()
                    self.events.reset(); self.startup.clear(); self.input_live=False
                    self.input=self.new_input(); self.reset_output()
                    self.provider_task=asyncio.create_task(self.run_provider(p.identity),name='active-provider')
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
                        event('OUTPUT_RECOVERY_COMPLETE',token,**self.downlink.stats)
                        self.downlink.reset()
                        self.eye_plans.reset()
                        self.conversation.played(token); self.last_activity=time.monotonic()
                        self.xai.release_response(token)
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
        self.conversation.close(); self.input.close(); self.provider_input.close(); self.input_live=False
        self.eye_plans.reset()  # Before awaiting history/input/provider cleanup.
        self.startup.clear(); self.events.reset()
        await self.interruptions.close()
        for task in list(self.repairs):
            if task is not asyncio.current_task(): task.cancel()
        if self.input.task:
            try: await self.input.task
            except asyncio.CancelledError: pass
        await self.provider_input.wait_closed()
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
                receiver=asyncio.create_task(self.xai.receive_events_forever(),name='provider-events')
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
            self.provider_input.start(token)
            self.input.start()
            # Drain without flooding the normal 16-frame ingress queue. New PCM
            # remains in startup until the transfer completes, preserving order.
            while self.startup.pending and self.conversation.identity==token:
                if (self.input.buffered_audio_frames>=self.config.input_queue_frames or
                    self.input.buffered_audio_frames+len(self.provider_input.pending)>=self.provider_input.capacity):
                    await asyncio.sleep(.001); continue
                self.input.admit(self.startup.pending.popleft())
            event('STARTUP_TRANSFER_COMPLETE', token,
                  buffered_frames=self.input.buffered_audio_frames,
                  dropped_full=self.input.stats.dropped_full,
                  startup_dropped=self.startup.dropped)
            self.input_live=True
            await receiver
        except asyncio.CancelledError: raise
        except Exception as exc:
            event('PROVIDER_DISCONNECTED',token,provider=self.config.voice_provider,
                  error=type(exc).__name__,reason=str(exc)[:160])
        finally:
            if receiver:
                receiver.cancel()
                try: await receiver
                except asyncio.CancelledError: pass
                except Exception: pass
            await self.xai.close()
            if self.conversation.identity==token:
                await self.fail('provider_closed')

    async def recover_output(self):
        while True:
            await asyncio.sleep(.02)
            if self.eye_plans.identity is not None and not self.output_valid(self.eye_plans.identity):
                self.eye_plans.reset()
            else:
                self.eye_plans.poll()
            if not self.downlink.poll():
                await self.fail('output_end_unconfirmed')

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
            event('PROVIDER_INPUT_STATS',self.conversation.identity,**self.provider_input.snapshot())
            if self.conversation.response:
                event('OUTPUT_PACING_STATS',self.conversation.response,**self.output_pacer.stats())
                event('OUTPUT_RECOVERY_STATS',self.conversation.response,**self.downlink.stats)
                event('EYES_PLAN_STATS',self.conversation.response,enabled=self.eyes_capable,**self.eye_plans.stats)

    async def check_timeouts(self, now):
        age=now-self.last_peer_seen
        if self.conversation.identity and not self.closing and age>=self.config.core_disconnect_timeout_sec:
            event('CORE_DISCONNECTED',self.conversation.identity,absent_ms=round(age*1000),
                  timeout_sec=self.config.core_disconnect_timeout_sec)
            await self.fail('peer_timeout')

    def connection_lost(self, exc):
        self.conversation.close(); self.input.close(); self.provider_input.close(); self.events.close(); self.downlink.reset()
        self.eye_plans.reset()
        self.expressions.reset('connection_lost')
        if self.provider_task: self.provider_task.cancel()
        for task in self.tasks: task.cancel()
        for task in self.repairs: task.cancel()
        self.input_recorder.close(); self.output_recorder.close(); self.opus_encoder.close()
        if self.xai.ws: asyncio.create_task(self.xai.close())
