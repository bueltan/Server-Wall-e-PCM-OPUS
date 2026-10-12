"""UDP v2 client lifecycle. Future registry entries each own one instance."""
import asyncio
import secrets
import struct
from bridge.legacy.session_v1 import ClientSession
from bridge.transport.protocols.session_v2 import Identity,Packet,Kind,encode,decode
from bridge.providers.xai.tagged import TaggedXai
from bridge.transport.ingress import InboundSession
from bridge.diagnostics.logging import event

class ManagedSession(ClientSession):
    def __init__(self, config):
        super().__init__(config)
        self.server_nonce=secrets.randbits(64) or 1
        self.client_nonce=None
        self.retired_clients=[]
        self.active=None
        self.last_turn=0
        self.control_queue=asyncio.Queue(maxsize=8)
        self.control_task=None
        self.diagnostics_task=None
        self.counters=dict(invalid=0,foreign=0,stale=0,duplicate_controls=0,pending_repeats=0,
                           control_full=0,nacks=0,stale_output=0,opus_frames=0)
        self.pending_controls=set();self.acks={}
        self.commit_packet=None
        self.output_finished=False
        self.xai=TaggedXai(config,self.output_pcm_queue,self.valid)
        self.input.close()
        self.input=self._new_input()

    def _new_input(self):
        worker=InboundSession(self.pcm_packet_format,self.config,
            self._accept_audio,self._accept_end,self._commit_input,nack_sink=self._nack, silence_sink=self._accept_audio)
        worker.log_identity=self.active
        return worker

    def valid(self, token): return token is not None and token==self.active
    def ready(self): return self.xai.connected_event.is_set() and self.xai.ready_event.is_set()
    def connection_made(self, transport):
        super().connection_made(transport)
        self.control_task=asyncio.create_task(self._control_worker(),name='client-controls')
        self.diagnostics_task=asyncio.create_task(self._diagnostics(),name='client-diagnostics')

    def _summary(self, cause, identity=None):
        event("SESSION_STATS", identity or self.active, cause=cause,
              input_queue=self.input.queue.qsize(), reorder=len(self.input.reorder.pending),
              output_queue=self.output_pcm_queue.qsize(), pending_pcm=len(self.output_pending_pcm),
              controls=self.control_queue.qsize(), ready=self.ready(),
              **self.counters)
        event("INPUT_STATS", identity or self.active, cause=cause, **vars(self.input.stats))

    async def _diagnostics(self):
        while True:
            await asyncio.sleep(5)
            self._summary("periodic")

    def _send(self, kind, identity, sequence=0, body=b'', flags=0):
        if self.transport and self.remote_addr:
            self.transport.sendto(encode(Packet(kind,flags,identity,sequence,body)),self.remote_addr)

    def _key(self,p): return (p.kind,p.identity.turn,p.sequence)
    def _ack(self,p,ok=True):
        kind={Kind.START:Kind.START_ACK,Kind.COMMIT:Kind.COMMIT_ACK,Kind.CANCEL:Kind.CANCEL_ACK}[p.kind]
        reply=encode(Packet(kind,int(ok),p.identity,p.sequence))
        self.acks[self._key(p)]=reply;self.pending_controls.discard(self._key(p))
        while len(self.acks)>64: self.acks.pop(next(iter(self.acks)))
        self.transport.sendto(reply,self.remote_addr)
        event("CONTROL_ACK",p.identity,kind=kind.name,request=p.sequence,ok=ok)

    def datagram_received(self,data,addr):
        try: p=decode(data)
        except (ValueError,struct.error):
            self.counters["invalid"]+=1;return
        if p.flags or p.identity.client==0:
            self.counters["invalid"]+=1;return
        if p.kind==Kind.HELLO:
            if p.identity.server or p.identity.turn or p.identity.response or p.body: return
            if self.client_nonce is None:
                self.client_nonce=p.identity.client;self.remote_addr=addr
                event("PEER_BOUND",Identity(self.client_nonce,self.server_nonce,0,0),peer=addr)
            if addr!=self.remote_addr: return
            if p.identity.client!=self.client_nonce:
                if p.identity.client in self.retired_clients: return
                # Same endpoint, new CORE boot: clear old remote state before adoption.
                if self.active is not None:
                    old=self.active
                    cancel=Packet(Kind.CANCEL,0,old,0,struct.pack('!I',0))
                    if not self.control_queue.full():
                        self.pending_controls.add(self._key(cancel))
                        event("PEER_RESET_CANCEL",old,new_client=f"{p.identity.client:016x}")
                        self.active=None;self.input.close();self.control_queue.put_nowait(cancel)
                    return
                if self.pending_controls or not self.ready(): return
                self.retired_clients.append(self.client_nonce)
                self.retired_clients=self.retired_clients[-32:]
                self.client_nonce=p.identity.client;self.last_turn=0;self.acks.clear()
                self.server_nonce=secrets.randbits(64) or 1
                event("PEER_REPLACED",Identity(self.client_nonce,self.server_nonce,0,0),peer=addr)
            identity=Identity(self.client_nonce,self.server_nonce,0,0)
            self._send(Kind.HELLO_ACK,identity,p.sequence,flags=int(self.ready()));return
        if self.client_nonce is None:
            if p.kind==Kind.PING:
                self.transport.sendto(encode(Packet(Kind.ERROR,0,p.identity,p.sequence)),addr)
            return
        if addr!=self.remote_addr or p.identity.client!=self.client_nonce:
            self.counters["foreign"]+=1;return
        if p.identity.server!=self.server_nonce:
            self.counters["stale"]+=1;self._send(Kind.ERROR,p.identity,p.sequence);return
        if p.kind==Kind.PING:
            self._send(Kind.PONG,p.identity,p.sequence,flags=int(self.ready()));return
        if not p.identity.turn or p.identity.response!=1: return
        if p.kind in (Kind.START,Kind.COMMIT,Kind.CANCEL):
            if (p.kind==Kind.START and p.body) or (p.kind==Kind.COMMIT and len(p.body)!=8) or (p.kind==Kind.CANCEL and len(p.body)!=4): return
            key=self._key(p)
            if key in self.acks:
                self.counters["duplicate_controls"]+=1
                event("CONTROL_DUPLICATE",p.identity,kind=p.kind.name,request=p.sequence)
                self.transport.sendto(self.acks[key],addr);return
            if key in self.pending_controls:
                self.counters["pending_repeats"]+=1;return
            if self.control_queue.full():
                self.counters["control_full"]+=1;event("CONTROL_QUEUE_FULL",p.identity,kind=p.kind.name);self._send(Kind.ERROR,p.identity,p.sequence);return
            self.pending_controls.add(key)
            event("CONTROL_RX",p.identity,kind=p.kind.name,request=p.sequence,
                  played_ms=int.from_bytes(p.body,"big") if p.kind==Kind.CANCEL else None)
            # Invalidate immediately; cancellation must outrank buffered audio.
            if p.kind==Kind.CANCEL and self.valid(p.identity):
                self._summary("cancel_admission",p.identity)
                self.active=None;self.input.close()
            self.control_queue.put_nowait(p);return
        if p.kind==Kind.MISSING and self.valid(p.identity) and not p.body:
            self.input.stats.cache_misses+=1;return
        if p.kind==Kind.PCM and self.valid(p.identity) and not self.input.finished:
            try: inner=self.input.validate(p.body)
            except ValueError:
                self.counters["invalid"]+=1;return
            if inner['seq']!=p.sequence:
                self.counters["invalid"]+=1;return
            self.input.admit(p.body)
        elif p.kind==Kind.PCM:
            self.counters["stale"]+=1

    async def _control_worker(self):
        while True:
            p=await self.control_queue.get()
            try:
                if p.kind==Kind.START:
                    if p.identity.turn<=self.last_turn or not self.ready() or (self.active is not None and not self.output_finished):
                        event("START_REJECTED",p.identity,ready=self.ready(),last_turn=self.last_turn,
                              active_turn=self.active.turn if self.active else None)
                        self._ack(p,False);continue
                    if self.active is not None and not self.xai.generation_done.is_set():
                        try: await asyncio.wait_for(self.xai.generation_done.wait(), 1.0)
                        except TimeoutError:
                            self._ack(p,False);continue
                    # Serialize remote buffer setup with commit/cancellation.
                    async with self.xai.operation_lock:
                        if not self.ready(): self._ack(p,False);continue
                        self.input.close()
                        if self.input.task:
                            try: await self.input.task
                            except asyncio.CancelledError: pass
                        self.xai.items.clear();self.xai.output_done.clear()
                        self.active=p.identity;self.last_turn=p.identity.turn;self.output_finished=False
                        self.input=self._new_input();self.input.start();self.commit_packet=None
                        self._reset_downlink_turn_state()
                    event("TURN_START",p.identity,request=p.sequence)
                    self._ack(p)
                elif p.kind==Kind.CANCEL:
                    # Old/tombstoned turn cancellation must not cancel another turn.
                    if self.active is not None and self.active!=p.identity:
                        self._ack(p,p.identity.turn<=self.last_turn);continue
                    self.active=None;self.last_turn=max(self.last_turn,p.identity.turn)
                    self.input.close()
                    if self.input.task:
                        try: await self.input.task
                        except asyncio.CancelledError: pass
                    self.inbound_started=False;self.input_recorder.close();self._reset_downlink_turn_state()
                    ok=await self.xai.cancel_turn(p.identity,int.from_bytes(p.body,'big'))
                    self._ack(p,ok)
                elif p.kind==Kind.COMMIT:
                    if not self.valid(p.identity): self._ack(p,False);continue
                    if self.commit_packet is not None:
                        # Different request ID for the same COMMIT cannot generate twice.
                        if self.input.finished: self._ack(p)
                        else: self._ack(p,False)
                        continue
                    self.commit_packet=p
                    seq,pts=struct.unpack('!II',p.body)
                    event("INPUT_CLOSE",p.identity,final_seq=seq,final_pts=pts)
                    self.input.admit(self.pcm_packet_format.pack_end_packet(seq,pts))
                    if not self.input.admit_commit(): self._ack(p,False)
            except Exception as exc:
                event('CONTROL_ERROR',p.identity,kind=p.kind.name,error=type(exc).__name__);self.active=None;self._ack(p,False)

    async def _accept_audio(self,packet):
        token=self.active
        if token is None: return False
        if not self.inbound_started:
            event("INPUT_FIRST_APPEND",token,seq=packet['seq'],bytes=len(packet['payload']))
        return await super()._accept_audio(packet)

    async def _commit_input(self):
        p=self.commit_packet
        if p is None or not self.valid(p.identity): return False
        self.inbound_started=False;self.input_recorder.close()
        ok=await self.xai.commit_turn(p.identity)
        if self.valid(p.identity): self._ack(p,ok)
        return ok

    def _nack(self,base,mask):
        if self.active is not None:
            self.counters['nacks']+=1
            self._send(Kind.NACK,self.active,base,struct.pack('!IH',base,mask))

    async def _udp_output_sender(self):
        while True:
            token,chunk=await self.output_pcm_queue.get()
            if not self.valid(token):
                self.counters["stale_output"]+=1;continue
            if chunk is None:
                await self._flush_tagged(token,True)
                if not self.valid(token): continue
                for i in range(self.config.output_end_repeat_count):
                    if not self.valid(token): break
                    inner=self.opus_packet_format.pack_end_packet(self.output_sequence,self.output_pts_samples)
                    self._send(Kind.OPUS,token,self.output_sequence,inner)
                    if i+1<self.config.output_end_repeat_count: await asyncio.sleep(self.config.output_end_repeat_interval)
                if not self.valid(token): continue
                self.output_active=False;self.output_finished=True;self.output_recorder.close()
                event("OUTPUT_END",token,final_seq=self.output_sequence,pts=self.output_pts_samples)
                self._summary("output_end",token)
                # Keep identity for CANCEL while CORE still drains its jitter/DMA.
                continue
            if not self.output_active:
                self.output_active=True
                event("OUTPUT_FIRST_PCM",token,bytes=len(chunk))
                if self.config.save_output_wav:
                    self.output_recorder.start('downlink',f'_{token.turn}')
            if self.config.save_output_wav: self.output_recorder.write(chunk)
            self.output_pending_pcm.extend(chunk)
            await self._flush_tagged(token,False)

    async def _flush_tagged(self,token,final):
        while self.valid(token) and (len(self.output_pending_pcm)>=self.config.opus_frame_bytes or (final and self.output_pending_pcm)):
            frame=bytes(self.output_pending_pcm[:self.config.opus_frame_bytes])
            del self.output_pending_pcm[:self.config.opus_frame_bytes]
            frame+=b'\0'*(self.config.opus_frame_bytes-len(frame))
            payload=self.opus_encoder.encode_pcm16_frame(frame)
            inner=self.opus_packet_format.pack_audio_packet(self.output_sequence,self.output_pts_samples,self.config.opus_frame_samples,payload)
            self._send(Kind.OPUS,token,self.output_sequence,inner)
            self.counters["opus_frames"]+=1
            self.output_sequence+=1;self.output_pts_samples+=self.config.opus_frame_samples
            await asyncio.sleep(self.config.output_send_interval_sec)

    async def _run_xai(self):
        while True:
            try:
                await self.xai.connect();await self.xai.receive_events_forever()
            except Exception as exc: event('XAI_DISCONNECTED',self.active,error=type(exc).__name__,reason=str(exc)[:200])
            finally:
                self.xai.connected_event.clear();self.xai.ready_event.clear()
                if self.active is not None:
                    old=self.active;self.active=None;self.input.close();self._send(Kind.ERROR,old)
                await self.xai.close()
            await asyncio.sleep(2)

    def connection_lost(self,exc):
        self._summary("shutdown")
        self.active=None
        if self.diagnostics_task: self.diagnostics_task.cancel()
        if self.control_task: self.control_task.cancel()
        super().connection_lost(exc)
