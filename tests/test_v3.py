"""Offline v3 integration: actual coordinator, provider adapter and Opus pipeline."""
import asyncio
import base64
import json
import struct
import unittest
from dataclasses import replace
from bridge.config import BridgeConfig
from bridge.continuous_session import ContinuousSession
from bridge.conversation import Conversation, Phase
from bridge.conversation_protocol import Identity, Packet, Kind, Event, encode, decode
from bridge.reliable_events import ReliableEvents

class Transport:
    def __init__(self): self.sent=[]
    def sendto(self,data,addr): self.sent.append(decode(data))

class Provider:
    def __init__(self, adapter): self.adapter=adapter;self.calls=[];self.closed=False
    async def send(self, raw):
        p=json.loads(raw);self.calls.append(p)
        kind=p['type']
        confirmations={'input_audio_buffer.clear':'input_audio_buffer.cleared',
            'conversation.item.truncate':'conversation.item.truncated',
            'conversation.item.delete':'conversation.item.deleted', 'session.update':'session.updated'}
        if kind in confirmations:
            await self.adapter.handle_event({'type':confirmations[kind], 'item_id':p.get('item_id','')})
        elif kind=='response.cancel':
            await self.adapter.handle_event({'type':'response.done','response':{'id':p['response_id'],'status':'cancelled'}})
    async def close(self): self.closed=True

async def settle():
    for _ in range(20): await asyncio.sleep(0)

class ModelTests(unittest.TestCase):
    def test_startup_buffer_has_a_hard_limit(self):
        from bridge.startup_audio import StartupAudio
        b=StartupAudio(2)
        self.assertTrue(b.admit(b'first'));self.assertTrue(b.admit(b'second'))
        self.assertFalse(b.admit(b'extra'));self.assertEqual(b.dropped,1)
        self.assertEqual(list(b.pending),[b'first',b'second'])
        b.clear();self.assertFalse(b.pending)

    def test_identity_and_independent_input(self):
        c=Conversation();i=Identity(1,2,3,0);c.open(i)
        self.assertIsNone(c.speech_started());c.speech_stopped();first=c.response_started()
        self.assertEqual(first,Identity(1,2,3,1));self.assertEqual(c.identity,i)
        self.assertEqual(c.speech_started(),first);second=c.response_started()
        c.played(first);self.assertEqual(c.response,second)
        c.played(second);self.assertEqual(c.phase,Phase.LISTENING)
        c.close();self.assertIsNone(c.response_started())

    def test_v2_is_explicitly_incompatible(self):
        p=Packet(Kind.START,0,Identity(1,2,3,0),4)
        b=encode(p);self.assertEqual(b[:5],b'PIU3\x03');self.assertEqual(decode(b),p)
        for bad in [b'PIU2'+b[4:],b[:4]+b'\x02'+b[5:],b[:-1],b+b'x']:
            with self.assertRaises(ValueError): decode(bad)

class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg=replace(BridgeConfig(),save_input_wav=False,save_output_wav=False,
                    output_send_interval_sec=0,output_end_repeat_interval=0)
        self.s=ContinuousSession(cfg);self.s.transport=Transport()
        self.peer=('127.0.0.1',1234)
        self.provider=Provider(self.s.xai);self.s.xai.ws=self.provider
        self.connects=0
        async def connect():
            self.connects+=1;self.provider.closed=False;self.s.xai.ws=self.provider
            self.s.xai.responses.clear();self.s.xai.items.clear();self.s.xai.done.clear()
            self.s.xai.connected_event.set();self.s.xai.ready_event.set()
        async def receive(): await asyncio.Event().wait()
        self.s.xai.connect=connect;self.s.xai.receive_events_forever=receive
        self.s.events.start()
        self.s.tasks.append(asyncio.create_task(self.s.run_controls()))
        self.s.tasks.append(asyncio.create_task(self.s._udp_output_sender()))
        self.deliver(Kind.HELLO,Identity(11,0,0,0),1)
        self.i=Identity(11,self.s.server_nonce,1,0)
        self.deliver(Kind.START,self.i,2);await settle()
        self.assertEqual(self.s.conversation.identity,self.i)
        await self.ack_events()

    async def asyncTearDown(self):
        self.s.connection_lost(None);await settle()

    def deliver(self, kind, identity=None, seq=0, body=b'', peer=None):
        self.s.datagram_received(encode(Packet(kind,0,identity or self.i,seq,body)),peer or self.peer)

    async def ack_events(self, played=0):
        acked=set()
        for _ in range(8):
            await settle()
            for p in list(self.s.transport.sent):
                if p.kind==Kind.EVENT and (p.identity,p.sequence) not in acked:
                    self.deliver(Kind.EVENT_ACK,p.identity,p.sequence,struct.pack('!I',played))
                    acked.add((p.identity,p.sequence))

    async def response(self, name='r1', count=1):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':name}})
        token=self.s.conversation.response
        await self.s.xai.handle_event({'type':'response.output_item.added','response_id':name,
            'item':{'id':'item-'+name,'role':'assistant'}})
        for _ in range(count):
            await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':name,
                'item_id':'item-'+name,'delta':base64.b64encode(bytes(640)).decode()})
        await self.s.xai.handle_event({'type':'response.done','response':{'id':name,'status':'completed'}})
        return token

    async def test_pcm_during_response_and_next_response_ids(self):
        token=await self.response();await settle()
        self.assertFalse(any(p.kind==Kind.OPUS for p in self.s.transport.sent))
        await self.ack_events()
        self.assertTrue(any(p.kind==Kind.OPUS for p in self.s.transport.sent))
        self.deliver(Kind.PCM,self.i,0,self.s.pcm_packet_format.pack_audio_packet(0,0,320,bytes(640)))
        await settle();self.assertTrue(any(p['type']=='input_audio_buffer.append' for p in self.provider.calls))
        self.deliver(Kind.PLAYED,token,3,struct.pack('!I',20));await settle()
        self.assertEqual(self.s.conversation.phase,Phase.LISTENING)
        self.assertEqual(self.s.conversation.identity,self.i)
        next_token=await self.response('r2');await self.ack_events()
        self.assertEqual(next_token.response,token.response+1)
        self.assertFalse(any(p['type'] in ('input_audio_buffer.commit','response.create') for p in self.provider.calls))

    async def test_vad_interruption_truncates_without_clearing_new_audio(self):
        token=await self.response(count=4);await self.ack_events()
        self.provider.calls.clear()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        self.assertIsNone(self.s.conversation.response)
        self.assertFalse(self.s.xai.history_ready.is_set())
        await self.ack_events(played=40)
        self.assertTrue(self.s.xai.history_ready.is_set())
        trunc=[p for p in self.provider.calls if p['type']=='conversation.item.truncate']
        self.assertEqual(trunc[0]['audio_end_ms'],40)
        self.assertFalse(any(p['type']=='input_audio_buffer.clear' for p in self.provider.calls))
        self.assertNotIn(token,self.s.xai.items)

    async def test_unheard_response_deleted_and_stale_audio_ignored(self):
        await self.response(count=2);await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(0)
        self.assertTrue(any(p['type']=='conversation.item.delete' for p in self.provider.calls))
        before=len(self.s.transport.sent)
        await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':'r1',
            'delta':base64.b64encode(bytes(640)).decode()})
        await settle();self.assertEqual(len(self.s.transport.sent),before)

    async def test_manual_wake_uses_position_before_core_flush(self):
        token=await self.response(count=4);await self.ack_events()
        self.deliver(Kind.INTERRUPT,token,3,struct.pack('!I',40));await settle()
        await self.ack_events(0)
        self.assertTrue(any(p['type']=='conversation.item.truncate' and p['audio_end_ms']==40 for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)

    async def test_close_preempts_unacknowledged_manual_interrupt(self):
        token=await self.response(count=4);await self.ack_events()
        self.deliver(Kind.INTERRUPT,token,3,bytes(4));await settle()
        # No ACK of the interruption event: close must still finish promptly.
        self.deliver(Kind.CANCEL,self.i,4,bytes(4));await settle()
        self.assertTrue(self.provider.closed)
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(any(p.kind==Kind.CANCEL_ACK and p.sequence==4 for p in self.s.transport.sent))
        self.assertFalse(self.s.tasks[0].done())

    async def test_duplicate_start_and_played_are_idempotent(self):
        self.deliver(Kind.START,self.i,2);await settle()
        self.assertEqual(self.s.last_conversation,1)
        token=await self.response();await self.ack_events()
        for _ in range(2):
            self.deliver(Kind.PLAYED,token,3,bytes(4));await settle()
        self.assertEqual(self.s.conversation.phase,Phase.LISTENING)
        self.assertEqual(sum(p.kind==Kind.PLAYED_ACK for p in self.s.transport.sent),2)

    async def test_close_disconnects_and_next_wake_opens_fresh(self):
        token=await self.response(count=4);await self.ack_events()
        self.deliver(Kind.CANCEL,token,3,struct.pack('!I',40));await settle()
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(self.provider.closed)
        self.assertIsNone(self.s.provider_task)
        self.assertFalse(self.s.xai.connected_event.is_set())
        self.assertFalse(hasattr(self.s,'context_expires_at'))
        self.i=Identity(self.i.client,self.i.server,2,0)
        self.deliver(Kind.START,self.i,4);await settle();await self.ack_events()
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertEqual(self.connects,2)
        self.assertFalse(self.s.xai.items)
        await self.response('r2');await settle()
        self.assertEqual(self.s.events.pending[0].sequence,2) # READY was event 1

    async def test_startup_audio_is_retained_until_provider_ready(self):
        self.deliver(Kind.CANCEL,self.i,3,bytes(4));await settle()
        gate=asyncio.Event();original=self.s.xai.connect
        async def delayed(): await gate.wait();await original()
        self.s.xai.connect=delayed
        self.i=Identity(self.i.client,self.i.server,2,0)
        self.deliver(Kind.START,self.i,4);await settle()
        self.provider.calls.clear()
        for seq in range(2):
            self.deliver(Kind.PCM,self.i,seq,self.s.pcm_packet_format.pack_audio_packet(seq,seq*320,320,bytes([seq])*640))
        self.assertEqual(len(self.s.startup.pending),2)
        self.assertFalse(any(p['type']=='input_audio_buffer.append' for p in self.provider.calls))
        gate.set();await settle();await self.ack_events()
        appends=[base64.b64decode(p['audio']) for p in self.provider.calls if p['type']=='input_audio_buffer.append']
        self.assertEqual(appends,[bytes(640),bytes([1])*640])
        self.assertTrue(self.s.input_live)

    async def test_connect_failure_does_not_reconnect_in_background(self):
        self.deliver(Kind.CANCEL,self.i,3,bytes(4));await settle()
        attempts=[]
        async def failing(): attempts.append(1);raise OSError('offline')
        self.s.xai.connect=failing
        self.i=Identity(self.i.client,self.i.server,2,0)
        self.deliver(Kind.START,self.i,4);await settle()
        self.assertIsNone(self.s.conversation.identity)
        self.assertEqual(len(attempts),1)
        self.assertTrue(any(p.kind==Kind.ERROR for p in self.s.transport.sent))

    async def test_vad_cancellation_waits_for_specific_old_response(self):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'old'}})
        await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':'old',
            'item_id':'old-item','delta':base64.b64encode(bytes(1280)).decode()})
        await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(20)
        self.assertFalse(self.s.xai.history_ready.is_set())
        await self.s.xai.handle_event({'type':'response.done','response':{'id':'old','status':'cancelled'}})
        await settle();self.assertTrue(self.s.xai.history_ready.is_set())
        self.assertTrue(any(p['type']=='conversation.item.truncate' for p in self.provider.calls))

    async def test_foreign_ping_cannot_redirect(self):
        self.deliver(Kind.PING,peer=('127.0.0.2',9999))
        self.assertEqual(self.s.remote_addr,self.peer)
        self.assertEqual(self.s.counters['foreign'],1)

    async def test_pcm_end_and_manual_commit_rejected(self):
        self.deliver(Kind.PCM,self.i,0,self.s.pcm_packet_format.pack_end_packet(0,0))
        self.deliver(Kind.COMMIT,self.i,4,bytes(8));await settle()
        self.assertEqual(self.s.counters['invalid'],1)
        self.assertEqual(self.s.conversation.identity,self.i)

class ReliableTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_backend_does_not_open_provider(self):
        from unittest.mock import AsyncMock
        s=ContinuousSession(replace(BridgeConfig(),save_input_wav=False,save_output_wav=False))
        s.xai.connect=AsyncMock()
        s.connection_made(Transport());await settle()
        s.xai.connect.assert_not_awaited()
        s.connection_lost(None);await settle()


    async def test_ack_identity_and_retry(self):
        sent=[];failed=[]
        async def fail(reason): failed.append(reason)
        events=ReliableEvents(sent.append,fail);events.start()
        i=Identity(1,2,3,4);future=events.emit(i,Event.RESPONSE_STARTED,1)
        await asyncio.sleep(.17)
        self.assertGreaterEqual(len(sent),2)
        events.ack(Packet(Kind.EVENT_ACK,0,Identity(1,2,3,5),1,bytes(4)))
        self.assertFalse(future.done())
        events.ack(Packet(Kind.EVENT_ACK,0,i,1,struct.pack('!I',123)))
        self.assertEqual(await future,123);events.close();await settle()
        self.assertFalse(failed)
