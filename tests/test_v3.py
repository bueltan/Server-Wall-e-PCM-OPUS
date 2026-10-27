"""Offline v3 integration: actual coordinator, provider adapter and Opus pipeline."""
import asyncio
import base64
import json
import struct
import unittest
from dataclasses import replace
from bridge.config import BridgeConfig
from bridge.conversation.session import ContinuousSession
from bridge.conversation.state import Conversation, Phase
from bridge.transport.protocols.conversation import Identity, Packet, Kind, Event, encode, decode
from bridge.transport.reliable_events import ReliableEvents

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
        from bridge.audio.startup_audio import StartupAudio
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

    def test_session_timeouts_reject_disabled_or_nonfinite_values(self):
        for name in ('core_disconnect_timeout_sec','history_repair_timeout_sec',
                     'history_item_timeout_sec','history_confirmation_timeout_sec'):
            for value in (0,-1,float('nan'),float('inf')):
                with self.subTest(name=name,value=value), self.assertRaises(ValueError):
                    ContinuousSession(replace(BridgeConfig(),**{name:value}))

    def test_v2_is_explicitly_incompatible(self):
        p=Packet(Kind.START,0,Identity(1,2,3,0),4)
        b=encode(p);self.assertEqual(b[:5],b'PIU3\x03');self.assertEqual(decode(b),p)
        for bad in [b'PIU2'+b[4:],b[:4]+b'\x02'+b[5:],b[:-1],b+b'x']:
            with self.assertRaises(ValueError): decode(bad)

class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg=replace(BridgeConfig(),voice_provider="xai",save_input_wav=False,save_output_wav=False,
                    output_send_interval_sec=0,output_end_repeat_interval=0,provider_input_send_interval_sec=0)
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

    async def test_unconfirmed_end_fails_instead_of_reporting_played(self):
        token=await self.response();await self.ack_events();await settle()
        self.s.downlink.config=replace(self.s.config,output_end_timeout_sec=.01)
        task=asyncio.create_task(self.s.recover_output());self.s.tasks.append(task)
        for _ in range(30):
            if self.s.conversation.identity is None and self.provider.closed: break
            await asyncio.sleep(.01)
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(self.provider.closed)
        self.assertTrue(any(p.kind==Kind.ERROR and p.identity.turn==token.turn for p in self.s.transport.sent))

    async def test_downlink_real_pipeline_recovery_end_ack_and_stale_ids(self):
        token=await self.response(count=4);await self.ack_events();await settle()
        packets=[p for p in self.s.transport.sent if p.kind==Kind.OPUS and p.identity==token]
        self.assertEqual(len(packets),4)
        end=[p for p in self.s.transport.sent if p.kind==Kind.AUDIO_END][-1]
        self.assertEqual(end.sequence,4)
        self.deliver(Kind.DOWNLINK_NACK,token,1,struct.pack('!IH',1,3))
        self.s.downlink.poll()
        repeated=[p for p in self.s.transport.sent if p.kind==Kind.OPUS and p.identity==token][4:]
        self.assertEqual(repeated,packets[1:3])
        self.deliver(Kind.AUDIO_END_ACK,token,4)
        self.assertIsNone(self.s.downlink.end)
        # END receipt does not mark playback finished or remove response identity.
        self.assertEqual(self.s.conversation.response,token)
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(0)
        before=len(self.s.transport.sent)
        self.deliver(Kind.DOWNLINK_NACK,token,1,struct.pack('!IH',1,3))
        self.s.downlink.poll()
        self.assertEqual(len(self.s.transport.sent),before)

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

    async def test_startup_transfer_waits_for_reorder_capacity_on_sequence_gap(self):
        self.deliver(Kind.CANCEL,self.i,3,bytes(4));await settle()
        gate=asyncio.Event();original=self.s.xai.connect
        async def delayed(): await gate.wait();await original()
        self.s.xai.connect=delayed
        self.i=Identity(self.i.client,self.i.server,2,0)
        self.deliver(Kind.START,self.i,4);await settle()
        self.provider.calls.clear()
        # Enough backlog to overflow the 16-frame reorder buffer if startup
        # only checks queue depth. Frame 1 is deliberately unavailable.
        sequences=[0]+list(range(2,81))
        for seq in sequences:
            pcm=struct.pack('<h',seq)*320
            self.deliver(Kind.PCM,self.i,seq,self.s.pcm_packet_format.pack_audio_packet(seq,seq*320,320,pcm))
        gate.set();await self.ack_events()
        async def drained():
            while not self.s.input_live or len([p for p in self.provider.calls
                    if p['type']=='input_audio_buffer.append'])<len(sequences)+1:
                await asyncio.sleep(.001)
        await asyncio.wait_for(drained(),2)
        self.assertEqual(self.s.input.stats.dropped_full,0)
        self.assertEqual(self.s.input.stats.missing,1)
        self.assertEqual(self.s.input.stats.silence_frames,1)
        appends=[base64.b64decode(p['audio']) for p in self.provider.calls
                 if p['type']=='input_audio_buffer.append']
        self.assertEqual(appends,[bytes(640),bytes(640)]+[struct.pack('<h',n)*320 for n in range(2,81)])
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertFalse(self.s.startup.pending)

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

    async def test_vad_repair_does_not_require_response_done_when_item_is_known(self):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'old'}})
        await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':'old',
            'item_id':'old-item','delta':base64.b64encode(bytes(1280)).decode()})
        await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(20)
        self.assertTrue(self.s.xai.history_ready.is_set())
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

    async def test_disconnect_timeout_closes_provider_even_while_playing(self):
        token=await self.response(count=4);await self.ack_events()
        seen=self.s.last_peer_seen
        await self.s.check_timeouts(seen+self.s.config.core_disconnect_timeout_sec-.01)
        self.assertEqual(self.s.conversation.response,token)
        await self.s.check_timeouts(seen+self.s.config.core_disconnect_timeout_sec)
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(self.provider.closed)
        self.assertFalse(self.s.input_live)
        self.assertTrue(self.s.input.task.done())
        self.assertIsNone(self.s.provider_task)
        self.assertTrue(self.s.output_pcm_queue.empty())
        self.assertFalse(self.s.startup.pending)

    async def test_only_valid_core_traffic_refreshes_disconnect_timer(self):
        self.s.last_peer_seen=1
        self.deliver(Kind.PING,peer=('127.0.0.2',1234))
        self.deliver(Kind.PING,Identity(11,self.s.server_nonce+1,0,0))
        self.deliver(Kind.PING,Identity(11,self.s.server_nonce,0,0),body=b'invalid')
        self.deliver(Kind.EVENT_ACK,self.i,999,bytes(4))
        self.deliver(Kind.PCM,self.i,body=b'invalid')
        self.assertEqual(self.s.last_peer_seen,1)
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_stopped'})
        self.assertEqual(self.s.last_peer_seen,1)
        self.deliver(Kind.PING,Identity(11,self.s.server_nonce,0,0))
        self.assertGreater(self.s.last_peer_seen,1)

    async def test_disconnect_cancels_history_wait_and_next_conversation_can_open(self):
        await self.response();await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await settle()  # deliberately do not acknowledge CORE playback
        self.assertTrue(self.s.interruptions.pending)
        await self.s.check_timeouts(self.s.last_peer_seen+self.s.config.core_disconnect_timeout_sec)
        self.assertTrue(self.provider.closed)
        self.assertFalse(self.s.interruptions.pending)
        self.assertTrue(self.s.xai.history_ready.is_set())
        self.i=Identity(self.i.client,self.i.server,2,0)
        self.deliver(Kind.START,self.i,5);await settle();await self.ack_events()
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertEqual(self.connects,2)

    async def test_created_then_interrupted_repairs_late_item_without_done(self):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'early'}})
        token=self.s.conversation.response
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(0)
        self.assertTrue(self.s.interruptions.pending)
        await self.s.xai.handle_event({'type':'response.output_item.added','response_id':'early',
            'item':{'id':'early-item','role':'assistant'}})
        await settle()
        self.assertTrue(any(p['type']=='conversation.item.delete' and p['item_id']=='early-item' for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertFalse(self.s.interruptions.pending)
        self.assertTrue(self.s.xai.history_ready.is_set())
        self.assertFalse(any(p.kind==Kind.OPUS and p.identity==token for p in self.s.transport.sent))

    async def early_interruption(self):
        self.s.xai.config=replace(self.s.xai.config,history_item_timeout_sec=.01)
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'early'}})
        token=self.s.conversation.response
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(0)
        return token

    async def finish_repair(self):
        async def finished():
            while self.s.interruptions.pending: await asyncio.sleep(.001)
        await asyncio.wait_for(finished(),1)

    async def test_early_vad_cancellation_keeps_conversation_and_next_response(self):
        token=await self.early_interruption()
        await self.finish_repair()
        cancels=[p for p in self.provider.calls if p['type']=='response.cancel']
        self.assertEqual([{k:v for k,v in p.items() if k!='event_id'} for p in cancels],
            [{'type':'response.cancel','response_id':'early'}])
        self.assertTrue(cancels[0]['event_id'])
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertEqual(self.s.conversation.phase,Phase.LISTENING)
        self.assertTrue(self.s.xai.history_ready.is_set())
        self.assertFalse(self.provider.closed)
        self.assertFalse(any(p.kind in (Kind.ERROR,Kind.CANCEL_ACK) for p in self.s.transport.sent))
        self.assertFalse(any(p.kind==Kind.OPUS and p.identity==token for p in self.s.transport.sent))
        self.deliver(Kind.PCM,self.i,0,self.s.pcm_packet_format.pack_audio_packet(0,0,320,bytes(640)))
        await settle()
        self.assertTrue(any(p['type']=='input_audio_buffer.append' for p in self.provider.calls))
        self.assertFalse(any(p['type']=='input_audio_buffer.clear' for p in self.provider.calls))
        next_token=await self.response('next');await self.ack_events()
        self.assertGreater(next_token.response,token.response)
        self.assertTrue(any(p.kind==Kind.OPUS and p.identity==next_token for p in self.s.transport.sent))

    async def test_early_vad_terminal_without_output_needs_no_fallback(self):
        await self.early_interruption()
        await self.s.xai.handle_event({'type':'response.done','response':{'id':'early','status':'cancelled','output':[]}})
        await self.finish_repair()
        self.assertFalse(any(p['type']=='response.cancel' for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertTrue(self.s.xai.history_ready.is_set())

    async def test_early_cancel_repairs_items_in_terminal_snapshot(self):
        original_send=self.provider.send
        async def terminal_output(raw):
            payload=json.loads(raw)
            if payload['type']=='response.cancel':
                self.provider.calls.append(payload)
                await self.s.xai.handle_event({'type':'response.done','response':{
                    'id':'early','status':'cancelled','output':[{'id':'snapshot-item','role':'assistant'}]}})
            else: await original_send(raw)
        self.provider.send=terminal_output
        await self.early_interruption();await self.finish_repair()
        self.assertTrue(any(p['type']=='conversation.item.delete' and p['item_id']=='snapshot-item'
                            for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertTrue(self.s.xai.history_ready.is_set())

    async def test_early_cancel_can_repair_concrete_item_without_terminal(self):
        original_send=self.provider.send
        async def cancel_item(raw):
            payload=json.loads(raw)
            if payload['type']=='response.cancel':
                self.provider.calls.append(payload)
                await self.s.xai.handle_event({'type':'response.output_item.added',
                    'response_id':'early','item':{'id':'cancel-item','role':'assistant'}})
            else: await original_send(raw)
        self.provider.send=cancel_item
        await self.early_interruption();await self.finish_repair()
        self.assertTrue(any(p['type']=='conversation.item.delete' and p['item_id']=='cancel-item'
                            for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertTrue(self.s.xai.history_ready.is_set())

    async def test_early_cancel_without_any_confirmation_remains_unsafe(self):
        original_send=self.provider.send
        async def ignore_cancel(raw):
            payload=json.loads(raw)
            if payload['type']=='response.cancel': self.provider.calls.append(payload)
            else: await original_send(raw)
        self.provider.send=ignore_cancel
        await self.early_interruption();await self.finish_repair()
        self.assertEqual(sum(p['type']=='response.cancel' for p in self.provider.calls),1)
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(self.provider.closed)
        self.assertTrue(any(p.kind==Kind.ERROR for p in self.s.transport.sent))

    async def test_vad_already_cancelled_response_preserves_conversation_and_new_audio(self):
        original_send=self.provider.send
        async def reject_cancel(raw):
            payload=json.loads(raw)
            if payload['type']=='response.cancel':
                self.provider.calls.append(payload)
                await self.s.xai.handle_event({'type':'error','error':{
                    'code':'invalid_request_error','message':'No active response to cancel',
                    'event_id':payload['event_id']}})
            else: await original_send(raw)
        self.provider.send=reject_cancel
        await self.early_interruption();await self.finish_repair()
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertFalse(self.provider.closed)
        self.assertTrue(self.s.xai.history_ready.is_set())
        self.assertFalse(self.s.xai.pending_operations)
        self.deliver(Kind.PCM,self.i,0,self.s.pcm_packet_format.pack_audio_packet(0,0,320,bytes(640)))
        await settle()
        self.assertTrue(any(p['type']=='input_audio_buffer.append' for p in self.provider.calls))
        self.assertFalse(any(p['type']=='input_audio_buffer.clear' for p in self.provider.calls))
        await self.response('after-pause');await self.ack_events()
        self.assertEqual(self.s.conversation.identity,self.i)

    async def test_vad_already_removed_unheard_item_preserves_conversation(self):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'early'}})
        await self.s.xai.handle_event({'type':'response.output_item.added','response_id':'early',
            'item':{'id':'removed-item','role':'assistant'}})
        original_send=self.provider.send
        async def reject_delete(raw):
            payload=json.loads(raw)
            if payload['type']=='conversation.item.delete':
                self.provider.calls.append(payload)
                await self.s.xai.handle_event({'type':'error','error':{
                    'code':'invalid_request_error','message':'Item removed-item does not exist',
                    'event_id':payload['event_id']}})
            else: await original_send(raw)
        self.provider.send=reject_delete
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(0);await self.finish_repair()
        self.assertEqual(self.s.conversation.identity,self.i)
        self.assertTrue(self.s.xai.history_ready.is_set());self.assertFalse(self.provider.closed)
        self.assertFalse(self.s.xai.pending_operations)

    async def test_ambiguous_cancel_rejection_still_requires_confirmed_history(self):
        original_send=self.provider.send
        async def reject_cancel(raw):
            payload=json.loads(raw)
            if payload['type']=='response.cancel':
                await self.s.xai.handle_event({'type':'error','error':{
                    'code':'invalid_request_error','message':'Invalid response_id parameter',
                    'event_id':payload['event_id']}})
            else: await original_send(raw)
        self.provider.send=reject_cancel
        await self.early_interruption();await self.finish_repair()
        self.assertIsNone(self.s.conversation.identity);self.assertTrue(self.provider.closed)

    async def test_wake_and_vad_share_one_repair(self):
        token=await self.response(count=4);await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        self.deliver(Kind.INTERRUPT,token,10,struct.pack('!I',40));await settle()
        self.assertEqual(len(self.s.interruptions.active),1)
        await self.ack_events(0)
        mutations=[p for p in self.provider.calls if p['type'].startswith('conversation.item.')]
        self.assertEqual(len(mutations),1)
        self.assertEqual(mutations[0]['audio_end_ms'],40)

    async def test_response_created_during_history_repair_closes_without_output(self):
        await self.response();await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'too-early'}})
        await settle()
        self.assertTrue(self.provider.closed)
        self.assertIsNone(self.s.conversation.identity)
        self.assertFalse(self.s.interruptions.pending)

    async def test_manual_cancel_is_sent_before_playback_ack(self):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':'streaming'}})
        token=self.s.conversation.response
        self.deliver(Kind.INTERRUPT,token,10,bytes(4));await settle()
        self.assertTrue(any(p['type']=='response.cancel' for p in self.provider.calls))
        self.assertTrue(self.s.interruptions.pending)
        await self.ack_events()
        self.assertFalse(self.s.interruptions.pending)

    async def test_late_audio_is_ignored_and_new_late_item_closes_uncertain_history(self):
        token=await self.response(count=4);await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(20)
        await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':'r1',
            'item_id':'item-r1','delta':base64.b64encode(bytes(640)).decode()})
        self.assertNotIn(token,self.s.xai.items)
        self.assertEqual(self.s.conversation.identity,self.i)
        await self.s.xai.handle_event({'type':'response.output_item.added','response_id':'r1',
            'item':{'id':'unexpected-late-item','role':'assistant'}})
        await settle()
        self.assertTrue(self.provider.closed)
        self.assertIsNone(self.s.conversation.identity)

    async def test_missing_core_event_ack_closes_provider_without_self_cancellation(self):
        await self.response()  # response start event deliberately unacknowledged
        async def closed():
            while not self.provider.closed or self.s.closing: await asyncio.sleep(.01)
        await asyncio.wait_for(closed(),2)
        self.assertFalse(self.s.closing)
        self.assertIsNone(self.s.conversation.identity)
        self.assertIsNone(self.s.provider_task)

    async def test_unconfirmed_repair_deadline_closes_provider(self):
        await self.response(count=4);await self.ack_events()
        self.s.config=replace(self.s.config,history_repair_timeout_sec=.03)
        original_send=self.provider.send
        async def no_confirmation(raw):
            payload=json.loads(raw)
            if payload['type'].startswith('conversation.item.'):
                self.provider.calls.append(payload)
                return
            await original_send(raw)
        self.provider.send=no_confirmation
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(20)
        async def closed():
            while not self.provider.closed or self.s.interruptions.pending: await asyncio.sleep(.001)
        await asyncio.wait_for(closed(),1)
        self.assertFalse(self.s.interruptions.pending)
        self.assertFalse(self.s.xai.confirmations)
        self.assertTrue(self.s.xai.history_ready.is_set())

    async def test_item_arriving_during_confirmation_is_repaired_too(self):
        await self.response(count=4);await self.ack_events()
        original_send=self.provider.send
        added=False
        async def another_item(raw):
            nonlocal added
            if json.loads(raw)['type']=='conversation.item.truncate' and not added:
                added=True
                await self.s.xai.handle_event({'type':'response.output_item.added','response_id':'r1',
                    'item':{'id':'additional-item','role':'assistant'}})
            await original_send(raw)
        self.provider.send=another_item
        # Reopen the old response mapping to model another item already in flight.
        token=self.s.conversation.response
        self.s.xai.responses['r1']=token
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events(20)
        self.assertTrue(any(p['type']=='conversation.item.delete' and p['item_id']=='additional-item'
                            for p in self.provider.calls))
        self.assertEqual(self.s.conversation.identity,self.i)

class ReliableTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_backend_does_not_open_provider(self):
        from unittest.mock import AsyncMock
        s=ContinuousSession(replace(BridgeConfig(),voice_provider="xai",save_input_wav=False,save_output_wav=False))
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
