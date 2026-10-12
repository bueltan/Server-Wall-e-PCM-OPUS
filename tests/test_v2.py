import asyncio
import base64
import json
import struct
import unittest
from dataclasses import replace
from unittest.mock import patch
from bridge.config import BridgeConfig
from bridge.managed_session import ManagedSession
from bridge.session_protocol import Identity, Packet, Kind, encode, decode
from bridge.tagged_xai import TaggedXai
from bridge.ingress import InboundSession
from bridge.protocol import PcmPacketFormat

class Transport:
    def __init__(self): self.sent=[]
    def sendto(self,data,addr): self.sent.append((decode(data),addr))

class FakeXai:
    def __init__(self):
        self.connected_event=asyncio.Event();self.connected_event.set()
        self.ready_event=asyncio.Event();self.ready_event.set()
        self.operation_lock=asyncio.Lock();self.items={};self.output_done=set()
        self.generation_done=asyncio.Event();self.generation_done.set()
        self.commits=[];self.cancels=[];self.audio=[]
    async def send_audio_append(self,pcm): self.audio.append(pcm);return True
    async def commit_turn(self,token): self.commits.append(token);return True
    async def cancel_turn(self,token,played): self.cancels.append((token,played));return True

async def settle():
    for _ in range(12): await asyncio.sleep(0)

class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config=replace(BridgeConfig(),save_input_wav=False,save_output_wav=False,
            input_recovery_window_sec=.12,input_reorder_window_sec=.04,
            output_send_interval_sec=0,output_end_repeat_interval=0)
        self.session=ManagedSession(self.config);self.s=self.session
        self.s.transport=Transport();self.peer=('127.0.0.1',1234)
        self.s.xai=FakeXai();self.s.control_task=asyncio.create_task(self.s._control_worker())
        self.s.output_task=asyncio.create_task(self.s._udp_output_sender())
        self.deliver(Kind.HELLO,Identity(11,0,0,0),1)
        self.id=Identity(11,self.s.server_nonce,1,1)
    async def asyncTearDown(self):
        self.s.connection_lost(None)
        for task in (self.s.control_task,self.s.input.task,self.s.output_task):
            if task:
                try: await task
                except asyncio.CancelledError: pass
    def deliver(self,kind,identity=None,seq=0,body=b'',peer=None):
        self.s.datagram_received(encode(Packet(kind,0,identity or self.id,seq,body)),peer or self.peer)
    async def start(self):
        self.deliver(Kind.START,seq=2);await settle();self.assertEqual(self.s.active,self.id)
    async def test_start_commit_idempotent_and_next_normal_turn(self):
        await self.start();self.deliver(Kind.START,seq=2);await settle()
        self.deliver(Kind.COMMIT,seq=3,body=struct.pack('!II',0,0));await asyncio.sleep(.16)
        self.deliver(Kind.COMMIT,seq=3,body=struct.pack('!II',0,0));await settle()
        self.assertEqual(self.s.xai.commits,[self.id])
        self.assertEqual(sum(p.kind==Kind.COMMIT_ACK for p,_ in self.s.transport.sent),2)
        self.s.output_pcm_queue.put_nowait((self.id,None));await settle()
        self.assertTrue(self.s.output_finished)
        self.id=replace(self.id,turn=2);await self.start()
    async def test_cancel_preempts_queued_audio_and_filters_old_id(self):
        await self.start();old=self.id
        self.s.output_pcm_queue.put_nowait((old,b'\0'*640))
        self.deliver(Kind.CANCEL,seq=3,body=struct.pack('!I',125))
        self.assertIsNone(self.s.active)
        self.deliver(Kind.START,replace(old,turn=2),4);await settle()
        self.assertEqual(self.s.xai.cancels,[(old,125)])
        self.assertEqual(self.s.active.turn,2)
        self.assertFalse(any(p.kind==Kind.OPUS for p,_ in self.s.transport.sent))
        self.deliver(Kind.PCM,old,0,self.s.pcm_packet_format.pack_audio_packet(0,0,320,b'\0'*640));await settle()
        self.assertEqual(self.s.xai.audio,[])
        self.deliver(Kind.CANCEL,old,3,struct.pack('!I',125));await settle()
        self.assertEqual(len(self.s.xai.cancels),1)
    async def test_old_end_sender_cannot_finish_new_turn(self):
        await self.start();old=self.id
        self.s.config=replace(self.s.config,output_end_repeat_interval=.02)
        self.s.output_pcm_queue.put_nowait((old,None));await settle()
        self.deliver(Kind.CANCEL,old,3,struct.pack('!I',0))
        self.id=replace(old,turn=2);self.deliver(Kind.START,self.id,4);await settle()
        self.assertEqual(self.s.active,self.id)
        await asyncio.sleep(.03)
        self.assertFalse(self.s.output_finished)
        self.assertTrue(all(p.identity==old for p,_ in self.s.transport.sent if p.kind==Kind.OPUS))

    async def test_foreign_peer_legacy_and_reboot_isolation(self):
        await self.start()
        self.s.datagram_received(b'COMMIT',self.peer)
        self.deliver(Kind.PING,Identity(12,self.s.server_nonce,0,0),peer=('127.0.0.2',1234))
        self.assertEqual(self.s.remote_addr,self.peer)
        self.deliver(Kind.HELLO,Identity(12,0,0,0),10);await settle()
        self.assertIsNone(self.s.active)
        self.deliver(Kind.HELLO,Identity(12,0,0,0),11)
        self.assertEqual(self.s.client_nonce,12)
        self.assertNotEqual(self.s.server_nonce,self.id.server)
        self.deliver(Kind.HELLO,Identity(11,0,0,0),12)
        self.assertEqual(self.s.client_nonce,12)
    async def test_nack_recovers_before_commit_and_does_not_reopen(self):
        await self.start()
        fmt=self.s.pcm_packet_format
        self.deliver(Kind.PCM,seq=1,body=fmt.pack_audio_packet(1,320,320,b'\1'*640));await settle()
        requests=[p for p,_ in self.s.transport.sent if p.kind==Kind.NACK]
        self.assertEqual(struct.unpack('!IH',requests[-1].body),(0,1))
        self.deliver(Kind.PCM,seq=0,body=fmt.pack_audio_packet(0,0,320,b'\0'*640));await settle()
        self.deliver(Kind.COMMIT,seq=9,body=struct.pack('!II',2,640));await asyncio.sleep(.16)
        self.assertEqual(self.s.xai.audio,[b'\0'*640,b'\1'*640])
        self.assertEqual(self.s.input.stats.missing,0)
        self.deliver(Kind.PCM,seq=0,body=fmt.pack_audio_packet(0,0,320,b'\0'*640));await settle()
        self.assertEqual(len(self.s.xai.audio),2)
    async def test_unrecoverable_tail_inserts_silence_after_recovery_window(self):
        await self.start();self.deliver(Kind.COMMIT,seq=9,body=struct.pack('!II',2,640))
        await asyncio.sleep(.16)
        self.assertEqual(self.s.input.stats.missing,2)
        self.assertEqual(self.s.xai.audio,[bytes(1280)])
        self.assertEqual(self.s.input.stats.silence_frames,2)
        self.assertEqual(self.s.xai.commits,[self.id])

    async def test_expired_middle_gap_inserts_silence_before_next_frame(self):
        await self.start()
        fmt=self.s.pcm_packet_format
        self.deliver(Kind.PCM,seq=0,body=fmt.pack_audio_packet(0,0,320,b'\1'*640))
        self.deliver(Kind.PCM,seq=2,body=fmt.pack_audio_packet(2,640,320,b'\2'*640))
        await asyncio.sleep(.16)
        self.assertEqual(self.s.xai.audio,[b'\1'*640,bytes(640),b'\2'*640])
        self.assertEqual(self.s.input.stats.silence_frames,1)

    async def test_large_gap_silence_capped_at_200ms(self):
        await self.start()
        self.deliver(Kind.COMMIT,seq=9,body=struct.pack('!II',50,16000))
        await asyncio.sleep(.16)
        self.assertEqual(self.s.input.stats.missing,50)
        self.assertEqual(self.s.input.stats.silence_frames,10)
        self.assertEqual(self.s.xai.audio,[bytes(6400)])

class WebSocket:
    def __init__(self): self.events=asyncio.Queue();self.sent=[]
    def __aiter__(self): return self
    async def __anext__(self): return json.dumps(await self.events.get())
    async def close(self): pass
    async def send(self,raw):
        event=json.loads(raw);self.sent.append(event);kind=event['type']
        replies={'conversation.item.truncate':'conversation.item.truncated',
                 'conversation.item.delete':'conversation.item.deleted','input_audio_buffer.clear':'input_audio_buffer.cleared'}
        if kind in replies: self.events.put_nowait({'type':replies[kind],'item_id':event.get('item_id','')})
        if kind=='response.cancel': self.events.put_nowait({'type':'response.done','response':{'id':'r1'}})

class XaiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token=Identity(1,2,3,1);self.active=self.token
        self.x=TaggedXai(BridgeConfig(),asyncio.Queue(),lambda t:t==self.active)
        self.x.ws=WebSocket();self.x.connected_event.set();self.x.ready_event.set()
        self.task=asyncio.create_task(self.x.receive_events_forever())
    async def asyncTearDown(self):
        self.task.cancel()
        try: await self.task
        except asyncio.CancelledError: pass
    async def emit(self,event): self.x.ws.events.put_nowait(event);await settle()
    async def generation(self):
        self.assertTrue(await self.x.commit_turn(self.token))
        await self.emit({'type':'response.created','response':{'id':'r1'}})
        for item in ['a','b']:
            await self.emit({'type':'response.output_audio.delta','response_id':'r1','item_id':item,'content_index':0,
                'delta':base64.b64encode(b'\0'*3200).decode()})
    async def test_cancel_confirms_truncate_delete_clear(self):
        await self.generation();self.active=None
        self.assertTrue(await self.x.cancel_turn(self.token,50))
        operations=[{k:v for k,v in e.items() if k!='event_id'} for e in self.x.ws.sent if e['type'].startswith('conversation.')]
        self.assertEqual(operations,[{'type':'conversation.item.truncate','item_id':'a','content_index':0,'audio_end_ms':50},
            {'type':'conversation.item.delete','item_id':'b'}])
        self.assertEqual(self.x.ws.sent[-1]['type'],'input_audio_buffer.clear')
    async def test_cancel_before_first_audio_removes_unheard_item(self):
        await self.x.commit_turn(self.token)
        await self.emit({'type':'response.created','response':{'id':'r1'}})
        await self.emit({'type':'response.output_item.added','response_id':'r1',
            'item':{'id':'unheard','role':'assistant'}})
        self.active=None
        self.assertTrue(await self.x.cancel_turn(self.token,0))
        self.assertTrue(any(e['type']=='conversation.item.delete' and e.get('item_id')=='unheard' for e in self.x.ws.sent))

    async def test_scoped_error_keeps_receiver_alive_and_reports_full_operation(self):
        from bridge.operation_errors import ProviderOperationError
        original_send=self.x.ws.send
        async def reject(raw):
            payload=json.loads(raw)
            if payload['type']=='conversation.item.delete':
                self.x.ws.events.put_nowait({'type':'error','error':{'code':'invalid_request_error',
                    'message':'Deletion refused for this item', 'event_id':payload['event_id']}})
            else: await original_send(raw)
        self.x.ws.send=reject
        with self.assertRaises(ProviderOperationError) as failure:
            await self.x._confirmed({'type':'conversation.item.delete','item_id':'x'},'conversation.item.deleted','x')
        self.assertIn('Deletion refused',str(failure.exception))
        self.assertFalse(self.task.done())
        await self.x._confirmed({'type':'input_audio_buffer.clear'},'input_audio_buffer.cleared')
        self.assertEqual(self.x.pending_operations,{})

    async def test_finished_response_can_still_be_truncated(self):
        await self.generation();await self.emit({'type':'response.done','response':{'id':'r1'}})
        self.assertTrue(await self.x.cancel_turn(self.token,150))
        self.assertFalse(any(e['type']=='response.cancel' for e in self.x.ws.sent))
        trunc=[e for e in self.x.ws.sent if e['type']=='conversation.item.truncate']
        self.assertEqual(trunc[0]['item_id'],'b');self.assertEqual(trunc[0]['audio_end_ms'],50)
    async def test_old_events_do_not_finish_new_generation(self):
        self.x.pending=self.token;self.x.generation_done.clear()
        await self.emit({'type':'response.done','response':{'id':'old'}})
        self.assertFalse(self.x.generation_done.is_set());self.assertEqual(self.x.pending,self.token)
        await self.emit({'type':'response.output_audio.delta','response_id':'old','delta':'AA=='})
        self.assertTrue(self.x.output_pcm_queue.empty())
    async def test_missing_confirmation_closes_and_rejects_next_turn(self):
        async def ignore(raw): self.x.ws.sent.append(json.loads(raw))
        self.x.ws.send=ignore
        with patch('bridge.tagged_xai.asyncio.wait_for',side_effect=TimeoutError):
            self.assertFalse(await self.x.cancel_turn(self.token,0))
        self.assertFalse(self.x.connected_event.is_set());self.assertFalse(await self.x.commit_turn(self.token))

if __name__=='__main__': unittest.main()
