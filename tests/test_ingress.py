import asyncio
import sys
import unittest
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge.config import BridgeConfig
from bridge.transport.protocols.audio import PcmPacketFormat
from bridge.transport.ingress import InboundSession, ReorderBuffer, IngressStats

class ReorderTests(unittest.TestCase):
    def setUp(self):
        self.stats=IngressStats();self.buffer=ReorderBuffer(.04,16,self.stats)
    def test_reordering_no_initial_delay_for_contiguous_audio(self):
        b=self.buffer;b.push({'seq':0},0)
        self.assertEqual([p['seq'] for p in b.drain(0)],[0])
        b.push({'seq':2},.01);self.assertEqual(b.drain(.049),[])
        b.push({'seq':1},.049)
        self.assertEqual([p['seq'] for p in b.drain(.049)],[1,2])
        self.assertEqual(self.stats.missing,0)
    def test_missing_no_synthetic_frames_duplicates_and_late(self):
        b=self.buffer;b.push({'seq':2},0);b.push({'seq':2},.01)
        self.assertEqual(b.drain(.039),[])
        self.assertEqual([p['seq'] for p in b.drain(.04)],[2])
        b.push({'seq':1},.05);self.assertEqual(b.drain(.09),[])
        self.assertEqual((self.stats.missing,self.stats.duplicates_late),(2,2))
    def test_rollover_and_bounded_pending(self):
        b=ReorderBuffer(.04,2,self.stats,expected=0xffffffff)
        b.push({'seq':0},0);b.push({'seq':0xffffffff},.01);b.push({'seq':1},.02)
        self.assertEqual([p['seq'] for p in b.drain(.02)],[0xffffffff,0])
        self.assertEqual(self.stats.dropped_full,1)

class InboundTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config=replace(BridgeConfig(),input_queue_frames=3,input_send_timeout_sec=.03)
        self.fmt=PcmPacketFormat(self.config);self.events=[]
        async def audio(packet): self.events.append(('audio',packet['seq']));return True
        async def end(packet): self.events.append(('end',packet['seq']))
        async def commit(): self.events.append(('commit',));return True
        self.session=InboundSession(self.fmt,self.config,audio,end,commit)
    async def asyncTearDown(self):
        self.session.close()
        if self.session.task:
            try: await self.session.task
            except asyncio.CancelledError: pass
    def audio(self,seq): return self.fmt.pack_audio_packet(seq,(seq*320)&0xffffffff,320,b'\1\0'*320)
    async def until(self,predicate):
        async with asyncio.timeout(1):
            while not predicate(): await asyncio.sleep(.001)
    async def test_reserved_controls_under_saturation_and_duplicate_commit(self):
        s=self.session
        for seq in range(3): self.assertTrue(s.admit(self.audio(seq)))
        self.assertFalse(s.admit(self.audio(3)))
        self.assertTrue(s.admit(self.fmt.pack_end_packet(4,1280)))
        self.assertTrue(s.admit_commit());self.assertFalse(s.admit_commit())
        self.assertEqual(s.queue.qsize(),5)
        s.start();await self.until(lambda:('commit',) in self.events)
        self.assertEqual(self.events,[('audio',0),('audio',1),('audio',2),('end',4),('commit',)])
        self.assertEqual(s.stats.dropped_full,1);self.assertEqual(s.stats.missing,1)
    async def test_trailing_reordered_audio_before_commit_and_end_idempotent(self):
        s=self.session;s.start();s.admit(self.audio(0))
        s.admit(self.fmt.pack_end_packet(3,960));s.admit_commit()
        await asyncio.sleep(.01)
        s.admit(self.audio(2));s.admit(self.audio(1))
        self.assertFalse(s.admit(self.fmt.pack_end_packet(3,960)))
        await self.until(lambda:('commit',) in self.events)
        self.assertEqual(self.events,[('audio',0),('audio',1),('audio',2),('end',3),('commit',)])
        self.assertEqual(s.stats.missing,0)
    async def test_xai_timeout_drops_frame_then_processes_controls(self):
        s=self.session
        async def blocked(packet): await asyncio.sleep(1)
        s.audio_sink=blocked;s.admit(self.audio(0))
        s.admit(self.fmt.pack_end_packet(1,320));s.admit_commit();s.start()
        await self.until(lambda:('commit',) in self.events)
        self.assertEqual(s.stats.send_failed,1);self.assertEqual(self.events,[('end',1),('commit',)])
    async def test_disconnected_xai_does_not_stop_worker(self):
        s=self.session
        async def unavailable(packet): return False
        s.audio_sink=unavailable;s.admit(self.audio(0));s.admit_commit();s.start()
        await self.until(lambda:('commit',) in self.events)
        self.assertEqual(s.stats.send_failed,1);self.assertFalse(s.task.done())
    async def test_new_turn_with_lost_initial_frame_and_commit_duplicates(self):
        s=self.session;s.admit(self.audio(0));s.admit_commit();s.start()
        await self.until(lambda:('commit',) in self.events)
        self.assertFalse(s.admit_commit())
        self.assertTrue(s.admit(self.audio(1)));self.assertTrue(s.admit_commit())
        await self.until(lambda:self.events.count(('commit',))==2)
        self.assertEqual(self.events,[('audio',0),('commit',),('audio',1),('commit',)])
    async def test_gap_expiry_while_listening_and_no_silence_padding(self):
        s=self.session;s.start();s.admit(self.audio(0));s.admit(self.audio(2))
        await self.until(lambda:('audio',2) in self.events)
        self.assertEqual(self.events,[('audio',0),('audio',2)])
        self.assertEqual(s.stats.missing,1)
        self.assertTrue(s.admit(self.audio(1)))
        await self.until(lambda:s.stats.duplicates_late==1)
        self.assertEqual(len(self.events),2)

    async def test_end_trailing_loss_count_and_control_failures(self):
        s=self.session
        async def broken_end(packet): raise OSError('recording failure')
        async def broken_commit(): return False
        s.end_sink=broken_end;s.commit_sink=broken_commit
        s.admit(self.audio(0));s.admit(self.fmt.pack_end_packet(4,1280));s.admit_commit();s.start()
        await self.until(lambda:s.finished and not s.sealing)
        self.assertEqual(s.stats.missing,3);self.assertEqual(s.stats.send_failed,2)
        self.assertFalse(s.task.done())

    async def test_coordination_pause_does_not_consume_send_timeout(self):
        s=self.session;gate=asyncio.Event();entered=asyncio.Event()
        async def before_audio():
            entered.set();await gate.wait()
        s.before_audio=before_audio
        s.admit(self.audio(0));s.start();await entered.wait()
        # Longer than the actual send timeout, but no transport send is attempted.
        await asyncio.sleep(self.config.input_send_timeout_sec*2)
        self.assertEqual(s.stats.send_failed,0)
        self.assertEqual(self.events,[])
        for seq in range(1,6): s.admit(self.audio(seq))
        self.assertGreater(s.stats.dropped_full,0)
        gate.set();await self.until(lambda:('audio',0) in self.events)
        self.assertEqual(s.stats.send_failed,0)

    async def test_shutdown_cancels_blocked_send(self):
        s=self.session;entered=asyncio.Event()
        async def blocked(packet): entered.set();await asyncio.Future()
        s.audio_sink=blocked;s.admit(self.audio(0));s.start();await entered.wait()
        s.close()
        with self.assertRaises(asyncio.CancelledError): await s.task

    async def test_malformed_controls_and_pcm_rejected_before_queue(self):
        s=self.session
        data=self.fmt.pack_audio_packet(0,0,320,b'\0'*640)
        for invalid in [data[:5]+b'\x80'+data[6:],self.fmt.pack_audio_packet(0,0,10,b'\0'*20),
                        self.fmt.pack_end_packet(1,320)+b'bad']:
            self.assertFalse(s.admit(invalid))
        self.assertEqual(s.queue.qsize(),0);self.assertEqual(s.stats.invalid,3)
    async def test_audio_during_blocked_commit_never_crosses_barrier(self):
        s=self.session;entered=asyncio.Event();release=asyncio.Event()
        async def commit(): entered.set();await release.wait();self.events.append(('commit',));return True
        s.commit_sink=commit;s.admit(self.audio(0));s.admit_commit();s.start()
        await entered.wait()
        self.assertFalse(s.admit(self.audio(1)));self.assertFalse(s.admit(self.audio(0)))
        release.set();await self.until(lambda:('commit',) in self.events)
        self.assertEqual(s.queue.qsize(),0);self.assertEqual(self.events,[('audio',0),('commit',)])

if __name__=='__main__': unittest.main()

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_lookahead_still_accepts_missing_head(self):
        stats=IngressStats(); b=ReorderBuffer(.04,2,stats)
        b.push({'seq':1},0); b.push({'seq':2},0)
        self.assertTrue(b.push({'seq':0},.03))
        self.assertEqual([p['seq'] for p in b.drain(.03)],[0,1,2])
        self.assertEqual(stats.dropped_full,0)

    async def test_busy_queue_does_not_delay_real_nack_or_request_queued_audio(self):
        config=replace(BridgeConfig(),input_queue_frames=4)
        fmt=PcmPacketFormat(config); requests=[]; audio=[]
        async def send(p): audio.append(p['seq'])
        async def end(p): pass
        async def commit(): return True
        session=InboundSession(fmt,config,send,end,commit,
            nack_sink=lambda base,mask: requests.append((base,mask)))
        # Frame zero is genuinely absent; frame one is queued and must not be requested.
        for seq in (2,1,3,4):
            session.admit(fmt.pack_audio_packet(seq,seq*320,320,bytes(640)))
        kind,packet=session._take(session.queue.get_nowait())
        await session._consume(kind,packet)
        self.assertEqual(requests,[(0,1)])
        session.admit(fmt.pack_audio_packet(0,0,320,bytes(640)))
        while not session.queue.empty():
            kind,packet=session._take(session.queue.get_nowait())
            await session._consume(kind,packet)
        self.assertEqual(audio,list(range(5)))
        self.assertEqual(session.stats.nack_recovered,1)
        self.assertEqual(requests,[(0,1)])
