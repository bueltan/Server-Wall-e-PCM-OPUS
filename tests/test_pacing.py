"""Offline timing and impaired-network scenarios. No live provider or sockets."""
import asyncio
import unittest
import random
from dataclasses import replace
from bridge.audio.pacing import AudioPacer
from bridge.audio.pipeline import AudioPipeline
from bridge.config import BridgeConfig
from bridge.transport.protocols.conversation import Identity,Kind,Packet,Event
from bridge.transport.ingress import ReorderBuffer,IngressStats
from bridge.transport.reliable_events import ReliableEvents

class Clock:
    def __init__(self): self.t=0.0;self.oversleep=0.0
    def now(self): return self.t
    async def sleep(self,delay):
        self.t+=delay
        if delay>0: self.t+=self.oversleep;self.oversleep=0
        await asyncio.sleep(0)

class PacingTests(unittest.IsolatedAsyncioTestCase):
    async def test_processing_time_does_not_accumulate(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep);times=[]
        for _ in range(1500):
            clock.t+=.002  # encode + send preparation, formerly added to every wait
            await p.wait();times.append(clock.t);p.sent()
        self.assertAlmostEqual(times[-1]-times[0],1499*.02,places=7)
        self.assertEqual(p.stats()['fps'],50.0);self.assertEqual(p.rebases,0)

    async def test_overrun_rebases_and_limits_catchup(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        await p.wait();p.sent();clock.t+=.8
        times=[]
        for _ in range(20):
            await p.wait();times.append(clock.t);p.sent()
        self.assertEqual(p.rebases,1)
        self.assertTrue(all(b-a>=.01-1e-9 for a,b in zip(times,times[1:])))
        self.assertAlmostEqual(times[-1]-times[0],19*.02)

    async def test_scheduler_oversleep_does_not_create_burst(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        await p.wait();p.sent();clock.oversleep=.3
        await p.wait();p.sent();previous=clock.t
        await p.wait();p.sent()
        self.assertEqual(p.rebases,1);self.assertAlmostEqual(clock.t-previous,.02)

    async def test_small_lateness_catches_up_with_minimum_spacing(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        await p.wait();p.sent();clock.t=.05
        times=[]
        for _ in range(8):
            await p.wait();times.append(clock.t);p.sent()
        self.assertTrue(all(b-a>=.01-1e-9 for a,b in zip(times,times[1:])))
        self.assertEqual(p.rebases,0)

    async def test_reset_and_provider_gap_do_not_repay_idle_time(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        await p.wait();p.sent();clock.t=5
        await p.wait();p.sent();self.assertEqual(p.rebases,1)
        fresh=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        await fresh.wait();fresh.sent();self.assertEqual(fresh.sends,1);self.assertEqual(fresh.rebases,0)

    async def test_simulated_jitter_reorder_is_not_clock_drift(self):
        clock=Clock();p=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        rng=random.Random(42);sent=[];arrivals=[]
        # Bounded asymmetric jitter and deterministic isolated/burst loss.
        lost={60,61,62,180,301}
        for seq in range(500):
            clock.t+=.002;await p.wait();p.sent();sent.append(clock.t)
            if seq not in lost: arrivals.append((clock.t+rng.uniform(.02,.14),seq))
        arrivals.sort()
        self.assertTrue(any(a[1]>b[1] for a,b in zip(arrivals,arrivals[1:])))
        self.assertEqual(len(arrivals),495)
        self.assertAlmostEqual(sent[-1]-sent[0],499*.02,places=7)
        # Exercise production reorder limits. Late packets may legitimately be discarded.
        cfg=BridgeConfig()
        stats=IngressStats();reorder=ReorderBuffer(cfg.input_reorder_window_sec,cfg.input_queue_frames,stats);delivered=[]
        for now,seq in arrivals:
            delivered.extend(x['seq'] for x in reorder.drain(now))
            reorder.push({'seq':seq},now)
            delivered.extend(x['seq'] for x in reorder.drain(now))
        delivered.extend(x['seq'] for x in reorder.drain(arrivals[-1][0]+1,True))
        self.assertEqual(delivered,sorted(set(delivered)))
        self.assertFalse(set(delivered)&lost)
        self.assertEqual(len(delivered)+stats.missing,500)
        self.assertGreater(stats.duplicates_late,0)
        self.assertEqual(stats.missing,len(lost)+stats.duplicates_late+stats.dropped_full)

    async def test_real_opus_pipeline_uses_deadlines_and_continuous_pts(self):
        cfg=replace(BridgeConfig(),save_input_wav=False,save_output_wav=False)
        pipeline=AudioPipeline(cfg);token=Identity(1,2,3,1);clock=Clock();wire=[]
        pipeline.output_valid=lambda t:t==token
        pipeline._send=lambda kind,t,seq,body:wire.append((clock.t,seq,pipeline.opus_packet_format.unpack_packet(body)))
        pipeline.counters={'opus_frames':0}
        pipeline.output_pacer=AudioPacer(.02,clock=clock.now,sleep=clock.sleep)
        encode=pipeline.opus_encoder.encode_pcm16_frame
        def processing(frame):clock.t+=.002;return encode(frame)
        pipeline.opus_encoder.encode_pcm16_frame=processing
        try:
            pipeline.output_pending_pcm.extend(bytes(cfg.opus_frame_bytes*500))
            await pipeline._flush_tagged(token,False)
            self.assertEqual(len(wire),500)
            self.assertAlmostEqual(wire[-1][0]-wire[0][0],499*.02,places=7)
            self.assertEqual([p[1] for p in wire],list(range(500)))
            self.assertEqual([p[2]['pts_samples'] for p in wire],[seq*320 for seq in range(500)])
            self.assertEqual(pipeline.output_sequence,500)
        finally:pipeline.opus_encoder.close()

    async def test_cancel_during_pacing_never_sends_old_audio(self):
        cfg=replace(BridgeConfig(),save_input_wav=False,save_output_wav=False)
        pipeline=AudioPipeline(cfg);token=Identity(1,2,3,1);current=[token];wire=[]
        pipeline.output_valid=lambda t:t==current[0]
        pipeline._send=lambda *args:wire.append(args)
        pipeline.counters={'opus_frames':0}
        entered=asyncio.Event();resume=asyncio.Event()
        async def blocked(): entered.set();await resume.wait()
        pipeline.output_pacer.wait=blocked
        pipeline.output_pending_pcm.extend(bytes(cfg.opus_frame_bytes))
        task=asyncio.create_task(pipeline._flush_tagged(token,False))
        await entered.wait();current[0]=None;pipeline.reset_output();resume.set();await task
        self.assertEqual(wire,[]);self.assertEqual(pipeline.output_sequence,0)
        pipeline.opus_encoder.close()

class NetworkControls(unittest.IsolatedAsyncioTestCase):
    async def test_event_survives_loss_and_delayed_ack(self):
        attempts=[];timers=[];failures=[];ident=Identity(1,2,3,0)
        async def fail(cause):failures.append(cause)
        def send(packet):
            attempts.append(packet)
            # Drop first transmission; subsequent ACK takes 220 ms and races retry.
            if len(attempts)==2:
                ack=Packet(Kind.EVENT_ACK,0,ident,packet.sequence,(123).to_bytes(4,'big'))
                timers.append(asyncio.get_running_loop().call_later(.22,events.ack,ack))
        events=ReliableEvents(send,fail);events.start()
        try:
            future=events.emit(ident,Event.SPEECH_STARTED,1)
            self.assertEqual(await asyncio.wait_for(future,1),123)
            self.assertGreaterEqual(len(attempts),3);self.assertFalse(failures)
            self.assertEqual(len({p.sequence for p in attempts}),1)
        finally:
            events.close()
            for timer in timers:timer.cancel()
            await asyncio.sleep(0)

class RecoveryBudget(unittest.TestCase):
    def test_cache_recovery_limits_under_latency(self):
        # Lost frame followed 20 ms later; gap detection + NACK outward/return.
        # Maximum retention is 240 ms (12 frames); arrival of retransmission
        # must additionally fit the backend's 120-ms recovery window.
        cases=[(.02,.02,True,True),(.08,.08,True,False),(.14,.145,False,False)]
        for forward,backward,cached,on_time in cases:
            age_when_nack=.02+forward+.04+backward
            self.assertEqual(age_when_nack<.24,cached)
            self.assertEqual(forward+backward<=.12,on_time)
