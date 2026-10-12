"""Offline retention, repair isolation, pacing and lifecycle regressions."""
import asyncio
import time
import unittest
from dataclasses import replace
from bridge.config import BridgeConfig
from bridge.provider_input import ProviderInput
from bridge.ingress import InboundSession
from bridge.protocol import PcmPacketFormat

class ProviderInputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gate=asyncio.Event();self.sent=[];self.buffers=[];self.ingress=None
    async def asyncTearDown(self):
        if self.ingress:
            self.ingress.close()
            if self.ingress.task:
                self.ingress.task.cancel()
                try: await self.ingress.task
                except asyncio.CancelledError: pass
        for b in self.buffers:
            b.close();await b.wait_closed()
    def buffer(self, **changes):
        clock=changes.pop('clock',time.monotonic)
        async def sink(packet): self.sent.append((packet,time.monotonic()));return True
        b=ProviderInput(replace(BridgeConfig(),**changes),sink,self.gate.wait,clock)
        self.buffers.append(b);return b
    def packet(self,seq=0,frames=1):
        return dict(seq=seq,pts_samples=seq*320,frame_samples=frames*320,payload_len=640*frames,payload=b'\1\0'*320*frames)
    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate(): await asyncio.sleep(.001)
    async def test_five_seconds_retained_during_repair_without_sending(self):
        b=self.buffer(provider_input_send_interval_sec=0);b.start('test')
        for seq in range(250): await b.admit(self.packet(seq))
        await asyncio.sleep(.01)
        self.assertFalse(self.sent);self.assertEqual(b.snapshot()['queued_ms'],5000)
        await b.admit(self.packet(250));self.assertEqual(b.stats.dropped_full,1)
        self.gate.set();await self.until(lambda:b.stats.sent==250)
        self.assertEqual([p['seq'] for p,_ in self.sent],list(range(250)))
    async def test_expiry_and_silence_split_preserve_order_and_metadata(self):
        now=[0.0];b=self.buffer(clock=lambda:now[0],provider_input_send_interval_sec=0)
        await b.admit(self.packet(0));now[0]=5
        await b.admit(self.packet(1,3));self.assertEqual(b.stats.expired,1)
        self.gate.set();b.start('test');await self.until(lambda:b.stats.sent==3)
        self.assertEqual([(p['seq'],p['pts_samples'],p['frame_samples'],p['payload_len']) for p,_ in self.sent],[(n,n*320,320,640) for n in (1,2,3)])
    async def test_expiry_checked_after_repair_even_without_new_admissions(self):
        now=[0.0];b=self.buffer(clock=lambda:now[0]);b.start('test')
        await b.admit(self.packet());await asyncio.sleep(.01)
        now[0]=5;self.gate.set();await self.until(lambda:b.stats.expired==1)
        self.assertFalse(self.sent)
    async def test_catchup_has_no_burst(self):
        b=self.buffer();self.gate.set()
        await b.admit(self.packet(0,5));b.start('test')
        await self.until(lambda:b.stats.sent==5)
        self.assertGreaterEqual(self.sent[-1][1]-self.sent[0][1],.04)
    async def test_close_discards_retained_audio_and_cancels_blocked_send(self):
        b=self.buffer();await b.admit(self.packet(0,3));b.start('old')
        await asyncio.sleep(.01);b.close();await b.wait_closed()
        self.gate.set();await asyncio.sleep(.01)
        self.assertFalse(self.sent);self.assertEqual(b.stats.cleared,3)
        fresh=self.buffer();await fresh.admit(self.packet(7));fresh.start('new')
        await self.until(lambda:fresh.stats.sent==1)
        self.assertEqual([p['seq'] for p,_ in self.sent],[7])
    async def test_udp_reordering_and_nack_continue_while_provider_paused(self):
        b=self.buffer();b.start('test');requests=[]
        cfg=replace(BridgeConfig(),input_reorder_window_sec=.005)
        fmt=PcmPacketFormat(cfg)
        async def end(packet): pass
        async def commit(): return True
        def nack(base,mask): requests.append((base,mask))
        self.ingress=InboundSession(fmt,cfg,b.admit,end,commit,nack_sink=nack,silence_sink=b.admit)
        self.ingress.start()
        def admit(seq):
            p=self.packet(seq);self.ingress.admit(fmt.pack_audio_packet(seq,p['pts_samples'],320,p['payload']))
        admit(0);admit(2)
        await self.until(lambda:bool(requests))
        admit(1);await self.until(lambda:b.stats.enqueued==3)
        self.assertFalse(self.sent);self.assertEqual(self.ingress.stats.dropped_full,0)
        self.gate.set();await self.until(lambda:b.stats.sent==3)
        self.assertEqual([p['seq'] for p,_ in self.sent],[0,1,2])
    async def test_sink_failure_is_counted_and_worker_continues(self):
        b=self.buffer(provider_input_send_interval_sec=0);self.gate.set()
        async def sink(packet):
            if packet['seq']==0: raise OSError('test')
            return True
        b.sink=sink;await b.admit(self.packet(0,2));b.start('test')
        await self.until(lambda:b.stats.sent==1)
        self.assertEqual(b.stats.send_failed,1);self.assertFalse(b.task.done())
