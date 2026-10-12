"""Ordered PCM retention and provider sending, independent of UDP recovery."""
import asyncio
from collections import deque
from dataclasses import dataclass
import math
import time
from bridge.diagnostics.logging import event

@dataclass
class InputBufferStats:
    enqueued: int = 0
    sent: int = 0
    dropped_full: int = 0
    expired: int = 0
    cleared: int = 0
    send_failed: int = 0
    peak_frames: int = 0
    max_send_age_ms: int = 0

class ProviderInput:
    FRAME_BYTES=640
    FRAME_SEC=.02

    def __init__(self, config, sink, ready, clock=time.monotonic):
        for name in ('provider_input_buffer_sec','provider_input_max_age_sec'):
            value=getattr(config,name)
            if not math.isfinite(value) or not .02<=value<=30:
                raise ValueError(f'{name} must be in .02..30 seconds')
        if not math.isfinite(config.provider_input_send_interval_sec) or not 0<=config.provider_input_send_interval_sec<=.02:
            raise ValueError('provider_input_send_interval_sec must be in 0..0.02')
        self.config,self.sink,self.ready,self.clock=config,sink,ready,clock
        self.capacity=max(1,int(config.provider_input_buffer_sec/self.FRAME_SEC))
        self.pending=deque()
        self.wake=asyncio.Event()
        self.stats=InputBufferStats()
        self.task=None;self.identity=None;self.closed=False

    def start(self, identity):
        if self.task is None and not self.closed:
            self.identity=identity
            self.task=asyncio.create_task(self.run(),name='provider-input')

    def _expire(self, now):
        while self.pending and now-self.pending[0][0]>=self.config.provider_input_max_age_sec:
            self.pending.popleft();self.stats.expired+=1

    async def admit(self, packet):
        # No provider wait here: ingress ordering/NACK must continue during repair.
        if self.closed: return True
        payload=packet['payload']
        if not payload or len(payload)%self.FRAME_BYTES:
            raise ValueError('Provider input requires complete 20 ms PCM frames')
        now=self.clock();self._expire(now)
        for index in range(len(payload)//self.FRAME_BYTES):
            if len(self.pending)>=self.capacity:
                self.stats.dropped_full+=1;continue  # preserve existing ordered prefix
            frame=dict(packet,seq=(packet['seq']+index)&0xffffffff,
                       pts_samples=(packet.get('pts_samples',0)+index*320)&0xffffffff,
                       frame_samples=320,payload_len=640,payload=payload[index*640:(index+1)*640])
            self.pending.append((now,frame));self.stats.enqueued+=1
        self.stats.peak_frames=max(self.stats.peak_frames,len(self.pending))
        if self.pending: self.wake.set()
        # Intentional policy drops belong to buffer stats, not UDP send_failed.
        return True

    async def run(self):
        next_send=0.0
        while True:
            await self.wake.wait()
            await self.ready()  # history barrier is owned by the provider adapter
            delay=next_send-self.clock()
            if delay>0: await asyncio.sleep(delay)
            # A repair may have started during pacing; never cross its barrier.
            await self.ready()
            now=self.clock();self._expire(now)
            if not self.pending:
                self.wake.clear();continue
            admitted,packet=self.pending.popleft()
            if not self.pending: self.wake.clear()
            age=round((now-admitted)*1000)
            self.stats.max_send_age_ms=max(self.stats.max_send_age_ms,age)
            try:
                ok=await asyncio.wait_for(self.sink(packet),self.config.input_send_timeout_sec)
                if ok is False: self.stats.send_failed+=1
                else: self.stats.sent+=1
            except Exception as exc:
                self.stats.send_failed+=1
                event('PROVIDER_INPUT_SEND_FAILED',self.identity,error=type(exc).__name__)
            # No unlimited catch-up bursts after blocked sends/repair.
            next_send=self.clock()+self.config.provider_input_send_interval_sec

    def snapshot(self):
        now=self.clock()
        return dict(vars(self.stats),queued_frames=len(self.pending),
                    queued_ms=len(self.pending)*20,capacity_frames=self.capacity,
                    oldest_ms=round((now-self.pending[0][0])*1000) if self.pending else 0)

    def close(self):
        if self.closed: return
        self.closed=True
        self.stats.cleared+=len(self.pending);self.pending.clear();self.wake.clear()
        if self.task: self.task.cancel()
        event('PROVIDER_INPUT_CLOSED',self.identity,**self.snapshot())

    async def wait_closed(self):
        if self.task:
            try: await self.task
            except asyncio.CancelledError: pass
