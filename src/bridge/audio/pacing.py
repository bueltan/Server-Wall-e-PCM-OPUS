"""Monotonic downlink deadlines, independent of codecs, UDP and providers."""
import asyncio
import math

class AudioPacer:
    def __init__(self, interval, max_lag=0.04, clock=None, sleep=None):
        if not math.isfinite(interval) or interval < 0 or not math.isfinite(max_lag) or max_lag < 0:
            raise ValueError('Invalid audio pacing interval/lag')
        self.interval=interval
        self.max_lag=max_lag
        self.clock=clock
        self.sleep=sleep or asyncio.sleep
        self.deadline=self.last_sent=self.first_sent=None
        self.sends=self.rebases=0
        self.max_late=0.0

    def now(self):
        return self.clock() if self.clock else asyncio.get_running_loop().time()

    async def wait(self):
        now=self.now()
        if self.deadline is None: self.deadline=now
        # Do not repay a long outage as an unbounded burst of stale deadlines.
        late=max(0.0,now-self.deadline)
        self.max_late=max(self.max_late,late)
        if late>self.max_lag:
            self.deadline=now;self.rebases+=1
        # Small scheduler overruns may catch up, at most twice the nominal rate.
        target=self.deadline
        if self.last_sent is not None:
            target=max(target,self.last_sent+self.interval/2)
        while True:
            delay=target-self.now()
            if delay<=0: break
            await self.sleep(delay)
        # Yield also in tests configured with interval=0, or after a large overrun.
        await self.sleep(0)

    def sent(self):
        now=self.now()
        late=max(0.0,now-self.deadline)
        self.max_late=max(self.max_late,late)
        if late>self.max_lag:
            self.deadline=now;self.rebases+=1
        if self.first_sent is None: self.first_sent=now
        self.last_sent=now;self.sends+=1
        self.deadline+=self.interval

    def stats(self):
        span=(self.last_sent-self.first_sent) if self.sends>1 else 0.0
        return dict(frames=self.sends,span_ms=round(span*1000),
            fps=round((self.sends-1)/span,2) if span else 0.0,
            rebases=self.rebases,max_late_ms=round(self.max_late*1000,2))
