"""Bounded Opus resend cache and acknowledged END; no provider responsibilities."""
from collections import OrderedDict, deque
import math
import struct
import time
from bridge.transport.protocols.conversation import Kind
from bridge.diagnostics.logging import event

class DownlinkRecovery:
    def __init__(self, config, send):
        self.config, self.send = config, send
        for name in ('output_cache_ttl_sec','output_end_retry_sec','output_end_timeout_sec'):
            if not math.isfinite(getattr(config,name)) or getattr(config,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not 1 <= config.output_cache_frames <= 512:
            raise ValueError('output_cache_frames must be in 1..512')
        self.reset()

    def reset(self):
        self.token = None
        self.frames = OrderedDict()
        self.pending = deque(maxlen=32)
        self.end = None
        self.stats = dict(nack_frames=0, retransmitted=0, cache_miss=0, repeated=0,
                          expired=0, pending_full=0, send_failed=0, end_sent=0, end_acked=0)

    def remember(self, token, sequence, body, now=None):
        if token != self.token:
            self.reset(); self.token = token
        now = time.monotonic() if now is None else now
        self.frames[sequence] = [now, body, False]
        while len(self.frames)>self.config.output_cache_frames:
            self.frames.popitem(last=False)

    def request(self, packet, now=None):
        if packet.identity != self.token or packet.flags or len(packet.body)!=6:
            return False
        base, mask = struct.unpack('!IH', packet.body)
        if base != packet.sequence or not mask or mask & 0xff00:
            return False
        now = time.monotonic() if now is None else now
        for bit in range(8):
            if not mask & (1<<bit): continue
            sequence = (base+bit)&0xffffffff
            self.stats['nack_frames'] += 1
            frame = self.frames.get(sequence)
            if frame is None:
                self.stats['cache_miss'] += 1
            elif now-frame[0]>self.config.output_cache_ttl_sec:
                self.stats['expired'] += 1
            elif frame[2]:
                self.stats['repeated'] += 1
            elif len(self.pending)==self.pending.maxlen:
                self.stats['pending_full'] += 1
            else:
                frame[2] = True  # one attempt, including local send failure
                self.pending.append(sequence)
        return True

    def finish(self, token, sequence, body, now=None):
        if token != self.token:
            self.reset(); self.token=token  # empty response still needs END
        now = time.monotonic() if now is None else now
        self.end = [sequence, body, now, now]
        # END admission does not spend another retransmission budget mid-cycle.
        try:
            self.send(Kind.AUDIO_END,token,sequence,body)
            self.stats['end_sent'] += 1
        except OSError:
            self.stats['send_failed'] += 1

    def acknowledge(self, packet):
        if (packet.identity!=self.token or packet.flags or packet.body or
                self.end is None or packet.sequence!=self.end[0]):
            return False
        self.stats['end_acked'] += 1
        event('OUTPUT_END_ACK', self.token, final_seq=packet.sequence)
        self.end = None
        return True

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        for _ in range(min(2,len(self.pending))):
            sequence=self.pending.popleft()
            frame=self.frames.get(sequence)
            if frame is None or now-frame[0]>self.config.output_cache_ttl_sec:
                self.stats['expired'] += 1; continue
            try:
                self.send(Kind.OPUS,self.token,sequence,frame[1])
                self.stats['retransmitted'] += 1
            except OSError:
                self.stats['send_failed'] += 1
        if self.end:
            sequence, body, start, last = self.end
            if now-start>=self.config.output_end_timeout_sec:
                event('OUTPUT_END_TIMEOUT',self.token,final_seq=sequence,**self.stats)
                self.end=None
                return False
            if last is None or now-last>=self.config.output_end_retry_sec:
                try:
                    self.send(Kind.AUDIO_END,self.token,sequence,body)
                    self.stats['end_sent'] += 1
                except OSError:
                    self.stats['send_failed'] += 1
                self.end[3]=now
        return True
