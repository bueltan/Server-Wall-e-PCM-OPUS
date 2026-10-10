"""One bounded, ordered input worker per client session. No network dependencies."""
import asyncio
from .diagnostics import event
from dataclasses import dataclass
from typing import Awaitable, Callable

MASK = 0xffffffff

@dataclass
class IngressStats:
    accepted: int = 0
    dropped_full: int = 0
    duplicates_late: int = 0
    missing: int = 0
    send_failed: int = 0
    invalid: int = 0
    duplicate_controls: int = 0
    nack_frames: int = 0
    silence_frames: int = 0
    cache_misses: int = 0

class ReorderBuffer:
    def __init__(self, window: float, capacity: int, stats: IngressStats, expected: int = 0):
        self.window, self.capacity, self.stats = window, capacity, stats
        self.expected = expected
        self.pending = {}
        self.deadline = None

    def push(self, packet: dict, now: float) -> None:
        seq = packet['seq']
        distance = (seq - self.expected) & MASK
        if distance >= 0x80000000 or seq in self.pending:
            self.stats.duplicates_late += 1
            return
        if len(self.pending) >= self.capacity:
            self.stats.dropped_full += 1
            return
        self.pending[seq] = packet
        if seq != self.expected and self.deadline is None:
            self.deadline = now + self.window

    def drain(self, now: float, force: bool = False) -> list:
        ready = []
        while self.pending:
            if self.expected not in self.pending:
                if not force and (self.deadline is None or now < self.deadline):
                    break
                seq = min(self.pending, key=lambda s: (s - self.expected) & MASK)
                self.stats.missing += (seq - self.expected) & MASK
                self.expected = seq
            ready.append(self.pending.pop(self.expected))
            self.expected = (self.expected + 1) & MASK
            if self.expected in self.pending:
                continue
            if self.pending:
                # Existing out-of-order packets have already waited until this deadline.
                if self.deadline is None:
                    self.deadline = now + self.window
            else:
                self.deadline = None
        return ready

class InboundSession:
    """Admission is synchronous; only run() invokes the async sinks.

    Audio may be dropped, but two reserved slots preserve END and COMMIT.
    ManagedSession gates admission by v2 identity and completed COMMIT. The legacy
    reference caller retains automatic reopen. NACK and silence sinks are opt-in;
    the production v2 session supplies both.
    """
    def __init__(self, packet_format, config, audio_sink: Callable[[dict], Awaitable[bool]],
                 end_sink: Callable[[dict], Awaitable[None]], commit_sink: Callable[[], Awaitable[bool]], nack_sink=None, silence_sink=None):
        if config.input_queue_frames < 1 or config.input_reorder_window_sec < 0 or config.input_send_timeout_sec <= 0:
            raise ValueError('Invalid ingress queue/window/timeout settings')
        self.format, self.config = packet_format, config
        self.audio_sink, self.end_sink, self.commit_sink = audio_sink, end_sink, commit_sink
        self.stats = IngressStats()
        self.nack_sink = nack_sink
        self.silence_sink = silence_sink
        self.requested = set()
        self.queue = asyncio.Queue(maxsize=config.input_queue_frames + 2)
        self.audio_queued = 0
        self.end_admitted = False
        self.commit_admitted = False
        self.finished = False
        self.sealing = False
        self.closing_deadline = None
        self.reorder = ReorderBuffer(config.input_reorder_window_sec, config.input_queue_frames, self.stats)
        self.task = None
        self.end_packet = None

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.run(), name='client-input')

    def close(self):
        if self.task is not None:
            self.task.cancel()

    def validate(self, data: bytes) -> dict:
        packet = self.format.unpack_packet(data)
        self.format.validate_packet(packet)
        if packet['flags'] not in (0, self.config.flag_end):
            raise ValueError('Unsupported PCM flags')
        if self.format.is_end_packet(packet):
            if packet['frame_samples'] != 0 or packet['payload_len'] != 0:
                raise ValueError('END must have no samples or payload')
        elif packet['frame_samples'] != 320 or packet['payload_len'] != 640:
            raise ValueError('Expected PCM16 mono 16 kHz, 320 samples / 640 bytes')
        return packet

    def admit(self, data: bytes) -> bool:
        try:
            packet = self.validate(data)
        except ValueError:
            self.stats.invalid += 1
            return False
        if self.sealing or (self.closing_deadline is not None and
                asyncio.get_running_loop().time() >= self.closing_deadline):
            self.stats.duplicates_late += 1
            return False
        if self.format.is_end_packet(packet):
            if self.end_admitted or self.finished:
                self.stats.duplicate_controls += 1
                return False
            self.end_admitted = True
            self.queue.put_nowait(('end', packet))
            return True
        if self.finished:
            self.finished = False
            self.end_admitted = self.commit_admitted = False
            self.end_packet = None
            self.reorder = ReorderBuffer(self.config.input_reorder_window_sec, self.config.input_queue_frames, self.stats)
        if self.audio_queued >= self.config.input_queue_frames:
            self.stats.dropped_full += 1
            return False
        self.queue.put_nowait(('audio', packet))
        self.audio_queued += 1
        self.stats.accepted += 1
        return True

    def admit_commit(self) -> bool:
        if self.commit_admitted or self.finished:
            self.stats.duplicate_controls += 1
            return False
        self.commit_admitted = True
        self.queue.put_nowait(('commit', None))
        return True

    def _take(self, item):
        if item[0] == 'audio':
            self.audio_queued -= 1
        return item

    async def _send(self, callback, *args):
        try:
            ok = await asyncio.wait_for(callback(*args), self.config.input_send_timeout_sec)
            if ok is False:
                self.stats.send_failed += 1
        except Exception as exc:
            # CancelledError is a BaseException and still terminates the worker.
            self.stats.send_failed += 1
            event('INPUT_SEND_FAILED',getattr(self,'log_identity',None),error=type(exc).__name__)

    def _request_gaps(self):
        if self.nack_sink is None or self.audio_queued >= max(1, self.config.input_queue_frames // 2):
            return
        expected = self.reorder.expected
        self.requested = {s for s in self.requested if ((s-expected) & MASK) < 8}
        targets = list(self.reorder.pending)
        if self.end_packet is not None:
            targets.append(self.end_packet['seq'])
        distances = [(s-expected) & MASK for s in targets]
        distances = [d for d in distances if 0 < d < 0x80000000]
        if not distances:
            return
        limit = min(8, min(distances))
        mask = sum(1 << i for i in range(limit)
                   if ((expected+i) & MASK) not in self.requested)
        if not mask:
            return
        self.nack_sink(expected, mask)
        self.stats.nack_frames += mask.bit_count()
        self.requested.update((expected+i) & MASK for i in range(limit) if mask & (1 << i))
        # Extend once for the current gap, never on duplicate traffic.
        if self.reorder.deadline is not None:
            self.reorder.deadline = max(self.reorder.deadline,
                asyncio.get_running_loop().time() + self.config.input_recovery_window_sec)

    async def _pad_gap(self, missing, sequence):
        if self.silence_sink is None:
            return
        frames = min(missing, max(0, self.config.input_silence_max_ms // 20))
        if not frames:
            return
        self.stats.silence_frames += frames
        # One bounded WS append per gap. Keep silence in the accepted recording too.
        await self._send(self.silence_sink, {'seq': sequence, 'pts_samples': sequence*320,
            'frame_samples': frames*320, 'payload': bytes(frames*640)})

    async def _emit(self, force=False):
        cursor = self.reorder.expected
        for packet in self.reorder.drain(asyncio.get_running_loop().time(), force):
            missing = (packet['seq'] - cursor) & MASK
            if 0 < missing < 0x80000000:
                await self._pad_gap(missing, cursor)
            await self._send(self.audio_sink, packet)
            cursor = (packet['seq'] + 1) & MASK

    async def _consume(self, kind, packet):
        now = asyncio.get_running_loop().time()
        if kind == 'audio':
            if self.end_packet is not None and ((packet['seq']-self.end_packet['seq']) & MASK) < 0x80000000:
                self.stats.duplicates_late += 1
                return
            # Drain expired frames before admitting another to bound pending storage.
            await self._emit()
            self.reorder.push(packet, now)
            self._request_gaps()
            await self._emit()
        elif kind == 'end':
            self.end_packet = packet
            self._request_gaps()

    async def _finish(self):
        # Allow trailing UDP packets to arrive after END/COMMIT, without spawning
        # tasks for them. Deadline is fixed, so traffic cannot extend it forever.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (self.config.input_recovery_window_sec if self.nack_sink else self.config.input_reorder_window_sec)
        self.closing_deadline = deadline
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                kind, packet = self._take(await asyncio.wait_for(self.queue.get(), remaining))
            except TimeoutError:
                break
            if kind != 'commit':
                await self._consume(kind, packet)
        self.sealing = True
        # Drain packets admitted within the closing window while a WS send awaited.
        while not self.queue.empty():
            kind, packet = self._take(self.queue.get_nowait())
            if kind != 'commit':
                await self._consume(kind, packet)
        await self._emit(force=True)
        if self.end_packet is not None:
            distance = (self.end_packet['seq'] - self.reorder.expected) & MASK
            if distance < 0x80000000:
                self.stats.missing += distance
                await self._pad_gap(distance, self.reorder.expected)
            await self._send(self.end_sink, self.end_packet)
        # Seal input before awaiting COMMIT: frames arriving while it is sent
        # cannot be appended after the commit barrier by this worker.
        self.finished = True
        await self._send(self.commit_sink)
        self.closing_deadline = None
        self.sealing = False
        event('INPUT_CLOSE_SUMMARY',getattr(self,'log_identity',None),**vars(self.stats))

    async def run(self):
        while True:
            timeout = None
            if self.reorder.deadline is not None:
                timeout = max(0, self.reorder.deadline - asyncio.get_running_loop().time())
            try:
                if timeout is None:
                    item = await self.queue.get()
                elif timeout == 0:
                    await self._emit()
                    continue
                else:
                    item = await asyncio.wait_for(self.queue.get(), timeout)
            except TimeoutError:
                await self._emit()
                continue
            kind, packet = self._take(item)
            if kind == 'commit':
                await self._finish()
            else:
                await self._consume(kind, packet)
