"""Ordered, bounded server controls; duplicates return the original playback ACK."""
import asyncio
from bridge.transport.protocols.conversation import Kind, Packet
from bridge.diagnostics.logging import event

class ReliableEvents:
    def __init__(self, send, failed):
        self.send, self.failed = send, failed
        self.queue = asyncio.Queue(maxsize=8)
        self.sequence = 0
        self.pending = None
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.run(), name='server-events')

    def emit(self, identity, code, utterance):
        if self.sequence == 0xffffffff:
            raise RuntimeError('Event sequence exhausted')
        self.sequence += 1
        future = asyncio.get_running_loop().create_future()
        packet = Packet(Kind.EVENT, 0, identity, self.sequence,
                        bytes([code]) + utterance.to_bytes(4, 'big'))
        self.queue.put_nowait((packet, future))
        return future

    def ack(self, packet):
        if self.pending is None or len(packet.body) != 4 or packet.flags:
            return
        sent, future = self.pending
        if packet.identity == sent.identity and packet.sequence == sent.sequence and not future.done():
            future.set_result(int.from_bytes(packet.body, 'big'))
            return True
        return False

    async def run(self):
        while True:
            packet, future = await self.queue.get()
            self.pending = (packet, future)
            try:
                for attempt in range(8):
                    self.send(packet)
                    try:
                        await asyncio.wait_for(asyncio.shield(future), .15)
                        break
                    except TimeoutError:
                        event('EVENT_RETRY', packet.identity, sequence=packet.sequence, attempt=attempt+1)
                else:
                    future.cancel()
                    await self.failed('event_ack_timeout')
                    return
            finally:
                self.pending = None

    def reset(self):
        self.close(); self.sequence=0; self.pending=None; self.start()

    def close(self):
        if self.task and self.task is not asyncio.current_task(): self.task.cancel()
        if self.pending: self.pending[1].cancel()
        while not self.queue.empty(): self.queue.get_nowait()[1].cancel()
