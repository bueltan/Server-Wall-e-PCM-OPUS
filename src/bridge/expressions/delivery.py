"""One optional plan per response; bounded retries never gate or fail audio."""
import time
from bridge.transport.protocols.conversation import Kind, Packet
from bridge.transport.protocols.eyes import encode_plan
from bridge.diagnostics.logging import event


class EyePlanDelivery:
    RETRY_SEC = 0.1
    DEADLINE_SEC = 0.5
    MAX_ATTEMPTS = 3

    def __init__(self, send):
        self.send = send
        self.stats = dict(plans=0, attempts=0, acked=0, unconfirmed=0, send_errors=0)
        self.reset()

    def reset(self):
        self.identity = None
        self.pending = None
        self.attempts = 0
        self.started = self.last_sent = 0.0

    def start(self, identity, plan, now=None):
        # The sequence is immutable for the whole response, even after its ACK.
        if self.identity == identity:
            return False
        body = encode_plan(plan)
        if not identity or not all((identity.client, identity.server, identity.turn, identity.response)):
            raise ValueError('plan_requires_response_identity')
        self.reset()
        self.identity = identity
        self.pending = Packet(Kind.EYES_PLAN, 0, identity, 1, body)
        self.started = time.monotonic() if now is None else now
        self.stats['plans'] += 1
        self._attempt(self.started)
        return True

    def _attempt(self, now):
        self.attempts += 1
        self.stats['attempts'] += 1
        self.last_sent = now
        try:
            self.send(self.pending)
        except (OSError, RuntimeError):
            self.stats['send_errors'] += 1
            event('EYES_PLAN_SEND_ERROR', self.identity, attempt=self.attempts)
        else:
            event('EYES_PLAN_SENT', self.identity, sequence=1, attempt=self.attempts,
                  bytes=len(self.pending.body))

    def poll(self, now=None):
        if self.pending is None:
            return
        now = time.monotonic() if now is None else now
        if now - self.started >= self.DEADLINE_SEC:
            self.pending = None
            self.stats['unconfirmed'] += 1
            event('EYES_PLAN_UNCONFIRMED', self.identity, attempts=self.attempts)
        elif self.attempts < self.MAX_ATTEMPTS and now - self.last_sent >= self.RETRY_SEC:
            self._attempt(now)

    def acknowledge(self, packet):
        if (self.pending is None or packet.kind != Kind.EYES_PLAN_ACK or packet.flags or packet.body or
                packet.identity != self.identity or packet.sequence != self.pending.sequence):
            return False
        self.pending = None
        self.stats['acked'] += 1
        event('EYES_PLAN_ACK', self.identity, sequence=packet.sequence)
        return True
