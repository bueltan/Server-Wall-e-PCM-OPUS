"""One bounded interruption transaction per response, independent of UDP/xAI schemas."""
import asyncio
import time
from dataclasses import dataclass
from .diagnostics import event


@dataclass
class Repair:
    token: object
    conversation: object
    played_floor: int = 0
    manual: bool = False
    phase: str = 'core_ack'
    task: object = None


class InterruptionCoordinator:
    def __init__(self, session):
        self.session = session
        self.active = {}

    @property
    def pending(self):
        return bool(self.active)

    def begin(self, token, acknowledgement, *, manual=False, played_ms=0):
        if token is None:
            return
        existing = self.active.get(token)
        if existing:
            existing.played_floor = max(existing.played_floor, played_ms)
            existing.manual |= manual
            event('INTERRUPT_MERGED', token, manual=manual, played_ms=played_ms)
            return existing.task
        repair = Repair(token, self.session.conversation.identity, played_ms, manual)
        self.active[token] = repair
        self.session.xai.pause_input()
        repair.task = self.session.schedule(self.run(repair, acknowledgement))
        return repair.task

    async def run(self, repair, acknowledgement):
        session = self.session
        started = time.monotonic()
        event('INTERRUPT_BEGIN', repair.token, manual=repair.manual)
        try:
            await asyncio.wait_for(self.execute(repair, acknowledgement),
                                   session.config.history_repair_timeout_sec)
            event('INTERRUPT_COMPLETE', repair.token,
                  elapsed_ms=round((time.monotonic()-started)*1000))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event('HISTORY_REPAIR_FAILED', repair.token, phase=repair.phase,
                  error=type(exc).__name__, elapsed_ms=round((time.monotonic()-started)*1000))
            if session.conversation.identity == repair.conversation and not session.closing:
                await session.fail('history_uncertain')
        finally:
            self.active.pop(repair.token, None)
            # close_remote joins cancelled transactions before reopening a provider.
            session.xai.resume_input()

    async def execute(self, repair, acknowledgement):
        # Cancellation can progress while UDP playback acknowledgement is in flight.
        if repair.manual:
            repair.phase = 'cancel_send'
            await self.session.xai.cancel_response(repair.token)
        repair.phase = 'core_ack'
        played = max(await acknowledgement, repair.played_floor)
        event('INTERRUPT_PLAYBACK_ACK', repair.token, played_ms=played)
        repair.phase = 'history_confirmation'
        await self.session.xai.repair_history(repair.token, played,
            phase=lambda value: setattr(repair,"phase",value))

    async def close(self):
        current = asyncio.current_task()
        tasks = [r.task for r in self.active.values() if r.task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
