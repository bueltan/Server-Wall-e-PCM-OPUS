"""Durable, cursor-based event polling for future PIUU notification delivery.

No audio is generated. The callback receives only state transitions, never code,
commands, prompts or summaries. Cursor commits follow successful delivery, so a
consumer must deduplicate by sequence (at-least-once delivery).
"""
import asyncio
import json
import os
from pathlib import Path
from bridge.codex.client import TaskClient
from bridge.diagnostics.logging import event

KINDS = {'running': 'task_started', 'completed': 'task_completed',
         'failed': 'task_failed', 'cancelled': 'task_cancelled'}


async def log_notification(payload):
    event('CODEX_NOTIFICATION', kind=payload['kind'], task_id=payload['task_id'],
          project_id=payload['project'], status=payload['status'], sequence=payload['sequence'])


class CodexNotifications:
    def __init__(self, client, path, deliver=log_notification):
        self.client, self.path, self.deliver = client, Path(path), deliver
        self.cursor = json.loads(self.path.read_text()) if self.path.exists() else 0
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError('invalid_codex_event_cursor')
        self.worker = None

    async def tick(self):
        response = await self.client.call('events', {'after': self.cursor, 'limit': 100})
        for item in response.get('events', []):
            if item['sequence'] <= self.cursor:
                continue
            if item['status'] in KINDS:
                await self.deliver({**item, 'kind': KINDS[item['status']]})
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = self.path.with_suffix('.tmp')
            with open(temporary, 'w', opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
                json.dump(item['sequence'], stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            self.cursor = item['sequence']

    def start(self):
        self.worker = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            try:
                await self.tick()
            except Exception:
                event('CODEX_NOTIFICATION_ERROR', message='No se pudo recuperar el historial de eventos')
            await asyncio.sleep(5)

    async def close(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)


def from_environment():
    return CodexNotifications(TaskClient.environment(), os.environ['PIUU_CODEX_EVENTS_STATE'])
