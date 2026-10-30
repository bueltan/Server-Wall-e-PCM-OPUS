"""Remote side task client. A lost reply never triggers mutation replay."""
import asyncio
import json
import os
from websockets.asyncio.client import connect
from bridge.codex.policy import PolicyError, token_file


class TaskClient:
    def __init__(self, host='127.0.0.1', port=4500, token='', timeout=12):
        if host != '127.0.0.1' or not 1 <= port <= 65535 or len(token) < 32:
            raise PolicyError('invalid_client_configuration')
        self.url, self.token, self.timeout = f'ws://{host}:{port}', token, timeout

    @classmethod
    def environment(cls):
        return cls(os.getenv('PIUU_CODEX_HOST', '127.0.0.1'), int(os.getenv('PIUU_CODEX_PORT', '4500')),
                   token_file(os.environ['PIUU_CODEX_TOKEN_FILE']))

    async def call(self, operation, arguments):
        try:
            async with asyncio.timeout(self.timeout):
                async with connect(self.url, additional_headers={'Authorization': 'Bearer ' + self.token},
                                   max_size=262144, open_timeout=self.timeout, close_timeout=1, proxy=None) as ws:
                    await ws.send(json.dumps({'id': 1, 'operation': operation, 'arguments': arguments}))
                    reply = json.loads(await ws.recv())
                    if reply.get('id') != 1:
                        raise ValueError('invalid_reply')
                    if 'error' in reply:
                        return {'error': reply['error'], 'retry_safe': operation not in ('execute', 'continue')}
                    return reply['result']
        except asyncio.CancelledError:
            raise
        except Exception:
            # The task may have been accepted. Use the SAME idempotency key on an
            # explicit retry, or consult task_list after connectivity recovers.
            return {'error': 'codex_transport_unavailable',
                    'outcome_unknown': operation in ('execute', 'continue', 'cancel'),
                    'retry_safe': False}
