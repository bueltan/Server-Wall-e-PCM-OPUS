"""Authenticated loopback task API, the only endpoint forwarded over SSH."""
import asyncio
import fcntl
import hmac
import json
import os
import shutil
from http import HTTPStatus
from pathlib import Path
from websockets.asyncio.server import serve
from bridge.codex.adapter import CodexAdapter
from bridge.codex.manager import TaskManager
from bridge.codex.policy import PolicyError, Registry, token_file
from bridge.codex.store import Store
from bridge.diagnostics.logging import configure_logging, event


class Gateway:
    def __init__(self, manager, token):
        if len(token) < 32:
            raise PolicyError('token_too_short')
        self.manager, self.token = manager, token
        self.slots = asyncio.Semaphore(16)

    def authenticate(self, connection, request):
        values = request.headers.get_all('Authorization')
        if len(values) != 1 or not hmac.compare_digest(values[0], 'Bearer ' + self.token):
            return connection.respond(HTTPStatus.UNAUTHORIZED, 'Unauthorized\n')
        # Browser origins must never reach a local development agent.
        if request.headers.get_all('Origin'):
            return connection.respond(HTTPStatus.FORBIDDEN, 'Forbidden\n')
        if request.path != '/':
            return connection.respond(HTTPStatus.NOT_FOUND, 'Not found\n')

    async def handle(self, socket):
        async for raw in socket:
            request_id = None
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict) or set(payload) != {'id', 'operation', 'arguments'}:
                    raise PolicyError('invalid_request')
                request_id = payload['id']
                if type(request_id) is not int or not 0 <= request_id < 2**53:
                    request_id = None
                    raise PolicyError('invalid_request')
                if not isinstance(payload['operation'], str):
                    raise PolicyError('invalid_operation')
                async with self.slots:
                    async with asyncio.timeout(20):
                        result = await self.manager.dispatch(payload['operation'], payload['arguments'])
                reply = {'id': request_id, 'result': result}
            except PolicyError as exc:
                reply = {'id': request_id, 'error': str(exc)}
            except (ValueError, KeyError, TypeError):
                reply = {'id': request_id, 'error': 'invalid_request'}
            except Exception:
                event('CODEX_GATEWAY_ERROR', message='Error interno del gateway')
                reply = {'id': request_id, 'error': 'codex_unavailable'}
            await socket.send(json.dumps(reply, ensure_ascii=False))

    async def start(self, host='127.0.0.1', port=4500):
        if host != '127.0.0.1':
            raise PolicyError('loopback_required')
        return await serve(self.handle, host, port, process_request=self.authenticate,
                           max_size=32768, max_queue=8, ping_interval=15, ping_timeout=30,
                           open_timeout=5, close_timeout=2, server_header=None)


def check_runtime():
    # The launcher must establish these boundaries before this network service
    # reads configuration or accepts connections. Never silently run unconfined.
    status = Path('/proc/self/status').read_text()
    if os.geteuid() == 0 or 'NoNewPrivs:\t1' not in status:
        raise PolicyError('isolated_runtime_required')
    virtual_devices = {'core', 'fd', 'full', 'null', 'ptmx', 'pts', 'random', 'shm',
                       'stderr', 'stdin', 'stdout', 'tty', 'urandom', 'zero'}
    if set(os.listdir('/dev')) - virtual_devices:
        raise PolicyError('private_devices_required')


async def main():
    check_runtime()
    os.umask(0o077)
    configure_logging()
    binary = os.getenv('PIUU_CODEX_BINARY') or shutil.which('codex')
    if not binary or not Path(binary).is_absolute():
        raise PolicyError('codex_binary_not_found')
    database = Path(os.environ['PIUU_CODEX_DB_PATH']).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Prevent a second manager from recovering another live manager's tasks.
    with open(str(database) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(database)
        manager = TaskManager(store, Registry(os.environ['PIUU_CODEX_PROJECTS_FILE']),
            CodexAdapter(binary, os.environ['PIUU_CODEX_HOME'],
                         float(os.getenv('PIUU_CODEX_RPC_TIMEOUT', '10'))),
            int(os.getenv('PIUU_CODEX_MAX_CONCURRENT_TASKS', '1')),
            int(os.getenv('PIUU_CODEX_TASK_TIMEOUT', '1800')))
        gateway = Gateway(manager, token_file(os.environ['PIUU_CODEX_TOKEN_FILE']))
        server = await gateway.start(os.getenv('PIUU_CODEX_HOST', '127.0.0.1'),
                                     int(os.getenv('PIUU_CODEX_PORT', '4500')))
        manager.start()
        try:
            await asyncio.Future()
        finally:
            server.close()
            await server.wait_closed()
            await manager.close()
            store.close()


if __name__ == '__main__':
    asyncio.run(main())
