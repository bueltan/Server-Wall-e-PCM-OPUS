"""Durable local queue. Remote transport lifetime never owns an execution."""
import asyncio
import time
from bridge.codex.policy import PolicyError, identifier, text
from bridge.codex.store import Store, TERMINAL
from bridge.diagnostics.logging import event


class TaskManager:
    def __init__(self, store, registry, adapter, concurrency=1, timeout=1800):
        if type(concurrency) is not int or not 1 <= concurrency <= 4 or not 1 <= timeout <= 86400:
            raise ValueError('invalid_task_limits')
        self.store, self.registry, self.adapter = store, registry, adapter
        self.concurrency, self.timeout = concurrency, timeout
        self.active = {}
        self.worker = None
        self.stopping = False
        self.wakeup = asyncio.Event()
        self.connected = None

    def start(self):
        self.store.recover()
        self.worker = asyncio.create_task(self._loop())

    async def close(self):
        self.stopping = True
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        jobs = [job for job, _ in self.active.values()]
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)

    async def _loop(self):
        next_reconcile = 0
        while True:
            if time.monotonic() >= next_reconcile:
                await self.reconcile()
                next_reconcile = time.monotonic() + 30
            blocked = {self.store.get(key)['project'] for key in self.active}
            blocked.update(task['project'] for task in self.store.uncertain())
            for task in self.store.pending():
                if len(self.active) >= self.concurrency:
                    break
                if task['project'] not in blocked:
                    cancel = asyncio.Event()
                    job = asyncio.create_task(self._run(task, cancel))
                    self.active[task['task_id']] = (job, cancel)
                    blocked.add(task['project'])
            self.wakeup.clear()
            try:
                await asyncio.wait_for(self.wakeup.wait(), .5)
            except TimeoutError:
                pass

    async def reconcile(self):
        for task in self.store.uncertain():
            try:
                result = await self.adapter.reconcile(task)
                if result and result[0] != 'interrupted':
                    self.store.update(task['task_id'], status=result[0], result=result[1],
                                      progress_message='Resultado recuperado de Codex')
            except (OSError, ValueError, ConnectionError, TimeoutError):
                pass  # Keep uncertainty. Never replay turn/start.

    async def _run(self, task, cancel):
        key = task['task_id']
        try:
            # Recheck cancellation after scheduling but before starting Codex.
            if self.store.get(key)['status'] != 'queued':
                return
            project = self.registry.get(task['project'])
            if task['parent']:
                parent = self.store.get(task['parent'])
                if parent['status'] not in TERMINAL or not parent['thread_id']:
                    raise PolicyError('session_not_ready')
                task['thread_id'] = parent['thread_id']
            self.store.update(key, status='running', started=time.time(), progress_message='Iniciando Codex')

            async def emit(**fields):
                self.store.update(key, **fields)
                if 'progress_message' in fields:
                    event('CODEX_TASK_PROGRESS', task_id=key, project_id=task['project'],
                          status='running', message=fields['progress_message'])

            async with asyncio.timeout(self.timeout):
                state, result = await self.adapter.run(task, project, emit, cancel)
            self.store.update(key, status=state, result=result, progress_message={
                'completed': 'Tarea terminada', 'failed': 'Codex informó un fallo',
                'cancelled': 'Cancelación confirmada', 'interrupted': 'Resultado pendiente de reconciliación'}[state])
        except asyncio.CancelledError:
            self.store.update(key, status='interrupted', progress_message='Servicio detenido; resultado sin confirmar')
            raise
        except (TimeoutError, ConnectionError, OSError):
            self.store.update(key, status='interrupted', progress_message='Conexión o tiempo agotado; resultado sin confirmar')
        except PolicyError as exc:
            # Policy exceptions contain only our fixed codes.
            self.store.update(key, status='failed', progress_message=str(exc))
        except Exception:
            self.store.update(key, status='interrupted', progress_message='Error interno; resultado sin confirmar')
        finally:
            self.active.pop(key, None)
            self.wakeup.set()

    async def dispatch(self, operation, args):
        if not isinstance(args, dict):
            raise PolicyError('invalid_arguments')
        fields = {
            'status': set(), 'projects': set(), 'execute': {'project', 'instruction', 'mode', 'idempotency_key'},
            'task_status': {'task_id'}, 'task_result': {'task_id', 'include_details'}, 'cancel': {'task_id'},
            'task_list': {'limit'}, 'continue': {'task_id', 'instruction', 'idempotency_key'},
            'events': {'after', 'limit'},
        }
        if operation not in fields or set(args) - fields[operation]:
            raise PolicyError('invalid_operation')
        if operation == 'status':
            connected = await self.adapter.available()
            if connected != self.connected:
                event('CODEX_CONNECTED' if connected else 'CODEX_DISCONNECTED',
                      task_id=None, project_id=None, status='connected' if connected else 'disconnected',
                      message='Disponibilidad del App Server')
                self.connected = connected
            return dict(connected=connected, agent='codex', status='busy' if self.active else 'idle',
                        active_tasks=len(self.active))
        if operation == 'projects':
            return self.registry.listing()
        if operation in ('execute', 'continue'):
            instruction = text(args.get('instruction'), 'invalid_instruction')
            key = identifier(args.get('idempotency_key'))
            parent_id = None
            if operation == 'continue':
                parent_id = identifier(args.get('task_id'))
                parent = self.store.get(parent_id)
                project_id = parent['project']
                if parent['status'] not in TERMINAL or not parent['thread_id']:
                    raise PolicyError('session_not_ready')
            else:
                project_id = args.get('project')
                if args.get('mode', 'autonomous') != 'autonomous':
                    raise PolicyError('unsupported_mode')
            self.registry.get(project_id)
            task = self.store.create(project_id, instruction, key, parent_id)
            self.wakeup.set()
            return {'task_id': task['task_id'], 'status': task['status']}
        if operation == 'task_list':
            limit = bounded(args.get('limit', 20), 1, 100)
            return {'tasks': [Store.public(t) for t in self.store.recent(limit)]}
        if operation == 'events':
            after = bounded(args.get('after', 0), 0, 2**63 - 1)
            limit = bounded(args.get('limit', 100), 1, 100)
            return {'events': self.store.events(after, limit)}
        key = identifier(args.get('task_id'))
        task = self.store.get(key)
        if operation == 'cancel' and task['status'] not in TERMINAL:
            self.store.update(key, cancel_requested=1)
            if task['status'] == 'queued':
                self.store.update(key, status='cancelled', progress_message='Cancelada antes de empezar')
            elif key in self.active:
                self.active[key][1].set()
            task = self.store.get(key)
        response = Store.public(task, operation == 'task_result')
        if operation == 'task_result':
            if type(args.get('include_details', False)) is not bool:
                raise PolicyError('invalid_arguments')
            if not args.get('include_details', False):
                response.pop('commands', None)
        return response


def bounded(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise PolicyError('invalid_limit')
    return value
