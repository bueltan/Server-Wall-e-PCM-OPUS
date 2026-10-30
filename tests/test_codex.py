"""Offline task lifecycle, real MCP/WS boundary and protocol regressions."""
import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus
from bridge.codex.adapter import AppServer, CodexAdapter, permission_config, check_result
from bridge.codex.client import TaskClient
from bridge.codex.gateway import Gateway, check_runtime
from bridge.codex.manager import TaskManager
from bridge.codex.policy import PolicyError, Project, Registry, private_file
from bridge.codex.store import Store
from bridge.config import BridgeConfig
from bridge.mcp_servers.codex import create_server
from bridge.notifications.codex import CodexNotifications
from bridge.providers.gemini.live import GeminiLive
from bridge.tools.manager import ToolManager, ServerSpec


def setUpModule():
    global logging_levels
    logging_levels = {name: logging.getLogger(name).level for name in ('', 'piuu')}


def tearDownModule():
    for name, level in logging_levels.items():
        logging.getLogger(name).setLevel(level)


class FakeAdapter:
    def __init__(self):
        self.online = True
        self.calls = []
        self.release = asyncio.Event()
        self.recovered = None

    async def available(self):
        return self.online

    async def run(self, task, project, emit, cancel):
        self.calls.append(task)
        await emit(thread_id='thread_one', turn_id='turn_one', progress_message='Ejecutando pruebas')
        while not self.release.is_set():
            if cancel.is_set():
                return 'cancelled', {'summary': None}
            await asyncio.sleep(.005)
        return 'completed', {'summary': 'Pruebas informadas por el agente', 'tests_passed': None,
                             'build_success': None, 'files_modified': None}

    async def reconcile(self, task):
        return self.recovered


class Fixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        (self.project / 'platformio.ini').write_text('[env:one]\n[env:two]\n')
        self.registry_path = self.root / 'projects.json'
        self.registry_path.write_text(json.dumps({'projects': [dict(id='test', name='Test', path=str(self.project))]}))
        self.registry_path.chmod(0o600)
        self.registry = Registry(self.registry_path)
        self.store = Store(self.root / 'tasks.db')
        self.adapter = FakeAdapter()
        self.manager = TaskManager(self.store, self.registry, self.adapter, timeout=1)
        self.server = None

    async def asyncTearDown(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        await self.manager.close()
        self.store.close()
        self.temporary.cleanup()

    async def enqueue(self, key='key', instruction='Run tests'):
        return await self.manager.dispatch('execute', dict(project='test', instruction=instruction, idempotency_key=key))

    async def wait_status(self, task_id, desired):
        for _ in range(300):
            row = self.store.get(task_id)
            if row['status'] == desired:
                return row
            await asyncio.sleep(.005)
        self.fail(f"Task did not reach {desired}: {self.store.get(task_id)['status']}")

    async def gateway(self):
        self.server = await Gateway(self.manager, 'x' * 40).start(port=0)
        return self.server.sockets[0].getsockname()[1]


class TaskTests(Fixture):
    async def test_create_execute_result_and_events(self):
        task = await self.enqueue()
        self.assertEqual(task['status'], 'queued')
        self.manager.start()
        await self.wait_status(task['task_id'], 'running')
        self.adapter.release.set()
        await self.wait_status(task['task_id'], 'completed')
        result = await self.manager.dispatch('task_result', {'task_id': task['task_id']})
        self.assertIsNone(result['tests_passed'])
        self.assertIsNone(result['build_success'])
        self.assertEqual([e['status'] for e in self.store.events()], ['queued', 'running', 'completed'])
        self.assertEqual(len((await self.manager.dispatch('task_list', {}))['tasks']), 1)

    async def test_duplicate_key_does_not_execute_twice(self):
        a, b = await asyncio.gather(self.enqueue(), self.enqueue())
        self.assertEqual(a, b)
        self.manager.start()
        await self.wait_status(a['task_id'], 'running')
        self.assertEqual(len(self.adapter.calls), 1)
        with self.assertRaisesRegex(PolicyError, 'idempotency_conflict'):
            await self.enqueue(instruction='different')

    async def test_cancel_queued(self):
        task = await self.enqueue()
        result = await self.manager.dispatch('cancel', {'task_id': task['task_id']})
        self.assertEqual(result['status'], 'cancelled')
        self.manager.start()
        await asyncio.sleep(.02)
        self.assertFalse(self.adapter.calls)

    async def test_cancel_running_confirmed(self):
        task = await self.enqueue()
        self.manager.start()
        await self.wait_status(task['task_id'], 'running')
        result = await self.manager.dispatch('cancel', {'task_id': task['task_id']})
        self.assertEqual(result['status'], 'running')
        self.assertTrue(result['cancel_requested'])
        await self.wait_status(task['task_id'], 'cancelled')

    async def test_continue_keeps_thread(self):
        task = await self.enqueue()
        self.adapter.release.set()
        self.manager.start()
        await self.wait_status(task['task_id'], 'completed')
        follow = await self.manager.dispatch('continue', dict(task_id=task['task_id'], instruction='next', idempotency_key='next'))
        await self.wait_status(follow['task_id'], 'completed')
        self.assertEqual(self.adapter.calls[1]['thread_id'], 'thread_one')

    async def test_continue_active_rejected(self):
        task = await self.enqueue()
        with self.assertRaisesRegex(PolicyError, 'session_not_ready'):
            await self.manager.dispatch('continue', dict(task_id=task['task_id'], instruction='next', idempotency_key='next'))

    async def test_timeout_uncertain_not_replayed(self):
        self.manager.timeout = .02
        task = await self.enqueue()
        self.manager.start()
        await self.wait_status(task['task_id'], 'interrupted')
        await self.enqueue('second')
        await asyncio.sleep(.55)
        self.assertEqual(len(self.adapter.calls), 1)

    async def test_same_project_is_serialized(self):
        first, second = await self.enqueue(), await self.enqueue('second')
        self.manager.concurrency = 2
        self.manager.start()
        await self.wait_status(first['task_id'], 'running')
        await asyncio.sleep(.03)
        self.assertEqual(self.store.get(second['task_id'])['status'], 'queued')

    async def test_restart_recovers_without_replay(self):
        task = await self.enqueue()
        self.store.update(task['task_id'], status='running', thread_id='saved', turn_id='saved')
        self.manager.start()
        await self.wait_status(task['task_id'], 'interrupted')
        self.assertFalse(self.adapter.calls)
        self.adapter.recovered = ('completed', {'summary': 'Recovered'})
        await self.manager.reconcile()
        self.assertEqual(self.store.get(task['task_id'])['status'], 'completed')
        self.assertFalse(self.adapter.calls)

    async def test_persistent_history(self):
        task = await self.enqueue()
        self.store.close()
        self.store = Store(self.root / 'tasks.db')
        self.manager.store = self.store
        self.assertEqual(self.store.get(task['task_id'])['status'], 'queued')
        self.assertEqual(len(self.store.events()), 1)

    async def test_offline_agent(self):
        self.adapter.online = False
        result = await self.manager.dispatch('status', {})
        self.assertFalse(result['connected'])
        self.adapter.online = True
        self.assertTrue((await self.manager.dispatch('status', {}))['connected'])

    async def test_shutdown_after_completion_is_bounded(self):
        task = await self.enqueue()
        self.adapter.release.set()
        self.manager.start()
        await self.wait_status(task['task_id'], 'completed')
        async with asyncio.timeout(1):
            await self.manager.close()
        self.assertTrue(self.manager.worker.done())

    async def test_invalid_requests(self):
        for operation, args in [
            ('execute', dict(project='../outside', instruction='x', idempotency_key='x')),
            ('execute', dict(project='unknown', instruction='x', idempotency_key='x')),
            ('execute', dict(project='test', instruction='x', idempotency_key='x', mode='unsafe')),
            ('execute', dict(project='test', instruction='x', idempotency_key='x', cwd='/tmp')),
            ('execute', dict(project='test', instruction='', idempotency_key='x')),
            ('task_list', {'limit': True}), ('events', {'after': -1}),
            ('command/exec', {}), ('execute', []), ('execute', dict(project='test', instruction='x')),
        ]:
            with self.subTest(operation=operation, args=args), self.assertRaises(PolicyError):
                await self.manager.dispatch(operation, args)


class PolicyTests(Fixture):
    async def test_environments_and_no_paths_returned(self):
        data = self.registry.listing()
        self.assertEqual(data['projects'][0]['environments'], ['one', 'two'])
        self.assertNotIn(str(self.project), json.dumps(data))

    async def test_symlink_escape(self):
        (self.project / 'outside').symlink_to(self.root)
        with self.assertRaisesRegex(PolicyError, 'symlink_escape'):
            self.registry.get('test')
        self.assertFalse(self.registry.listing()['projects'][0]['available'])

    async def test_root_replaced_by_symlink(self):
        self.project.rename(self.root / 'old')
        self.project.symlink_to(self.root / 'old')
        with self.assertRaisesRegex(PolicyError, 'project_path_changed'):
            self.registry.get('test')

    async def test_private_config_required(self):
        self.registry_path.chmod(0o644)
        with self.assertRaisesRegex(PolicyError, 'private_file_required'):
            Registry(self.registry_path)

    async def test_registry_traversal_rejected(self):
        data = json.loads(self.registry_path.read_text())
        data['projects'][0]['path'] = str(self.project) + '/../project'
        self.registry_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(PolicyError, 'invalid_project_path'):
            Registry(self.registry_path)

    async def test_permission_profile_is_restricted(self):
        config = permission_config(self.registry.get('test'))
        profile = config['permissions']['piuu']
        self.assertFalse(profile['network']['enabled'])
        self.assertNotIn(':root', profile['filesystem'])
        self.assertEqual(profile['filesystem'][str(self.project)], 'write')
        self.assertEqual(profile['filesystem'][str(self.project / '.env')], 'deny')
        self.assertEqual(config['shell_environment_policy']['inherit'], 'none')

    async def test_unclean_codex_home_rejected(self):
        (self.root / 'config.toml').write_text('')
        adapter = AppServer('/not/used', self.root)
        with self.assertRaisesRegex(PolicyError, 'not_clean'):
            await adapter.start()


class GatewayTests(Fixture):
    async def test_authentication_and_origin(self):
        port = await self.gateway()
        for headers in ({}, {'Authorization': 'Bearer wrong'},
                        {'Authorization': 'Bearer ' + 'x'*40, 'Origin': 'http://example.com'}):
            with self.assertRaises(InvalidStatus):
                async with connect(f'ws://127.0.0.1:{port}', additional_headers=headers, proxy=None):
                    pass
        self.assertEqual(self.store.recent(), [])

    async def test_bad_operations_and_extra_fields(self):
        client = TaskClient(port=await self.gateway(), token='x'*40)
        self.assertEqual((await client.call('execute', {'cwd': '/'}))['error'], 'invalid_operation')
        self.assertEqual((await client.call('raw_codex_rpc', {}))['error'], 'invalid_operation')

    async def test_disconnect_does_not_cancel_and_reconnect_deduplicates(self):
        port = await self.gateway()
        client = TaskClient(port=port, token='x'*40)
        self.manager.start()
        args = dict(project='test', instruction='test', idempotency_key='persist')
        first = await client.call('execute', args)
        await self.wait_status(first['task_id'], 'running')
        self.server.close()
        await self.server.wait_closed()
        self.assertEqual(self.store.get(first['task_id'])['status'], 'running')
        offline = await client.call('execute', args)
        self.assertTrue(offline['outcome_unknown'])
        self.server = await Gateway(self.manager, 'x'*40).start(port=port)
        repeated = await client.call('execute', args)
        self.assertEqual(repeated['task_id'], first['task_id'])
        self.assertEqual(len(self.adapter.calls), 1)
        self.adapter.release.set()
        await self.wait_status(first['task_id'], 'completed')

    async def test_loopback_required(self):
        with self.assertRaisesRegex(PolicyError, 'loopback_required'):
            await Gateway(self.manager, 'x'*40).start('0.0.0.0', 0)

    async def test_real_mcp_stdio(self):
        port = await self.gateway()
        token = self.root / 'token'
        token.write_text('x'*40)
        token.chmod(0o600)
        env = dict(PATH=os.environ['PATH'], PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'),
                   PIUU_CODEX_TOKEN_FILE=str(token), PIUU_CODEX_PORT=str(port))
        manager = ToolManager([ServerSpec('codex', sys.executable, ('-m', 'bridge.mcp_servers.codex'), env)], 10)
        try:
            await manager.start()
            self.assertEqual(len(manager.declarations), 8)
            self.assertIn('codex_continue', manager.routes)
            result = await manager.call('codex_execute', dict(project='test', instruction='test', idempotency_key='mcp'))
            self.assertTrue(result['ok'])
            self.assertEqual(result['data']['status'], 'queued')
            invalid = await manager.call('codex_execute', dict(project='../bad', instruction='test', idempotency_key='mcp'))
            self.assertFalse(invalid['ok'])
        finally:
            await manager.close()

    async def test_notifications_cursor_survives_restart(self):
        client = TaskClient(port=await self.gateway(), token='x'*40)
        task = await self.enqueue()
        self.manager.start()
        await self.wait_status(task['task_id'], 'running')
        deliver = AsyncMock()
        notices = CodexNotifications(client, self.root/'cursor', deliver)
        await notices.tick()
        self.assertEqual(deliver.await_args.args[0]['kind'], 'task_started')
        self.adapter.release.set()
        await self.wait_status(task['task_id'], 'completed')
        await notices.tick()
        self.assertEqual(deliver.await_count, 2)
        await CodexNotifications(client, self.root/'cursor', deliver).tick()
        self.assertEqual(deliver.await_count, 2)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_events_and_cancel(self):
        class FakeServer:
            def __init__(self, *args):
                self.notifications = asyncio.Queue()
                self.reader = asyncio.create_task(asyncio.sleep(10))
                self.calls = []
            async def start(self, project=None):
                return self
            async def rpc(self, method, params):
                self.calls.append((method, params))
                if method == 'thread/start':
                    return {'thread': {'id': 'thread'}, 'activePermissionProfile': {'id': 'piuu'},
                            'sandbox': {'type': 'workspaceWrite', 'networkAccess': False}}
                if method == 'turn/start':
                    self.notifications.put_nowait({'method': 'item/completed', 'params': {'threadId': 'thread',
                        'item': {'type': 'commandExecution', 'command': 'pio run', 'exitCode': 0, 'durationMs': 10}}})
                    self.notifications.put_nowait({'method': 'item/completed', 'params': {'threadId': 'thread',
                        'item': {'type': 'agentMessage', 'text': 'Hecho'}}})
                    self.notifications.put_nowait({'method': 'turn/completed', 'params': {'threadId': 'thread',
                        'turn': {'id': 'turn', 'status': 'completed'}}})
                    return {'turn': {'id': 'turn'}}
            async def close(self):
                self.reader.cancel()
                await asyncio.gather(self.reader, return_exceptions=True)
        adapter = CodexAdapter('unused', '/unused', server_factory=FakeServer)
        result = await adapter.run({'task_id': 'task', 'instruction': 'test'}, Project('p','p',Path('/tmp')),
                                   AsyncMock(), asyncio.Event())
        self.assertEqual(result[0], 'completed')
        self.assertEqual(result[1]['commands'][0]['exit_code'], 0)
        self.assertEqual(result[1]['summary'], 'Hecho')
        self.assertIsNone(result[1]['tests_passed'])

    async def test_gemini_registration_and_default_disabled(self):
        config = replace(BridgeConfig(), codex_enabled=True)
        provider = GeminiLive(config, None, lambda *_: True, None)
        provider.codex_manager.declarations = [{'name': 'codex_status'}]
        setup = provider.setup()['setup']
        self.assertTrue(any(d['name'] == 'codex_status' for t in setup['tools'] for d in t['functionDeclarations']))
        self.assertIn('idempotency_key', setup['systemInstruction']['parts'][0]['text'])
        self.assertEqual(len(provider.codex_manager.servers), 1)
        await provider.close()

    async def test_check_evidence_requires_recognized_literal_commands(self):
        self.assertTrue(check_result([{'command': "python3 -m unittest -v", 'exit_code': 0}], 'test'))
        self.assertTrue(check_result([{'command': "/bin/bash -lc 'python3 -m unittest -v'", 'exit_code': 0}], 'test'))
        self.assertFalse(check_result([{'command': 'pytest', 'exit_code': 1}], 'test'))
        self.assertIsNone(check_result([{'command': "/bin/bash -lc 'pytest || true'", 'exit_code': 0}], 'test'))
        self.assertIsNone(check_result([{'command': 'pio run -t upload', 'exit_code': 0}], 'build'))
        self.assertIsNone(check_result([{'command': 'echo tests passed', 'exit_code': 0}], 'test'))
        self.assertTrue(check_result([{'command': 'pio run -e native', 'exit_code': 0}], 'build'))
        self.assertIsNone(check_result([], 'build'))

    async def test_gateway_requires_nnp_and_private_devices(self):
        with patch('bridge.codex.gateway.Path.read_text', return_value='NoNewPrivs:\t1'), \
             patch('bridge.codex.gateway.os.geteuid', return_value=1000), \
             patch('bridge.codex.gateway.os.listdir', return_value=['null', 'zero', 'urandom']):
            check_runtime()
            with patch('bridge.codex.gateway.Path.read_text', return_value='NoNewPrivs:\t0'):
                with self.assertRaisesRegex(PolicyError, 'isolated_runtime_required'):
                    check_runtime()
            with patch('bridge.codex.gateway.os.listdir', return_value=['null', 'ttyUSB0']):
                with self.assertRaisesRegex(PolicyError, 'private_devices_required'):
                    check_runtime()
            with patch('bridge.codex.gateway.os.geteuid', return_value=0):
                with self.assertRaisesRegex(PolicyError, 'isolated_runtime_required'):
                    check_runtime()

    async def test_mcp_tools_have_safe_schema(self):
        server = create_server(AsyncMock())
        tools = await server.list_tools()
        self.assertEqual(len(tools), 8)
        execute = next(t for t in tools if t.name == 'execute')
        self.assertIn('idempotency_key', execute.inputSchema['required'])
        self.assertTrue(execute.annotations.destructiveHint)


if __name__ == '__main__':
    unittest.main()
