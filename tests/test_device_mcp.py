"""Offline UDP transactions plus real local stdio -> WebSocket -> session checks."""
import asyncio
import json
import secrets
import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus
from bridge.config import BridgeConfig
from bridge.conversation.session import ContinuousSession
from bridge.mcp_servers.device import DeviceClient, create_server
from bridge.providers.gemini.live import GeminiLive
from bridge.tools.device_gateway import DeviceGateway
from bridge.tools.manager import ToolManager, device_server_spec
from bridge.transport.device_control import VolumeTransactions, volume_body
from bridge.transport.protocols.conversation import CAP_VOLUME_CONTROL, Identity, Kind, Packet, encode, decode


class DeviceSession:
    def __init__(self):
        self.config = BridgeConfig()
        self.conversation = SimpleNamespace(identity=Identity(1, 2, 3, 0))
        self.volume_capable = True
        self.closing = False
        self.transport = object()
        self.last_peer_seen = __import__('time').monotonic()
        self.volume = VolumeTransactions(self, timeout=.08, retry=.01)
        self.sent = []
        self.level, self.restore, self.saved = 50, 50, 20
        self.ack = True
        self.drop_first = False
        self.seen = {}
        self.applied = 0

    def _send(self, kind, identity, sequence, body):
        self.sent.append(Packet(kind, 0, identity, sequence, body))
        if sequence not in self.seen:
            operation, value, persist = struct.unpack('!BbB', body)
            if operation == 1:
                self.level = value
            elif operation == 2:
                self.level = max(0, min(100, self.level + value))
            elif operation == 3:
                self.level = 0 if value else self.level or self.restore
            if self.level:
                self.restore = self.level
            if persist:
                self.saved = self.level
            self.applied += operation != 0
            self.seen[sequence] = bytes((0, self.level, self.restore, self.saved))
            if self.drop_first:
                return
        if self.ack:
            self.volume.receive(Packet(Kind.VOLUME_ACK, 0, identity, sequence, self.seen[sequence]))


class VolumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_and_unexpected_exceptions_are_sanitized_logged_and_release_pending(self):
        private = 'PRIVATE_token_payload'
        for exception, code in ((OSError(private), 'device_transport_error'),
                                (RuntimeError(private), 'device_internal_error')):
            session = DeviceSession()
            session._send = lambda *args: (_ for _ in ()).throw(exception)
            with self.assertLogs('piuu', level='INFO') as logs:
                result = await session.volume.call('adjust_volume', {'delta': -20},
                                                   session.conversation.identity, 'a' * 32)
            self.assertEqual(result['error'], code)
            self.assertEqual(result['status'], 'unknown')
            self.assertIn('DEVICE_VOLUME_FINISHED', '\n'.join(logs.output))
            self.assertNotIn(private, str(result) + str(logs.output))
            self.assertIsNone(session.volume.pending)
            self.assertFalse(session.volume.lock.locked())

    async def test_retry_and_cancellation_logs_are_correlated_without_arguments(self):
        session = DeviceSession()
        session.drop_first = True
        with self.assertLogs('piuu', level='INFO') as logs:
            await session.volume.call('adjust_volume', {'delta': -20}, session.conversation.identity, 'a' * 32)
        self.assertIn('DEVICE_VOLUME_RETRY', str(logs.output))
        self.assertIn('request_id=' + repr('a' * 32), str(logs.output))
        session.ack = False
        with self.assertLogs('piuu', level='INFO') as logs:
            task = asyncio.create_task(session.volume.call('get_volume', {}, session.conversation.identity))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn('DEVICE_VOLUME_CANCELLED', str(logs.output))
        self.assertIsNone(session.volume.pending)

    async def test_validation_rejects_bool_numbers_unknown_fields_and_bad_operations(self):
        for operation, arguments in (
                ('set_volume', {'percent': True}), ('set_volume', {'percent': 101}),
                ('set_volume', {'percent': -1}), ('set_volume', {'percent': 5.5}),
                ('set_volume', {'percent': 50, 'persist': 1}), ('adjust_volume', {'delta': -101}),
                ('set_mute', {'muted': 1}), ('get_volume', {'extra': 1}),
                ('set_volume', {}), ('unknown', {}), ([], {})):
            with self.subTest(operation=operation, arguments=arguments), self.assertRaises(ValueError):
                volume_body(operation, arguments)

    async def test_retries_use_same_command_and_apply_delta_once(self):
        session = DeviceSession()
        session.drop_first = True
        result = await session.volume.call('adjust_volume', {'delta': -10}, session.conversation.identity)
        self.assertEqual(result['volume_percent'], 40)
        self.assertEqual(result['status'], 'confirmed')
        self.assertEqual(len(session.sent), 2)
        self.assertEqual(session.sent[0], session.sent[1])
        self.assertEqual(session.applied, 1)

    async def test_timeout_is_unknown_and_never_becomes_success(self):
        session = DeviceSession()
        session.ack = False
        result = await session.volume.call('adjust_volume', {'delta': -10}, session.conversation.identity)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['error'], 'volume_not_confirmed')
        self.assertFalse(result['retry_safe'])
        self.assertEqual(session.applied, 1)

    async def test_session_change_capability_and_busy_never_send_delayed_commands(self):
        session = DeviceSession()
        binding = session.conversation.identity
        session.volume_capable = False
        self.assertEqual((await session.volume.call('get_volume', {}, binding))['error'], 'volume_unsupported')
        session.volume_capable = True
        self.assertEqual((await session.volume.call('get_volume', {}, None))['error'], 'device_session_changed')
        await session.volume.lock.acquire()
        try:
            self.assertEqual((await session.volume.call('get_volume', {}, binding))['error'], 'device_busy')
        finally:
            session.volume.lock.release()
        self.assertEqual(session.sent, [])
        session.ack = False
        task = asyncio.create_task(session.volume.call('adjust_volume', {'delta': -5}, binding))
        await asyncio.sleep(0)
        session.conversation.identity = Identity(1, 2, 4, 0)
        result = await task
        self.assertEqual(result['error'], 'device_session_changed')
        self.assertEqual(len(session.sent), 1)

    async def test_ack_validation_save_failure_and_reset(self):
        session = DeviceSession()
        session.ack = False
        task = asyncio.create_task(session.volume.call('set_volume', {'percent': 30, 'persist': True}, session.conversation.identity))
        await asyncio.sleep(0)
        request = session.sent[-1]
        ack = Packet(Kind.VOLUME_ACK, 0, request.identity, request.sequence, bytes((2, 30, 30, 20)))
        for invalid in (replace(ack, sequence=99), replace(ack, flags=1),
                        replace(ack, identity=replace(ack.identity, server=99)),
                        replace(ack, body=b'bad'), replace(ack, body=bytes((0, 101, 30, 20))),
                        replace(ack, body=bytes((0, 30, 0, 20)))):
            self.assertFalse(session.volume.receive(invalid))
        self.assertTrue(session.volume.receive(ack))
        result = await task
        self.assertEqual(result['error'], 'volume_save_failed')
        self.assertTrue(result['applied'])
        task = asyncio.create_task(session.volume.call('get_volume', {}, session.conversation.identity))
        await asyncio.sleep(0)
        session.volume.reset()
        self.assertEqual((await task)['error'], 'device_session_changed')

    async def test_real_coordinator_negotiates_capability_and_confirms_only_matching_ack(self):
        session = ContinuousSession(replace(BridgeConfig(), device_mcp_enabled=True,
            save_input_wav=False, save_output_wav=False))
        sent = []
        session.transport = SimpleNamespace(sendto=lambda data, addr: sent.append(decode(data)))
        peer = ('127.0.0.1', 5000)
        try:
            session.datagram_received(encode(Packet(Kind.HELLO, CAP_VOLUME_CONTROL, Identity(1, 0, 0, 0), 1)), peer)
            self.assertTrue(sent[-1].flags & CAP_VOLUME_CONTROL)
            binding = Identity(1, session.server_nonce, 1, 0)
            session.conversation.open(binding)
            task = asyncio.create_task(session.volume.call('get_volume', {}, binding))
            await asyncio.sleep(0)
            request = sent[-1]
            ack = encode(Packet(Kind.VOLUME_ACK, 0, request.identity, request.sequence, bytes((0, 20, 20, 20))))
            session.datagram_received(ack, ('127.0.0.1', 5001))
            self.assertFalse(task.done())
            session.datagram_received(ack, peer)
            self.assertEqual((await task)['status'], 'confirmed')
        finally:
            session.connection_lost(None)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = DeviceSession()
        self.token = secrets.token_urlsafe(24)
        self.server = await DeviceGateway(self.session, self.token).start(0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.client = DeviceClient(self.port, self.token)

    async def asyncTearDown(self):
        await self.client.close()
        self.server.close()
        await self.server.wait_closed()

    async def test_persistent_client_mute_restore_and_session_binding(self):
        result = await self.client.call('set_volume', {'percent': 60})
        self.assertEqual(result['volume_percent'], 60)
        connection = self.client.websocket
        self.assertEqual((await self.client.call('set_mute', {'muted': True}))['volume_percent'], 0)
        self.assertEqual((await self.client.call('set_mute', {'muted': False}))['volume_percent'], 60)
        self.assertIs(self.client.websocket, connection)
        self.session.conversation.identity = Identity(1, 2, 4, 0)
        self.assertEqual((await self.client.call('adjust_volume', {'delta': -5}))['error'], 'device_session_changed')
        self.assertEqual(self.session.level, 60)

    async def test_gateway_dispatch_exception_is_safe_and_connection_survives(self):
        private = 'PRIVATE_token_payload'
        original = self.session.volume.call
        self.session.volume.call = AsyncMock(side_effect=RuntimeError(private))
        with self.assertLogs('piuu', level='INFO') as logs:
            result = await self.client.call('get_volume', {})
        self.assertEqual(result['error'], 'device_internal_error')
        self.assertNotIn(private, str(result) + str(logs.output))
        self.assertIn('DEVICE_WS_ERROR', str(logs.output))
        self.session.volume.call = original
        self.assertEqual((await self.client.call('get_volume', {}))['status'], 'confirmed')

    async def test_startup_bind_failure_is_logged_without_raw_exception(self):
        with self.assertLogs('piuu', level='INFO') as logs:
            with self.assertRaisesRegex(RuntimeError, '^Device gateway startup failed$'):
                await DeviceGateway(self.session, self.token).start(self.port)
        self.assertIn("phase='startup'", str(logs.output))
        self.assertNotIn(self.token, str(logs.output))

    async def test_authentication_origin_malformed_and_duplicate_requests(self):
        url = f'ws://127.0.0.1:{self.port}'
        with self.assertRaises(InvalidStatus):
            async with connect(url):
                pass
        with self.assertRaises(InvalidStatus):
            async with connect(url, origin='https://example.com', additional_headers={'Authorization': 'Bearer ' + self.token}):
                pass
        async with connect(url, additional_headers={'Authorization': 'Bearer ' + self.token}) as ws:
            await ws.send('[]')
            self.assertEqual(json.loads(await ws.recv())['result']['error'], 'invalid_volume_arguments')
            request = {'request_id': 'one', 'operation': 'adjust_volume', 'arguments': {'delta': -10}}
            await ws.send(json.dumps(request))
            self.assertEqual(json.loads(await ws.recv())['result']['volume_percent'], 40)
            await ws.send(json.dumps(request))
            self.assertEqual(json.loads(await ws.recv())['result']['error'], 'duplicate_device_request')
            self.assertEqual(self.session.applied, 1)

    async def test_real_stdio_discovery_schema_validation_and_websocket_round_trip(self):
        manager = ToolManager([device_server_spec(replace(BridgeConfig(),
            device_control_port=self.port, device_control_token=self.token))], timeout=10)
        try:
            await manager.start()
            self.assertEqual(set(manager.routes), {'device_get_volume', 'device_set_volume',
                'device_adjust_volume', 'device_set_mute'})
            result = await manager.call('device_adjust_volume', {'delta': -10})
            self.assertTrue(result['ok'])
            self.assertEqual(result['data']['volume_percent'], 40)
            result = await manager.call('device_set_volume', {'percent': 35, 'persist': True})
            self.assertEqual(result['data']['saved_percent'], 35)
            self.assertFalse((await manager.call('device_set_volume', {'percent': 101}))['ok'])
            self.assertFalse((await manager.call('device_set_mute', {'muted': 'true'}))['ok'])
            self.session.ack = False
            result = await manager.call('device_adjust_volume', {'delta': -5})
            self.assertFalse(result['ok'])
            self.assertEqual(result['error'], 'volume_not_confirmed')
        finally:
            await manager.close()


class GeminiDeviceTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_manager_spanish_instructions_and_mutation_tombstone(self):
        current = Identity(1, 2, 3, 1)
        gemini = GeminiLive(replace(BridgeConfig(), device_mcp_enabled=True), asyncio.Queue(),
                            lambda identity: identity == current, lambda *args: current)
        gemini.device_manager.declarations = [{'name': 'device_adjust_volume'}]
        gemini.device_manager.routes['device_adjust_volume'] = (None, 'adjust_volume')
        gemini.ws = AsyncMock()
        prompt = gemini.setup()['setup']['systemInstruction']['parts'][0]['text']
        self.assertIn('mean LOWER VOLUME', prompt)
        self.assertIn('ONE adjustment', prompt)
        self.assertIn('percentage points', prompt)
        self.assertIn('Without a number use -20 or +20', prompt)
        self.assertIn('HIGHER VOLUME, delta=+20', prompt)
        wait = asyncio.Event()
        async def device_call(*args):
            await wait.wait()
        gemini.device_manager.call = AsyncMock(side_effect=device_call)
        gemini.tool_manager.call = AsyncMock()
        call = {'id': 'vol-1', 'name': 'device_adjust_volume', 'args': {'delta': -10}}
        try:
            await gemini._tools([call])
            await asyncio.sleep(0)
            gemini._cancel_tools()
            await asyncio.sleep(0)
            await gemini._tools([call])
            self.assertEqual(gemini.device_manager.call.await_count, 1)
            gemini.tool_manager.call.assert_not_called()
            result = json.loads(gemini.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]
            self.assertEqual(result['response']['error'], 'volume_not_confirmed')
        finally:
            await gemini.close()


class ClientExceptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_failure_is_safe_logged_and_not_retried(self):
        private = 'PRIVATE_token_payload'
        client = DeviceClient(8766, private)
        with patch('bridge.mcp_servers.device.connect', AsyncMock(side_effect=OSError(private))) as connector:
            with self.assertLogs('piuu', level='INFO') as logs:
                result = await client.call('get_volume', {})
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error'], 'device_unavailable')
        self.assertEqual(connector.await_count, 1)
        self.assertNotIn(private, str(result) + str(logs.output))

    async def test_lost_or_malformed_response_is_unknown_and_never_replayed(self):
        for response in ('PRIVATE_token_payload', '[]', '{"request_id":"wrong","result":{}}'):
            client = DeviceClient(8766, 'PRIVATE_token_payload')
            socket = AsyncMock()
            socket.recv.return_value = response
            with patch('bridge.mcp_servers.device.connect', AsyncMock(return_value=socket)) as connector:
                with self.assertLogs('piuu', level='INFO') as logs:
                    result = await client.call('adjust_volume', {'delta': -20})
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['error'], 'volume_not_confirmed')
            self.assertEqual(socket.send.await_count, 1)
            self.assertEqual(connector.await_count, 1)
            self.assertIsNone(client.websocket)
            self.assertNotIn('PRIVATE_token_payload', str(result) + str(logs.output))

    async def test_tool_boundary_captures_unexpected_client_exception(self):
        private = 'PRIVATE_token_payload'
        # FastMCP normally configures root logging in its own subprocess. Keep
        # this in-process boundary test from changing other tests' logging.
        with patch('mcp.server.fastmcp.server.configure_logging'):
            server = create_server(SimpleNamespace(call=AsyncMock(side_effect=RuntimeError(private))))
        with self.assertLogs('piuu', level='INFO') as logs:
            result = await server.call_tool('get_volume', {})
        self.assertIn('device_internal_error', str(result))
        self.assertNotIn(private, str(result) + str(logs.output))
