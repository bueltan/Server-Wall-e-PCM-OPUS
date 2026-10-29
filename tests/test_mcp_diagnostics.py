"""Offline regression coverage for safe, actionable MCP failure diagnostics."""
import asyncio
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge.mcp_servers.calendar import CalendarAgenda, create_server as calendar_server
from bridge.mcp_servers.gmail import GmailMailbox, create_server as gmail_server
from bridge.mcp_servers.diagnostics import diagnostic_tool
from bridge.tools.manager import ToolManager, ServerSpec, calendar_server_spec

START = '2026-10-29T12:00:00-03:00'
END = '2026-10-29T12:05:00-03:00'
PRIVATE = 'PRIVATE_token_email_event_body'


class DomainDiagnosticsTests(unittest.TestCase):
    def test_calendar_validation_codes_never_access_google(self):
        factory = MagicMock()
        create = diagnostic_tool(CalendarAgenda(factory).create_event, 'calendar')
        for options, code in [
            ({'start': '2026-10-29T12:00:00'}, 'timestamp_requires_rfc3339_offset'),
            ({'start': '2026-02-30T12:00:00-03:00'}, 'invalid_event_datetime'),
            ({'time_zone': 'UTC'}, 'offset_does_not_match_time_zone'),
            ({'time_zone': PRIVATE}, 'invalid_time_zone'),
            ({'end': START}, 'end_must_follow_start'),
            ({'event_id': 'invalid-' + PRIVATE}, 'invalid_event_id'),
            ({'summary': ' '}, 'invalid_event_content'),
        ]:
            with self.subTest(code=code):
                result = create(**{'summary': 'Test', 'start': START, 'end': END, **options})
                self.assertEqual(result['error'], code)
                self.assertEqual(result['error_kind'], 'validation')
                self.assertNotIn(PRIVATE, str(result))
        factory.assert_not_called()

    def test_unexpected_exception_text_and_oauth_paths_are_not_returned(self):
        for exc, expected in [(ValueError(PRIVATE), 'gmail_unavailable'),
                              (RuntimeError('gmail_auth_required: ' + PRIVATE), 'gmail_auth_required'),
                              (OSError(PRIVATE), 'gmail_unavailable')]:
            def fail():
                raise exc
            result = diagnostic_tool(fail, 'gmail')()
            self.assertEqual(result['error'], expected)
            self.assertNotIn(PRIVATE, str(result))

    def test_google_read_error_status_and_reason_are_safe(self):
        for status, reason, code in [(403, 'accessNotConfigured', 'gmail_api_disabled'),
                                     (403, 'quotaExceeded', 'gmail_rate_limited'),
                                     (401, '', 'gmail_auth_required'),
                                     (403, '', 'gmail_access_denied'),
                                     (500, '', 'gmail_unavailable')]:
            exc = Exception(PRIVATE)
            exc.resp = SimpleNamespace(status=status)
            exc.content = json.dumps({'error': {'message': PRIVATE, 'errors': [{'reason': reason}]}})
            factory = MagicMock(side_effect=exc)
            read = diagnostic_tool(GmailMailbox(factory).read_message, 'gmail')
            result = read('abc')
            self.assertEqual(result['error'], code)
            self.assertEqual(result['http_status'], status)
            self.assertNotIn(PRIVATE, str(result))

    def test_send_distinguishes_google_rejection_from_uncertain_write(self):
        factory = MagicMock()
        send = factory.return_value.users.return_value.messages.return_value.send
        mailbox = GmailMailbox(factory)
        for status, expected in [(403, 'failed'), (503, 'unknown')]:
            exc = Exception(PRIVATE)
            exc.resp = SimpleNamespace(status=status)
            send.return_value.execute.side_effect = exc
            result = mailbox.send_message('test@example.com', 'Hi', 'Hello')
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['http_status'], status)
            self.assertFalse(result['retry_safe'])
            self.assertNotIn(PRIVATE, str(result))
        self.assertEqual(send.call_count, 2)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Requires MCP SDK')
class ManagerDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    @patch('mcp.server.fastmcp.server.configure_logging')
    async def test_all_owned_tools_keep_their_schema_and_return_safe_failures(self, configure):
        factory = MagicMock(side_effect=RuntimeError('gmail_auth_required: ' + PRIVATE))
        server = gmail_server(GmailMailbox(factory))
        tools = {tool.name: tool for tool in await server.list_tools()}
        self.assertEqual(tools['read_message'].inputSchema['required'], ['message_id'])
        result = await server.call_tool('read_message', {'message_id': 'abc'})
        self.assertIn('gmail_auth_required', str(result))
        self.assertNotIn(PRIVATE, str(result))
        server = calendar_server(CalendarAgenda(MagicMock()))
        result = await server.call_tool('create_event', {
            'summary': PRIVATE, 'start': START, 'end': END, 'time_zone': 'UTC'})
        self.assertIn('offset_does_not_match_time_zone', str(result))
        self.assertNotIn(PRIVATE, str(result))

    async def test_real_stdio_calendar_failure_is_actionable_and_correlated(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = calendar_server_spec()
            spec.env['PIUU_CALENDAR_TOKEN'] = str(Path(directory) / 'missing.json')
            manager = ToolManager([spec], timeout=10)
            with patch('bridge.tools.manager.event') as log:
                try:
                    await manager.start()
                    identity = SimpleNamespace(client=1, server=2, turn=3, response=4)
                    result = await manager.call('calendar_create_event', {
                        'summary': PRIVATE, 'start': START, 'end': END, 'time_zone': 'UTC'}, identity)
                    self.assertFalse(result['ok'])
                    self.assertEqual(result['error'], 'offset_does_not_match_time_zone')
                    result = await manager.call('calendar_create_event', {'summary': PRIVATE, 'start': [PRIVATE]})
                    self.assertEqual(result['error'], 'invalid_tool_arguments')
                    self.assertIn({'field': 'end', 'rule': 'required'}, result['validation'])
                    self.assertIn({'field': 'start', 'rule': 'type'}, result['validation'])
                    result = await manager.call('calendar_create_event', {
                        'summary': PRIVATE, 'start': START, 'end': END})
                    self.assertEqual(result['error'], 'calendar_auth_required')
                    self.assertFalse(result['ok'])
                    finished = [call.kwargs for call in log.call_args_list if call.args[0] == 'MCP_TOOL_FINISHED']
                    self.assertEqual([item['call_id'] for item in finished], [1, 2, 3])
                    self.assertTrue(all(item['duration_ms'] >= 0 for item in finished))
                    first = next(call for call in log.call_args_list if call.args[0] == 'MCP_TOOL_FINISHED')
                    self.assertIs(first.args[1], identity)
                    self.assertNotIn(PRIVATE, str(log.call_args_list))
                finally:
                    await manager.close()

    async def test_log_does_not_trust_remote_error_metadata(self):
        manager = ToolManager()
        with patch('bridge.tools.manager.event') as log:
            manager._finished('test_echo', 1, time.monotonic(), {'ok': False, 'data': {
                'error': PRIVATE, 'status': PRIVATE, 'error_kind': PRIVATE, 'http_status': PRIVATE,
                'retry_safe': PRIVATE, 'exception_type': PRIVATE}})
            self.assertNotIn(PRIVATE, str(log.call_args_list))
            self.assertEqual(log.call_args.kwargs['error'], 'mcp_tool_error')

    async def test_rejected_calls_and_full_queue_are_logged(self):
        manager = ToolManager()
        with patch('bridge.tools.manager.event') as log:
            result = await manager.call(PRIVATE, {})
            self.assertEqual(result['error'], 'unsupported_tool')
            self.assertNotIn(PRIVATE, str(log.call_args_list))
            manager.worker = asyncio.create_task(asyncio.Event().wait())
            manager.routes['test_tool'] = (None, 'tool')
            try:
                result = await manager.call('test_tool', PRIVATE)
                self.assertEqual(result['error'], 'invalid_tool_arguments')
                manager.queue = asyncio.Queue(1)
                manager.queue.put_nowait(None)
                result = await manager.call('test_tool', {})
                self.assertEqual(result['error'], 'tool_busy')
                self.assertEqual(log.call_args.kwargs['error'], 'tool_busy')
                self.assertNotIn(PRIVATE, str(log.call_args_list))
            finally:
                manager.queue.get_nowait()
                await manager.close()

    async def test_startup_failure_is_logged_without_command_or_credentials(self):
        manager = ToolManager([ServerSpec('test', '/missing/' + PRIVATE, (), {})], timeout=2)
        with patch('bridge.tools.manager.event') as log:
            with self.assertRaises(RuntimeError):
                await manager.start()
            self.assertTrue(any(call.args[0] == 'MCP_SERVER_ERROR' for call in log.call_args_list))
            self.assertNotIn(PRIVATE, str(log.call_args_list))

    async def test_remote_errors_transport_and_timeouts_are_logged(self):
        # Exercise the worker with a real MCP subprocess and a fake response at
        # the session boundary, without contacting Google or sending a write.
        with tempfile.TemporaryDirectory() as directory:
            spec = calendar_server_spec()
            spec.env['PIUU_CALENDAR_TOKEN'] = str(Path(directory) / 'missing.json')
            manager = ToolManager([spec], timeout=10)
            try:
                await manager.start()
                session, _ = manager.routes['calendar_list_events']
                for effect, code in [(TimeoutError(PRIVATE), 'tool_timeout'),
                                     (OSError(PRIVATE), 'tool_unavailable')]:
                    with patch.object(session, 'call_tool', AsyncMock(side_effect=effect)), \
                            patch('bridge.tools.manager.event') as log:
                        result = await manager.call('calendar_list_events', {})
                        self.assertEqual(result['error'], code)
                        self.assertNotIn(PRIVATE, str(log.call_args_list))
                remote = SimpleNamespace(isError=True, content=[SimpleNamespace(type='text', text=PRIVATE)])
                with patch.object(session, 'call_tool', AsyncMock(return_value=remote)), \
                        patch('bridge.tools.manager.event') as log:
                    result = await manager.call('calendar_list_events', {})
                    self.assertEqual(result['error'], 'mcp_tool_error')
                    self.assertNotIn(PRIVATE, str(result) + str(log.call_args_list))
            finally:
                await manager.close()
