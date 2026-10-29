"""Offline mailbox, Gemini scheduling and real MCP stdio contract checks."""
import asyncio
import base64
import importlib.util
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge.config import BridgeConfig
from bridge.mcp_servers.gmail import GmailMailbox, plain_text, save_token, credentials, SCOPES
from bridge.tools.manager import ToolManager, ServerSpec, gmail_server_spec
from bridge.providers.gemini.live import GeminiLive


class MailboxTests(unittest.TestCase):
    def test_search_pagination_metadata_and_bounds(self):
        service = MagicMock()
        api = service.users.return_value.messages.return_value
        api.list.return_value.execute.return_value = {'messages': [{'id': 'abc'}], 'nextPageToken': 'next'}
        api.get.return_value.execute.return_value = {'id': 'abc', 'payload': {'headers': [
            {'name': 'Subject', 'value': 'Hola'}]}}
        mailbox = GmailMailbox(lambda: service)
        result = mailbox.search_messages('is:unread', 3, 'page')
        self.assertEqual(result['messages'][0]['subject'], 'Hola')
        self.assertEqual(result['next_page_token'], 'next')
        api.list.assert_called_once_with(userId='me', q='is:unread', maxResults=3, pageToken='page')
        with self.assertRaises(ValueError): mailbox.search_messages(max_results=21)

    def test_multipart_charset_attachment_filter_and_truncation(self):
        part = {'mimeType': 'multipart/mixed', 'parts': [
            {'mimeType': 'text/html', 'body': {'data': 'ignored'}},
            {'mimeType': 'text/plain', 'filename': 'secret.txt', 'body': {'data': 'ignored'}},
            {'mimeType': 'text/plain', 'headers': [{'name': 'Content-Type',
                'value': 'text/plain; charset=iso-8859-1'}],
                'body': {'data': base64.urlsafe_b64encode('café'.encode('latin1')).decode().rstrip('=')}}]}
        self.assertEqual(plain_text(part), 'café')
        service = MagicMock()
        service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            'id': 'abc', 'payload': {'mimeType': 'text/plain', 'body': {
                'data': base64.urlsafe_b64encode(b'x' * 12001).decode()}}}
        result = GmailMailbox(lambda: service).read_message('abc')
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['body']), 12000)

    def test_send_utf8_mime_and_no_automatic_retries(self):
        from email import policy
        from email.parser import BytesParser
        service = MagicMock()
        send = service.users.return_value.messages.return_value.send
        send.return_value.execute.return_value = {'id': 'sent-123', 'threadId': 'thread-1'}
        result = GmailMailbox(lambda: service).send_message('test@example.com', 'Prueba ñ', 'Hola, café 🐦')
        self.assertEqual(result, {'status': 'sent', 'id': 'sent-123', 'thread_id': 'thread-1'})
        send.return_value.execute.assert_called_once_with(num_retries=0)
        wire = send.call_args.kwargs
        self.assertEqual(wire['userId'], 'me')
        message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(wire['body']['raw']))
        self.assertEqual(message['To'], 'test@example.com')
        self.assertEqual(message['Subject'], 'Prueba ñ')
        self.assertEqual(message.get_content().strip(), 'Hola, café 🐦')

    def test_invalid_send_never_calls_api_and_unknown_result_is_not_retried(self):
        factory = MagicMock()
        mailbox = GmailMailbox(factory)
        for recipient, subject, body in [
                ('invalid', 'Hi', 'Hello'), ('a@example.com,b@example.com', 'Hi', 'Hello'),
                ('a@example.com\r\nBcc: b@example.com', 'Hi', 'Hello'),
                ('a@example.com', 'Hi\nBcc: b@example.com', 'Hello'),
                ('a@example.com', '', 'Hello'), ('a@example.com', 'Hi', '')]:
            with self.subTest(recipient=recipient), self.assertRaises(ValueError):
                mailbox.send_message(recipient, subject, body)
        factory.assert_not_called()
        send = factory.return_value.users.return_value.messages.return_value.send
        send.return_value.execute.side_effect = TimeoutError('private email data')
        result = mailbox.send_message('a@example.com', 'Hi', 'Hello')
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['retry_safe'])
        self.assertNotIn('private', str(result))
        send.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('google') and importlib.util.find_spec('google.oauth2'), 'Requires Google auth')
    def test_old_readonly_token_requires_new_consent(self):
        import tempfile
        self.assertIn('https://www.googleapis.com/auth/gmail.send', SCOPES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            path.write_text(json.dumps({'token': 'test-token', 'refresh_token': 'test-refresh',
                'token_uri': 'https://oauth2.googleapis.com/token', 'client_id': 'test',
                'client_secret': 'test', 'scopes': [SCOPES[0]]}))
            with patch.dict('os.environ', {'PIUU_GMAIL_TOKEN': str(path)}):
                with self.assertRaisesRegex(RuntimeError, 'gmail_auth_required'):
                    credentials()

    def test_token_permissions_and_isolated_subprocess_environment(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / 'token.json'
            creds = MagicMock()
            creds.to_json.return_value = '{}'
            save_token(creds, token)
            self.assertEqual(token.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('GEMINI_API_KEY', gmail_server_spec().env)
        self.assertNotIn('XAI_API_KEY', gmail_server_spec().env)


class GeminiMCPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.g = GeminiLive(replace(BridgeConfig(), gmail_mcp_enabled=False), asyncio.Queue(),
                            lambda token: token == 1, lambda *args: 1)
        self.g.ws = AsyncMock()
        self.g.tool_manager.routes['gmail_read_message'] = (None, 'read_message')
        self.finished = asyncio.Event()
        async def slow(*args):
            await self.finished.wait()
            return {'ok': True, 'data': {'body': 'hola'}}
        self.g.tool_manager.call = AsyncMock(side_effect=slow)

    async def asyncTearDown(self):
        await self.g.close()

    async def call(self, cid='gmail-1'):
        await self.g.handle_event({'toolCall': {'functionCalls': [{
            'id': cid, 'name': 'gmail_read_message', 'args': {'message_id': 'abc'}}]}})
        await asyncio.sleep(0)

    async def test_nonblocking_survives_turn_complete_and_new_speech(self):
        await self.call()
        await self.call()
        self.g.tool_manager.call.assert_awaited_once()
        self.g.ws.send.assert_not_called()
        await self.g.handle_event({'serverContent': {'turnComplete': True}})
        await self.g.handle_event({'voiceActivity': {'type': 'ACTIVITY_START'}})
        self.finished.set()
        await asyncio.gather(*self.g.pending_tools.values())
        result = json.loads(self.g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]
        self.assertEqual(result['response']['scheduling'], 'WHEN_IDLE')
        self.assertTrue(result['response']['ok'])
        await self.call()
        self.g.tool_manager.call.assert_awaited_once()

    async def test_explicit_cancellation_drops_late_result(self):
        await self.call()
        task = self.g.pending_tools['gmail-1']
        await self.g.handle_event({'toolCallCancellation': {'ids': ['gmail-1']}})
        self.finished.set()
        await asyncio.gather(task, return_exceptions=True)
        self.g.ws.send.assert_not_called()
        self.assertFalse(self.g.pending_tools)

    async def test_cancelled_send_id_is_not_executed_again(self):
        self.g.tool_manager.routes['gmail_send_message'] = (None, 'send_message')
        message = {'toolCall': {'functionCalls': [{
            'id': 'send-1', 'name': 'gmail_send_message',
            'args': {'to': 'test@example.com', 'subject': 'Hi', 'body': 'Hello'}}]}}
        await self.g.handle_event(message)
        await asyncio.sleep(0)
        task = self.g.pending_tools['send-1']
        await self.g.handle_event({'toolCallCancellation': {'ids': ['send-1']}})
        await asyncio.gather(task, return_exceptions=True)
        await self.g.handle_event(message)
        self.g.tool_manager.call.assert_awaited_once()
        result = json.loads(self.g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]
        self.assertEqual(result['response']['error'], 'send_not_confirmed')
        self.assertFalse(result['response']['retry_safe'])

    async def test_setup_preserves_expression_blocking_and_gmail_nonblocking(self):
        self.g.tool_manager.declarations.append({'name': 'gmail_read_message', 'behavior': 'NON_BLOCKING'})
        declarations = self.g.setup()['setup']['tools'][0]['functionDeclarations']
        self.assertEqual([d['behavior'] for d in declarations], ['BLOCKING', 'NON_BLOCKING'])
        self.g.config = replace(self.g.config, expression_tools_enabled=False)
        self.assertEqual(len(self.g.setup()['setup']['tools'][0]['functionDeclarations']), 1)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Install requirements-mcp.txt for stdio integration')
class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_gmail_discovery_validation_and_auth_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            spec = gmail_server_spec()
            spec.env['PIUU_GMAIL_TOKEN'] = str(Path(directory) / 'missing.json')
            manager = ToolManager([spec], timeout=10)
            try:
                await manager.start()
                self.assertEqual(set(manager.routes), {'gmail_search_messages', 'gmail_read_message', 'gmail_send_message'})
                session, _ = manager.routes['gmail_send_message']
                annotations = {tool.name: tool.annotations for tool in (await session.list_tools()).tools}
                self.assertFalse(annotations['send_message'].readOnlyHint)
                self.assertFalse(annotations['send_message'].idempotentHint)
                self.assertTrue(annotations['read_message'].readOnlyHint)
                self.assertTrue(all(d['behavior'] == 'NON_BLOCKING' for d in manager.declarations))
                result = await manager.call('gmail_read_message', {'message_id': 'abc'})
                self.assertEqual(result, {'ok': False, 'error': 'mcp_tool_error'})
                self.assertEqual((await manager.call('gmail_read_message', {}))['error'], 'mcp_tool_error')
                self.assertEqual((await manager.call('gmail_read_message', 'invalid'))['error'], 'invalid_tool_arguments')
            finally:
                await manager.close()
            self.assertIsNone(manager.worker)
            self.assertFalse(manager.routes)

    async def test_stdio_success_timeout_and_queue_cancellation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'server.py'
            script.write_text('''import asyncio
from mcp.server.fastmcp import FastMCP
server = FastMCP('test')
@server.tool()
async def echo(value: str) -> dict:
    if value == 'slow':
        await asyncio.sleep(10)
    return {'value': value}
server.run(transport='stdio')
''')
            manager = ToolManager([ServerSpec('test', sys.executable, (str(script),), {})], timeout=10)
            try:
                await manager.start()
                self.assertEqual((await manager.call('test_echo', {'value': 'hello'}))['data'], {'value': 'hello'})
                manager.timeout = .05
                self.assertEqual((await manager.call('test_echo', {'value': 'slow'}))['error'], 'tool_timeout')
            finally:
                await manager.close()
