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
from bridge.mcp_servers.gmail import GmailMailbox, plain_text, save_token, credentials, SCOPES, GMAIL_SCOPES, CONTACT_SCOPES, CONTACT_READ_SCOPES, CONTACT_FIELDS, contacts_service, contacts_write_service
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

    def test_contact_search_warms_people_cache_and_returns_addresses_and_phones(self):
        service = MagicMock()
        search = service.people.return_value.searchContacts
        search.return_value.execute.side_effect = [{}, {'results': [{'person': {
            'resourceName': 'people/c123', 'names': [{'displayName': 'Ana Pérez'}],
            'emailAddresses': [{'value': 'ana@example.com', 'type': 'work',
                                'formattedType': 'Trabajo', 'metadata': {'primary': True}},
                               {'value': 'ana.home@example.com', 'type': 'home'}],
            'phoneNumbers': [{'value': '+54 11 1234', 'type': 'mobile'}]}}]}]
        mailbox = GmailMailbox(contacts_service_factory=lambda: service)
        with patch('bridge.mcp_servers.gmail.time.sleep') as sleep:
            result = mailbox.search_contacts(' Ana ', 5)
        self.assertEqual(search.call_args_list[0].kwargs,
                         {'query': '', 'readMask': CONTACT_FIELDS, 'pageSize': 1})
        self.assertEqual(search.call_args_list[1].kwargs,
                         {'query': 'Ana', 'readMask': CONTACT_FIELDS, 'pageSize': 5})
        sleep.assert_called_once_with(5)
        contact = result['contacts'][0]
        self.assertEqual(contact['resource_name'], 'people/c123')
        self.assertEqual(contact['names'], ['Ana Pérez'])
        self.assertEqual(len(contact['emails']), 2)
        self.assertEqual(contact['emails'][0]['label'], 'Trabajo')
        self.assertTrue(contact['emails'][0]['primary'])
        self.assertEqual(contact['phones'][0]['number'], '+54 11 1234')
        self.assertFalse(result['possibly_truncated'])

    def test_contact_get_reads_saved_contact_fields_only(self):
        service = MagicMock()
        service.people.return_value.get.return_value.execute.return_value = {'resourceName': 'people/c123'}
        result = GmailMailbox(contacts_service_factory=lambda: service).get_contact('people/c123')
        service.people.return_value.get.assert_called_once_with(
            resourceName='people/c123', personFields=CONTACT_FIELDS, sources=['READ_SOURCE_TYPE_CONTACT'])
        self.assertEqual(result, {'resource_name': 'people/c123', 'names': [], 'emails': [], 'phones': []})

    def test_invalid_contact_arguments_do_not_access_people(self):
        factory = MagicMock()
        mailbox = GmailMailbox(contacts_service_factory=factory)
        for query, limit in [('', 10), (' ', 10), ('x'*257, 10), ('Ana', 31), ('Ana', 0)]:
            with self.assertRaises(ValueError): mailbox.search_contacts(query, limit)
        for resource in ('', 'otherContacts/c123', 'people/', 'people/c123?personFields=biographies'):
            with self.assertRaises(ValueError): mailbox.get_contact(resource)
        factory.assert_not_called()

    def test_contact_failures_are_actionable_without_private_payloads(self):
        error = RuntimeError('gmail_auth_required: new consent needed')
        factory = MagicMock(side_effect=error)
        mailbox = GmailMailbox(contacts_service_factory=factory)
        self.assertEqual(mailbox.search_contacts('Ana')['error'], 'contacts_auth_required')
        error = Exception('private contact data')
        error.resp = MagicMock(status=403)
        error.content = json.dumps({'error': {'errors': [{'reason': 'accessNotConfigured'}]}}).encode()
        factory.side_effect = error
        result = mailbox.get_contact('people/c123')
        self.assertEqual(result['error'], 'people_api_disabled')
        self.assertNotIn('private', str(result))

    def test_contact_search_reports_empty_and_possibly_truncated_results(self):
        service = MagicMock()
        search = service.people.return_value.searchContacts.return_value.execute
        mailbox = GmailMailbox(contacts_service_factory=lambda: service)
        with patch('bridge.mcp_servers.gmail.time.sleep'):
            search.side_effect = [{}, {}]
            result = mailbox.search_contacts('nobody')
            self.assertEqual(result['contacts'], [])
            self.assertFalse(result['possibly_truncated'])
            search.side_effect = [{}, {'results': [{'person': {'resourceName': 'people/c123'}}]}]
            self.assertTrue(mailbox.search_contacts('Ana', 1)['possibly_truncated'])

    @unittest.skipUnless(importlib.util.find_spec('google') and importlib.util.find_spec('google.oauth2'), 'Requires Google auth')
    def test_existing_gmail_token_still_works_but_contacts_need_new_scope(self):
        import tempfile
        self.assertEqual(SCOPES, GMAIL_SCOPES + CONTACT_SCOPES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            path.write_text(json.dumps({'token': 'test-token', 'refresh_token': 'test-refresh',
                'token_uri': 'https://oauth2.googleapis.com/token', 'client_id': 'test',
                'client_secret': 'test', 'scopes': GMAIL_SCOPES, 'expiry': '2030-01-01T00:00:00Z'}))
            with patch.dict('os.environ', {'PIUU_GMAIL_TOKEN': str(path)}):
                self.assertTrue(credentials().has_scopes(GMAIL_SCOPES))
                with self.assertRaisesRegex(RuntimeError, 'gmail_auth_required'):
                    contacts_service()
            self.assertEqual(json.loads(path.read_text())['scopes'], GMAIL_SCOPES)

    def test_create_contact_sends_single_person_without_retries_or_email(self):
        service = MagicMock()
        create = service.people.return_value.createContact
        create.return_value.execute.return_value = {
            'resourceName': 'people/c789', 'names': [{'displayName': 'Ana Pérez'}],
            'emailAddresses': [{'value': 'ana@example.com'}], 'phoneNumbers': [{'value': '+54 11 1234'}]}
        mailbox = GmailMailbox(contacts_write_service_factory=lambda: service)
        result = mailbox.create_contact(' Ana ', ' Pérez ', 'ana@example.com', '+54 11 1234')
        self.assertEqual(result['status'], 'created')
        self.assertEqual(result['contact']['resource_name'], 'people/c789')
        create.assert_called_once_with(body={'names': [{'givenName': 'Ana', 'familyName': 'Pérez'}],
            'emailAddresses': [{'value': 'ana@example.com'}], 'phoneNumbers': [{'value': '+54 11 1234'}]},
            personFields=CONTACT_FIELDS, sources=['READ_SOURCE_TYPE_CONTACT'])
        create.return_value.execute.assert_called_once_with(num_retries=0)
        service.users.assert_not_called()

    def test_create_contact_accepts_phone_only_or_email_only(self):
        service = MagicMock()
        service.people.return_value.createContact.return_value.execute.return_value = {'resourceName': 'people/c789'}
        mailbox = GmailMailbox(contacts_write_service_factory=lambda: service)
        mailbox.create_contact('Ana', phone='12345')
        self.assertNotIn('emailAddresses', service.people.return_value.createContact.call_args.kwargs['body'])
        mailbox.create_contact('Ana', email='ana@example.com')
        self.assertNotIn('phoneNumbers', service.people.return_value.createContact.call_args.kwargs['body'])

    def test_invalid_contact_creation_does_not_call_google(self):
        factory = MagicMock()
        mailbox = GmailMailbox(contacts_write_service_factory=factory)
        for options in [{'given_name': ''}, {'given_name': 'Ana'},
                        {'given_name': 'Ana', 'email': 'invalid'},
                        {'given_name': 'Ana', 'email': 'a@example.com,b@example.com'},
                        {'given_name': 'Ana', 'email': 'a@example.com\nBcc: b@example.com'},
                        {'given_name': 'Ana', 'phone': 'not a number'},
                        {'given_name': 'Ana\nInjected', 'phone': '12345'}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                mailbox.create_contact(**options)
        factory.assert_not_called()

    def test_uncertain_contact_creation_is_not_retried_or_reported_as_saved(self):
        service = MagicMock()
        create = service.people.return_value.createContact
        create.return_value.execute.side_effect = TimeoutError('private contact data')
        mailbox = GmailMailbox(contacts_write_service_factory=lambda: service)
        result = mailbox.create_contact('Ana', email='ana@example.com')
        self.assertEqual(result, {'status': 'unknown', 'error': 'contact_create_not_confirmed', 'retry_safe': False})
        create.assert_called_once()
        self.assertNotIn('private', str(result))

    @unittest.skipUnless(importlib.util.find_spec('google') and importlib.util.find_spec('google.oauth2'), 'Requires Google auth')
    def test_readonly_contacts_grants_keep_lookup_working_but_not_creation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            path.write_text(json.dumps({'token': 'test-token', 'refresh_token': 'test-refresh',
                'token_uri': 'https://oauth2.googleapis.com/token', 'client_id': 'test',
                'client_secret': 'test', 'scopes': GMAIL_SCOPES + CONTACT_READ_SCOPES,
                'expiry': '2030-01-01T00:00:00Z'}))
            with patch.dict('os.environ', {'PIUU_GMAIL_TOKEN': str(path)}), \
                 patch('bridge.mcp_servers.gmail.build_service') as build:
                contacts_service()
                self.assertTrue(build.call_args.args[2].has_scopes(CONTACT_READ_SCOPES))
                result = GmailMailbox().create_contact('Ana', phone='12345')
                self.assertEqual(result['error'], 'contacts_write_auth_required')
                self.assertEqual(build.call_count, 1)
                # Full Contacts permission also supports lookup without readonly grants.
                token = json.loads(path.read_text())
                token['scopes'] = GMAIL_SCOPES + CONTACT_SCOPES
                path.write_text(json.dumps(token))
                contacts_service()
                self.assertTrue(build.call_args.args[2].has_scopes(CONTACT_SCOPES))

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

    async def test_cancelled_contact_creation_is_not_executed_again(self):
        self.g.tool_manager.routes['gmail_create_contact'] = (None, 'create_contact')
        message = {'toolCall': {'functionCalls': [{'id': 'create-1', 'name': 'gmail_create_contact',
                   'args': {'given_name': 'Ana', 'email': 'ana@example.com'}}]}}
        await self.g.handle_event(message)
        await asyncio.sleep(0)
        task = self.g.pending_tools['create-1']
        await self.g.handle_event({'toolCallCancellation': {'ids': ['create-1']}})
        await asyncio.gather(task, return_exceptions=True)
        await self.g.handle_event(message)
        self.g.tool_manager.call.assert_awaited_once()
        result = json.loads(self.g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]['response']
        self.assertEqual(result['error'], 'contact_create_not_confirmed')
        self.assertFalse(result['retry_safe'])

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
                self.assertEqual(set(manager.routes), {'gmail_search_messages', 'gmail_read_message', 'gmail_send_message', 'gmail_search_contacts', 'gmail_get_contact', 'gmail_create_contact'})
                session, _ = manager.routes['gmail_send_message']
                annotations = {tool.name: tool.annotations for tool in (await session.list_tools()).tools}
                self.assertFalse(annotations['send_message'].readOnlyHint)
                self.assertFalse(annotations['send_message'].idempotentHint)
                self.assertTrue(annotations['read_message'].readOnlyHint)
                self.assertTrue(annotations['search_contacts'].readOnlyHint)
                self.assertTrue(annotations['get_contact'].idempotentHint)
                self.assertFalse(annotations['create_contact'].readOnlyHint)
                self.assertFalse(annotations['create_contact'].idempotentHint)
                self.assertTrue(all(d['behavior'] == 'NON_BLOCKING' for d in manager.declarations))
                contact_result = await manager.call('gmail_get_contact', {'resource_name': 'people/c123'})
                self.assertEqual(contact_result['data']['error'], 'contacts_auth_required')
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
