"""Offline Calendar API contracts and real multi-server MCP routing."""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge.config import BridgeConfig
from bridge.mcp_servers.calendar import CalendarAgenda, api_failure, token_path, SCOPES
from bridge.mcp_servers.google_oauth import load_credentials
from bridge.providers.gemini.live import GeminiLive
from bridge.tools.manager import ToolManager, calendar_server_spec, gmail_server_spec

START = '2026-10-30T10:00:00-03:00'
END = '2026-10-30T11:00:00-03:00'


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.service = MagicMock()
        self.factory = MagicMock(return_value=self.service)
        self.agenda = CalendarAgenda(self.factory, 'America/Argentina/Buenos_Aires')

    def test_calendar_list_pagination_and_roles(self):
        self.service.calendarList.return_value.list.return_value.execute.return_value = {
            'items': [{'id': 'work', 'summary': 'Trabajo', 'accessRole': 'writer'}], 'nextPageToken': 'next'}
        result = self.agenda.list_calendars(5, 'page')
        self.assertEqual(result['calendars'][0]['accessRole'], 'writer')
        self.assertEqual(result['next_page_token'], 'next')
        self.assertTrue(result['current_time'].endswith('-03:00'))
        self.service.calendarList.return_value.list.assert_called_once_with(maxResults=5, pageToken='page')

    def test_list_expands_recurrence_orders_by_time_and_preserves_all_day(self):
        item = {'id': 'event', 'start': {'date': '2026-10-30'}, 'end': {'date': '2026-10-31'},
                'recurringEventId': 'series', 'summary': 'Fiesta'}
        request = self.service.events.return_value.list
        request.return_value.execute.return_value = {'items': [item], 'nextPageToken': 'next'}
        result = self.agenda.list_events(START, END, 'Fiesta', 'work', 5, 'page')
        self.assertEqual(result['events'][0]['start'], item['start'])
        self.assertEqual(result['events'][0]['recurringEventId'], 'series')
        self.assertEqual(result['next_page_token'], 'next')
        request.assert_called_once_with(calendarId='work', timeMin=START, timeMax=END,
            q='Fiesta', singleEvents=True, orderBy='startTime', maxResults=5, pageToken='page',
            timeZone='America/Argentina/Buenos_Aires')

    def test_get_bounds_description(self):
        self.service.events.return_value.get.return_value.execute.return_value = {
            'id': 'abc', 'description': 'x' * 12001}
        result = self.agenda.get_event('abc')
        self.assertEqual(len(result['description']), 12000)
        self.assertTrue(result['description_truncated'])

    def test_freebusy_missing_and_partial_errors_never_claim_available(self):
        execute = self.service.freebusy.return_value.query.return_value.execute
        for entry in ({}, {'errors': [{'reason': 'notFound'}]}, {'errors': [{'reason': 'internalError'}], 'busy': []}):
            execute.return_value = {'calendars': {'primary': entry}}
            result = self.agenda.check_availability(START, END)
            self.assertEqual(result['error'], 'availability_unknown')
            self.assertNotIn('available', result)
        execute.return_value = {'calendars': {'actual@example.com': {'busy': []}}}
        self.assertTrue(self.agenda.check_availability(START, END)['available'])
        execute.return_value = {'calendars': {'primary': {'busy': [{'start': START, 'end': END}]}}}
        self.assertFalse(self.agenda.check_availability(START, END)['available'])

    def test_create_timed_and_all_day_no_invites_or_retries(self):
        insert = self.service.events.return_value.insert
        insert.return_value.execute.return_value = {'id': 'abcde', 'start': {'dateTime': START}}
        result = self.agenda.create_event('Reunión', START, END, description='Prueba ñ', event_id='abcde')
        self.assertEqual(result['status'], 'created')
        wire = insert.call_args.kwargs
        self.assertEqual(wire['body']['id'], 'abcde')
        self.assertEqual(wire['body']['start'], {'dateTime': START, 'timeZone': 'America/Argentina/Buenos_Aires'})
        self.assertEqual(wire['sendUpdates'], 'none')
        self.assertNotIn('attendees', wire['body'])
        insert.return_value.execute.assert_called_once_with(num_retries=0)
        self.agenda.create_event('Feriado', '2026-10-30', '2026-10-31', all_day=True)
        body = insert.call_args.kwargs['body']
        self.assertEqual(body['end'], {'date': '2026-10-31'})
        self.assertRegex(body['id'], '^[0-9a-f]{32}$')

    def test_invalid_dates_offsets_ids_and_content_do_not_access_api(self):
        for start, end, options in [
            ('2026-10-30T10:00:00', END, {}), (END, START, {}),
            (START, END, {'time_zone': 'UTC'}),
            ('2026-10-30', '2026-10-30', {'all_day': True}),
            ('2026-02-30', '2026-03-01', {'all_day': True}),
            (START, END, {'event_id': 'invalid-id!'}), (START, END, {'calendar_id': ''})]:
            with self.subTest(start=start, options=options), self.assertRaises(ValueError):
                self.agenda.create_event('Test', start, end, **options)
        with self.assertRaises(ValueError): self.agenda.create_event(' ', START, END)
        with self.assertRaises(ValueError): self.agenda.list_events(START, END, max_results=51)
        with self.assertRaises(ValueError): self.agenda.check_availability(END, START)
        self.factory.assert_not_called()

    def test_unknown_create_result_contains_reconciliation_id(self):
        insert = self.service.events.return_value.insert
        insert.return_value.execute.side_effect = TimeoutError('private event content')
        result = self.agenda.create_event('Test', START, END, event_id='abcde')
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['event_id'], 'abcde')
        self.assertFalse(result['retry_safe'])
        self.assertNotIn('private', str(result))
        insert.assert_called_once()

    def test_api_disabled_diagnostic_is_sanitized(self):
        error = Exception('private event content')
        error.resp = MagicMock(status=403)
        error.content = json.dumps({'error': {'errors': [{'reason': 'accessNotConfigured'}]}}).encode()
        self.assertEqual(api_failure(error)['error'], 'calendar_api_disabled')
        self.service.events.return_value.list.return_value.execute.side_effect = error
        self.assertEqual(self.agenda.list_events()['error'], 'calendar_api_disabled')

    def test_calendar_environment_is_separate_from_gmail(self):
        with patch.dict(os.environ, {'PIUU_GMAIL_TOKEN': '/secret/gmail', 'GEMINI_API_KEY': 'secret',
                                    'PIUU_CALENDAR_TOKEN': '/secret/calendar'}):
            spec = calendar_server_spec('UTC')
            self.assertEqual(spec.env['PIUU_CALENDAR_TOKEN'], '/secret/calendar')
            self.assertEqual(spec.env['PIUU_CALENDAR_TIME_ZONE'], 'UTC')
            self.assertNotIn('PIUU_GMAIL_TOKEN', spec.env)
            self.assertNotIn('GEMINI_API_KEY', spec.env)
            self.assertEqual(str(token_path()), '/secret/calendar')
        self.assertNotIn('gmail', ' '.join(SCOPES))


class GeminiCalendarTests(unittest.IsolatedAsyncioTestCase):
    async def test_calendar_only_and_combined_server_configuration(self):
        for gmail, calendar, expected in [(False, False, []), (True, False, ['gmail']),
                                         (False, True, ['calendar']), (True, True, ['gmail', 'calendar'])]:
            config = replace(BridgeConfig(), gmail_mcp_enabled=gmail, calendar_mcp_enabled=calendar)
            g = GeminiLive(config, asyncio.Queue(), lambda _: True, lambda *args: 1)
            self.assertEqual([s.name for s in g.tool_manager.servers], expected)
        g.tool_manager.declarations = [{'name': 'calendar_list_events', 'behavior': 'NON_BLOCKING'}]
        setup = g.setup()['setup']
        self.assertIn('Calendar time zone: America/Argentina/Buenos_Aires', setup['systemInstruction']['parts'][0]['text'])
        self.assertEqual(setup['tools'][0]['functionDeclarations'][-1]['behavior'], 'NON_BLOCKING')

    async def test_cancelled_create_is_deduplicated_and_result_survives_turn(self):
        g = GeminiLive(replace(BridgeConfig(), gmail_mcp_enabled=False, calendar_mcp_enabled=False),
                       asyncio.Queue(), lambda _: True, lambda *args: 1)
        g.ws = AsyncMock()
        g.tool_manager.routes['calendar_create_event'] = (None, 'create_event')
        gate = asyncio.Event()
        async def slow(*args):
            await gate.wait()
            return {'ok': True, 'data': {'status': 'created'}}
        g.tool_manager.call = AsyncMock(side_effect=slow)
        message = {'toolCall': {'functionCalls': [{'id': 'event-1', 'name': 'calendar_create_event',
                   'args': {'summary': 'Test', 'start': START, 'end': END}}]}}
        try:
            await g.handle_event(message)
            await asyncio.sleep(0)
            task = g.pending_tools['event-1']
            await g.handle_event({'toolCallCancellation': {'ids': ['event-1']}})
            await asyncio.gather(task, return_exceptions=True)
            await g.handle_event(message)
            g.tool_manager.call.assert_awaited_once()
            response = json.loads(g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]['response']
            self.assertEqual(response['error'], 'calendar_write_not_confirmed')
            message['toolCall']['functionCalls'][0]['id'] = 'event-2'
            await g.handle_event(message)
            await asyncio.sleep(0)
            await g.handle_event({'serverContent': {'turnComplete': True}})
            gate.set()
            await asyncio.gather(*g.pending_tools.values())
            response = json.loads(g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]['response']
            self.assertEqual(response['scheduling'], 'WHEN_IDLE')
            self.assertEqual(response['data']['status'], 'created')
        finally:
            await g.close()


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Requires MCP SDK')
class CalendarMCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_combined_discovery_and_missing_auth_without_google_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            gmail, calendar = gmail_server_spec(), calendar_server_spec()
            gmail.env['PIUU_GMAIL_TOKEN'] = str(Path(directory) / 'missing-gmail.json')
            calendar.env['PIUU_CALENDAR_TOKEN'] = str(Path(directory) / 'missing-calendar.json')
            manager = ToolManager([gmail, calendar], timeout=10)
            try:
                await manager.start()
                self.assertEqual(len(manager.declarations), 11)
                self.assertIn('calendar_create_event', manager.routes)
                self.assertIn('gmail_send_message', manager.routes)
                self.assertTrue(all(d['behavior'] == 'NON_BLOCKING' for d in manager.declarations))
                session, _ = manager.routes['calendar_create_event']
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                self.assertFalse(tools['create_event'].annotations.readOnlyHint)
                self.assertFalse(tools['create_event'].annotations.idempotentHint)
                self.assertTrue(tools['check_availability'].annotations.readOnlyHint)
                result = await manager.call('calendar_list_events', {})
                self.assertEqual(result['data']['status'], 'failed')
                # Calendar cannot borrow an existing Gmail token or open a browser.
                self.assertEqual((await manager.call('gmail_read_message', {'message_id': 'abc'}))['error'], 'gmail_auth_required')
            finally:
                await manager.close()
            self.assertFalse(manager.routes)
