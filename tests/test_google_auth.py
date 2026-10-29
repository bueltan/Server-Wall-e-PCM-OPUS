"""Combined consent contract, offline checks and reuse by existing MCPs."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from bridge.mcp_servers import google_auth, google_oauth, gmail, calendar


def token(scopes, refresh='test-refresh'):
    return {'token': 'private-access-token', 'refresh_token': refresh,
            'token_uri': 'https://oauth2.googleapis.com/token', 'client_id': 'client',
            'client_secret': 'private-client-secret', 'scopes': scopes,
            'expiry': '2030-01-01T00:00:00Z'}


class CombinedAuthTests(unittest.TestCase):
    def test_exact_selected_scope_bundle_and_no_redundant_contact_readonly(self):
        self.assertEqual(len(google_auth.SCOPES), 8)
        self.assertEqual(set(google_auth.SCOPES), set(gmail.SCOPES + calendar.SCOPES +
            ['https://www.googleapis.com/auth/tasks', 'https://www.googleapis.com/auth/drive.file']))
        self.assertNotIn('https://www.googleapis.com/auth/contacts.readonly', google_auth.SCOPES)
        self.assertNotIn('https://www.googleapis.com/auth/drive', google_auth.SCOPES)
        self.assertNotIn('https://www.googleapis.com/auth/gmail.modify', google_auth.SCOPES)

    def test_combined_token_works_for_existing_gmail_contacts_and_calendar_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'google-token.json'
            path.write_text(json.dumps(token(google_auth.SCOPES)))
            with patch.dict('os.environ', {'PIUU_GMAIL_TOKEN': str(path), 'PIUU_CALENDAR_TOKEN': str(path)}), \
                 patch('bridge.mcp_servers.gmail.build_service') as build_gmail, \
                 patch('bridge.mcp_servers.calendar.build_service') as build_calendar:
                gmail.service()
                gmail.contacts_service()
                gmail.contacts_write_service()
                calendar.service()
                self.assertEqual(build_gmail.call_count, 3)
                build_calendar.assert_called_once()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertTrue(google_auth.check_token(path))
            self.assertNotIn('private', output.getvalue())

    def test_check_reports_missing_scopes_and_refresh_without_network_or_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            path.write_text(json.dumps(token(gmail.GMAIL_SCOPES, refresh='')))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(google_auth.check_token(path))
            self.assertIn('https://www.googleapis.com/auth/tasks', output.getvalue())
            self.assertIn('Falta refresh token', output.getvalue())
            self.assertNotIn('private', output.getvalue())
            path.write_text('invalid-json')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(google_auth.main(['--check', '--token', str(path)]), 1)

    def test_auth_invokes_one_consent_and_keeps_individual_tokens_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client, target = root / 'credentials.json', root / 'google-token.json'
            client.write_text('{}')
            old = root / 'gmail-token.json'
            old.write_text('old-token')
            def authorize(client_file, path, scopes, require_refresh_token=False):
                self.assertTrue(require_refresh_token)
                path.write_text(json.dumps(token(scopes)))
            with patch('bridge.mcp_servers.google_auth.authorize', side_effect=authorize) as auth, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(google_auth.main(['--auth', '--credentials', str(client), '--token', str(target)]), 0)
            auth.assert_called_once_with(client, target, google_auth.SCOPES, require_refresh_token=True)
            self.assertEqual(old.read_text(), 'old-token')

    def test_partial_consent_or_missing_refresh_does_not_replace_existing_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'token.json'
            path.write_text('preserved')
            creds = MagicMock()
            creds.has_scopes.return_value = True
            creds.granted_scopes = gmail.GMAIL_SCOPES  # Requested scopes != actual consent.
            creds.refresh_token = 'refresh'
            flow = MagicMock()
            flow.run_local_server.return_value = creds
            with patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file', return_value=flow):
                with self.assertRaisesRegex(RuntimeError, 'google_auth_missing_scopes'):
                    google_oauth.authorize('client.json', path, google_auth.SCOPES, require_refresh_token=True)
                creds.granted_scopes = google_auth.SCOPES
                creds.refresh_token = None
                with self.assertRaisesRegex(RuntimeError, 'google_auth_refresh_token_required'):
                    google_oauth.authorize('client.json', path, google_auth.SCOPES, require_refresh_token=True)
            self.assertEqual(path.read_text(), 'preserved')

    def test_list_scopes_and_auth_error_never_print_sensitive_exception(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(google_auth.main(['--list-scopes']), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 8)
        with tempfile.TemporaryDirectory() as directory:
            client = Path(directory) / 'credentials.json'
            client.write_text('{}')
            with patch('bridge.mcp_servers.google_auth.authorize', side_effect=RuntimeError('private-token')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(google_auth.main(['--auth', '--credentials', str(client)]), 1)
            self.assertNotIn('private-token', output.getvalue())
