"""Shared local OAuth helpers for PIUU Google MCP servers."""
import os
import tempfile


def save_token(credentials, path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.google-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(credentials.to_json())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_credentials(path, scopes, server):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    error = f'{server}_auth_required: run python -m bridge.mcp_servers.{server} --auth'
    if not path.exists():
        raise RuntimeError(error)
    # Preserve stored grants instead of overwriting them with requested scopes.
    creds = Credentials.from_authorized_user_file(str(path))
    if not creds.has_scopes(scopes):
        raise RuntimeError(error)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_token(creds, path)
    if not creds.valid or not creds.has_scopes(scopes):
        raise RuntimeError(error)
    return creds


def build_service(api, version, creds):
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build
    return build(api, version, http=AuthorizedHttp(creds, http=httplib2.Http(timeout=10)),
                 cache_discovery=False)


def authorize(client_file, path, scopes, require_refresh_token=False):
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), scopes)
    creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
    granted = creds.granted_scopes
    if not creds.has_scopes(scopes) or (granted is not None and not set(scopes).issubset(granted)):
        raise RuntimeError('google_auth_missing_scopes')
    if require_refresh_token and not creds.refresh_token:
        raise RuntimeError('google_auth_refresh_token_required')
    save_token(creds, path)
