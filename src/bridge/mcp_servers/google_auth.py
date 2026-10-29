"""One desktop OAuth consent for PIUU Gmail, Contacts, Calendar, Tasks and Drive."""
import argparse
import os
from pathlib import Path
from bridge.mcp_servers.gmail import SCOPES as GMAIL_SCOPES
from bridge.mcp_servers.calendar import SCOPES as CALENDAR_SCOPES
from bridge.mcp_servers.google_oauth import authorize

ROOT = Path(__file__).resolve().parents[3]
FUTURE_SCOPES = ['https://www.googleapis.com/auth/tasks',
                 'https://www.googleapis.com/auth/drive.file']
SCOPES = list(dict.fromkeys(GMAIL_SCOPES + CALENDAR_SCOPES + FUTURE_SCOPES))


def check_token(path):
    """Offline check of stored grants and refresh capability; never print tokens."""
    from google.oauth2.credentials import Credentials
    try:
        creds = Credentials.from_authorized_user_file(str(path))
    except (OSError, ValueError, KeyError, TypeError):
        print('No se pudo leer un token OAuth válido en la ruta indicada.')
        return False
    missing = [scope for scope in SCOPES if not creds.has_scopes([scope])]
    if missing:
        print('Faltan permisos:')
        for scope in missing:
            print('  ' + scope)
    if not creds.refresh_token:
        print('Falta refresh token; repetir --auth para permitir la renovación en el VPS.')
    if missing or not creds.refresh_token:
        return False
    print('Token preparado: los 8 permisos están presentes y hay refresh token.')
    print('Comprobación local; no verifica APIs habilitadas ni acceso en Google.')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--auth', action='store_true', help='Abrir consentimiento OAuth conjunto en la PC')
    action.add_argument('--check', action='store_true', help='Comprobar el token local sin consultar Google')
    action.add_argument('--list-scopes', action='store_true', help='Mostrar los permisos solicitados')
    parser.add_argument('--credentials', default=os.environ.get('PIUU_GOOGLE_CREDENTIALS', str(ROOT / 'credentials.json')))
    parser.add_argument('--token', default=os.environ.get('PIUU_GOOGLE_TOKEN', str(ROOT / '.secrets/google-token.json')))
    args = parser.parse_args(argv)
    if args.list_scopes:
        print('\n'.join(SCOPES))
        return 0
    path = Path(args.token).expanduser().resolve()
    if args.auth:
        client_file = Path(args.credentials).expanduser().resolve()
        if not client_file.is_file():
            print('No se encontró el cliente OAuth; indicar --credentials con la ruta de credentials.json.')
            return 1
        try:
            authorize(client_file, path, SCOPES, require_refresh_token=True)
        except Exception:
            # OAuth exceptions can contain authorization URLs or token payloads.
            print('No se completó la autorización conjunta. Revisar que el cliente sea Desktop app, '
                  'que la cuenta sea un usuario de prueba si corresponde y aceptar todos los permisos.')
            return 1
        print('Autorización conjunta guardada en ' + str(path))
    return 0 if check_token(path) else 1


if __name__ == '__main__':
    raise SystemExit(main())
