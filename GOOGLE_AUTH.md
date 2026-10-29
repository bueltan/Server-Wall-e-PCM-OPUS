# Autorización conjunta de Google para PIUU

Esta guía reúne en un consentimiento los permisos acordados para Gmail,
Contactos, Calendar, Tasks y archivos de Drive, Docs y Sheets. Gmail y Calendar
pueden consumir el mismo token. Tasks, Drive, Docs y Sheets todavía no tienen
herramientas implementadas: autorizar ahora prepara esas integraciones.

## 1. Habilitar las APIs

En [Google Cloud Console](https://console.cloud.google.com/apis/library), elegir
el mismo proyecto al que pertenece el `credentials.json` existente. Habilitar:

- Gmail API (ya habilitada en la instalación actual).
- People API, para Contactos.
- Google Calendar API.
- Google Tasks API.
- Google Drive API.
- Google Docs API.
- Google Sheets API.

Habilitar una API permite usarla en el proyecto; el consentimiento OAuth concede
acceso a tu cuenta. Ambos pasos son necesarios. No hace falta generar otro cliente
si el actual es de tipo **Desktop app / Aplicación de escritorio**.

En **Google Auth Platform → Data Access / Acceso a los datos**, agregar los ocho
scopes siguientes. Si la aplicación está en pruebas, comprobar en **Audience /
Público** que tu cuenta figure entre los usuarios de prueba.

```text
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.send
https://www.googleapis.com/auth/contacts
https://www.googleapis.com/auth/calendar.events
https://www.googleapis.com/auth/calendar.calendarlist.readonly
https://www.googleapis.com/auth/calendar.events.freebusy
https://www.googleapis.com/auth/tasks
https://www.googleapis.com/auth/drive.file
```

| Servicio | Alcance autorizado | Herramientas actuales de PIUU |
| --- | --- | --- |
| Gmail | Leer y enviar correo | Buscar, leer y enviar texto plano. |
| Contactos | Leer y gestionar contactos guardados | Buscar, consultar y crear. |
| Calendar | Leer calendarios, consultar disponibilidad y gestionar eventos | Consultar agenda/disponibilidad y crear eventos. |
| Tasks | Leer y gestionar listas y tareas | Pendiente. |
| Drive, Docs y Sheets | Archivos creados o autorizados para la aplicación | Pendiente. |

Los permisos de Contactos y Calendar también permiten operaciones que PIUU aún
no expone, como actualizar o eliminar. El permiso `drive.file` limita el acceso a
archivos creados por la aplicación o seleccionados/autorizados para ella, por
ejemplo mediante Google Picker; conocer un ID o pegar un enlace no concede acceso
por sí solo. No abre automáticamente todos tus documentos. Docs y Sheets admiten
este scope para los archivos autorizados, sin añadir permisos globales.

Referencias oficiales: [Drive](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[Tasks](https://developers.google.com/workspace/tasks/auth),
[Calendar](https://developers.google.com/workspace/calendar/api/auth),
[Docs](https://developers.google.com/workspace/docs/api/reference/rest/v1/documents/batchUpdate)
y [Sheets](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/batchUpdate).

## 2. Autorizar desde tu PC

Con el código actualizado y `credentials.json` en la raíz del proyecto:

```bash
cd /home/denis/Documents/PlatformIO/Projects/PIUU_SERVER
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.google_auth --auth --credentials "$PWD/credentials.json" --token "$PWD/.secrets/google-token.json"
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.google_auth --check --token "$PWD/.secrets/google-token.json"
```

Se abre el navegador en tu PC. Elegir la cuenta que usará PIUU y aceptar todos
los permisos. Google puede mostrar otra vez los de Gmail que ya habías aceptado.
El resultado se guarda en `.secrets/google-token.json`, con permisos privados,
sin sustituir los tokens anteriores de Gmail o Calendar. Si no se conceden todos
los permisos o falta el refresh token, no se reemplaza el archivo de destino.

`--check` verifica localmente los ocho scopes y la presencia de refresh token;
no consulta Google, no confirma que las APIs estén habilitadas ni que Google siga
aceptando el token. No imprime las credenciales. Para listar los scopes:

```bash
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.google_auth --list-scopes
```

El helper admite `PIUU_GOOGLE_CREDENTIALS` y `PIUU_GOOGLE_TOKEN` como rutas por
defecto; los argumentos explícitos tienen prioridad. Son variables del helper,
no sustituyen las variables de cada MCP en el servicio.

## 3. Copiar el token al VPS

Desde la PC, después de completar la autorización:

```bash
ssh denis@74.208.158.226 'mkdir -p /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets && chmod 700 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets'
scp -p .secrets/google-token.json denis@74.208.158.226:/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json.new
ssh denis@74.208.158.226 'chmod 600 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json.new && mv -f /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json.new /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json'
```

La ruta confirmada empieza con **`/home/denis/apps`**, no con `/apps`.
El cliente `credentials.json` ya está en ese repositorio del VPS. Mantenerlo
privado, igual que los tokens. Estos archivos están excluidos de Git.

## 4. Configurar el backend del VPS

Desplegar primero el código actualizado de PIUU_SERVER, incluidos los módulos de
Calendar, Contactos y OAuth compartido. El Python confirmado de `walle.service`
es `/home/denis/apps/.venv/bin/python`. Dentro del repositorio remoto:

```bash
cd /home/denis/apps/Server-Wall-e-PCM-OPUS
/home/denis/apps/.venv/bin/python -m pip install -r src/requirements-mcp.txt
PYTHONPATH=src /home/denis/apps/.venv/bin/python -m bridge.mcp_servers.google_auth --check --token "$PWD/.secrets/google-token.json"
sudo systemctl edit walle.service
```

Agregar o actualizar las entradas correspondientes del override existente,
conservando las demás variables del servicio:

```ini
[Service]
Environment="PIUU_VOICE_PROVIDER=gemini"
Environment="PIUU_GMAIL_MCP_ENABLED=1"
Environment="PIUU_GMAIL_CREDENTIALS=/home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json"
Environment="PIUU_GMAIL_TOKEN=/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json"
Environment="PIUU_CALENDAR_MCP_ENABLED=1"
Environment="PIUU_CALENDAR_CREDENTIALS=/home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json"
Environment="PIUU_CALENDAR_TOKEN=/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/google-token.json"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart walle.service
sudo journalctl -u walle.service -n 100 --no-pager
```

Buscar `MCP_SERVER_READY` para Gmail y Calendar. Probar primero consultas de
correo, contactos y agenda; luego pedir explícitamente las creaciones o envíos
que quieras probar. No existen todavía variables para activar Tasks o Drive.

Los avisos de Calendar diez minutos antes se configuran aparte en
[CALENDAR_MCP.md](CALENDAR_MCP.md); requieren el firmware CORE compatible con
notificaciones. La autorización OAuth y las herramientas por voz no requieren
ese cambio de firmware.

## Renovación y compatibilidad

Los flujos individuales de [Gmail](GMAIL_MCP.md) y [Calendar](CALENDAR_MCP.md)
siguen disponibles con sus archivos independientes. Si se usa el token conjunto,
repetir este helper `google_auth --auth` para ampliar o renovar consentimiento;
no apuntar una autorización individual al archivo conjunto, porque podría
reemplazarlo por otro con menos scopes. La renovación automática conserva los
scopes almacenados y guarda el archivo mediante reemplazo atómico privado.

Validación: 215 pruebas backend offline aprobadas, incluidos consentimiento
incompleto, refresh token ausente, compatibilidad del token conjunto con Gmail,
Contactos y Calendar, y errores sin exposición de secretos. No se realizó
consentimiento real, despliegue ni escritura en Google durante esta preparación.
