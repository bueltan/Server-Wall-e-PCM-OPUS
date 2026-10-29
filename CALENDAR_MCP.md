# Google Calendar MCP y avisos de PIUU

Para autorizar Gmail, Contactos, Calendar, Tasks y archivos de Drive juntos,
seguir [GOOGLE_AUTH.md](GOOGLE_AUTH.md). Permite usar un único token en Gmail y
Calendar; los pasos individuales de esta guía siguen disponibles como alternativa.

Flujo de herramientas: **PIUU → Gemini Live → Tool Manager → Calendar MCP → Calendar API**.
El servidor propio usa stdio, el mismo Python que el backend y herramientas
`NON_BLOCKING` con resultados `WHEN_IDLE`. Gmail y Calendar se pueden habilitar
por separado o juntos. Home MCP sigue pendiente.

## Herramientas

| Herramienta | Uso |
| --- | --- |
| `calendar_list_calendars` | Calendarios suscritos, IDs, permisos, zonas horarias y hora local actual. |
| `calendar_list_events` | Agenda o búsqueda por texto, orden cronológico, recurrencias expandidas y paginación. |
| `calendar_get_event` | Un evento por ID, con descripción limitada a 12.000 caracteres. |
| `calendar_check_availability` | Intervalos ocupados mediante FreeBusy; los errores no se interpretan como disponibilidad. |
| `calendar_create_event` | Crear eventos de texto, con horario o de día completo; devuelve ID y enlace. |

La primera versión no modifica ni elimina eventos, no crea recurrencias y no
agrega invitados. Una serie existente se consulta mediante sus ocurrencias.
El prompt solicita comprobar disponibilidad y aclarar conflictos antes de crear.
Esa comprobación no reserva el horario ni impide cambios concurrentes.

Los horarios usan RFC3339 con offset explícito, por ejemplo
`2026-10-30T15:00:00-03:00`. La zona IANA predeterminada es
`America/Argentina/Buenos_Aires`; el offset del evento debe coincidir con la zona.
Para eventos de día completo se usan fechas `YYYY-MM-DD` y el fin es exclusivo:
un evento del 30 de octubre lleva inicio `2026-10-30` y fin `2026-10-31`.

Los títulos y descripciones son datos no confiables, nunca instrucciones. Gemini
solicita fechas, horas o duración faltantes. `status=created` confirma la creación;
`status=unknown` indica que hay que consultar el `event_id` antes de volver a
intentarlo. No se reintenta automáticamente una creación ni se repite una llamada
cancelada con el mismo ID dentro de la caché de sesión. El ID enviado a Google
permite verificar una creación cuyo resultado sea incierto.

## Autorizar Calendar en la PC

Habilitar **Google Calendar API** en el mismo proyecto Google Cloud del cliente
OAuth Desktop de `credentials.json`. La API es distinta de Gmail API: habilitar
Gmail no habilita Calendar. Se puede reutilizar el archivo de cliente OAuth,
pero Calendar guarda un token separado en `.secrets/calendar-token.json`.

Scopes solicitados:

- `https://www.googleapis.com/auth/calendar.events`
- `https://www.googleapis.com/auth/calendar.calendarlist.readonly`
- `https://www.googleapis.com/auth/calendar.events.freebusy`

Desde la PC, con `credentials.json` en la raíz del proyecto:

```bash
cd /home/denis/Documents/PlatformIO/Projects/PIUU_SERVER
.venv/bin/python -m pip install -r src/requirements-mcp.txt
PIUU_CALENDAR_CREDENTIALS="$PWD/credentials.json" \
PIUU_CALENDAR_TOKEN="$PWD/.secrets/calendar-token.json" \
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.calendar --auth
```

Completar el consentimiento en el navegador. El flujo y la renovación del token
reutilizan los helpers OAuth de Gmail. El proceso normal nunca abre el navegador.
El token de Gmail no se reemplaza ni se usa como token de Calendar. Si cambian
los scopes de Calendar, repetir `--auth` y copiar de nuevo el token de Calendar.

Para probar localmente, agregar estas variables a las del proveedor Gemini:

```bash
export PIUU_CALENDAR_MCP_ENABLED=1
export PIUU_CALENDAR_TIME_ZONE=America/Argentina/Buenos_Aires
.venv/bin/python src/main.py
```

## Copiar el token y activar en el VPS

El proyecto del VPS está en `/home/denis/apps/Server-Wall-e-PCM-OPUS`, con acceso
`denis@74.208.158.226`. Las credenciales de cliente ya se copiaron para Gmail.
Después de autorizar Calendar en la PC:

```bash
ssh denis@74.208.158.226 'mkdir -p /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets && chmod 700 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets'
scp .secrets/calendar-token.json denis@74.208.158.226:/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/calendar-token.json
ssh denis@74.208.158.226 'chmod 600 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/calendar-token.json'
```

Desplegar los nuevos archivos del backend. El servicio `walle.service` usa
**`/home/denis/apps/.venv/bin/python`**, un entorno distinto del `.venv` dentro
del proyecto. Instalar los requirements en el entorno del servicio, desde el VPS:

```bash
cd /home/denis/apps/Server-Wall-e-PCM-OPUS
/home/denis/apps/.venv/bin/python -m pip install -r src/requirements-mcp.txt
sudo systemctl edit walle.service
```

En el override, agregar las variables de Calendar, conservando las de Gmail:

```ini
[Service]
Environment="PIUU_CALENDAR_MCP_ENABLED=1"
Environment="PIUU_CALENDAR_CREDENTIALS=/home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json"
Environment="PIUU_CALENDAR_TOKEN=/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/calendar-token.json"
Environment="PIUU_CALENDAR_TIME_ZONE=America/Argentina/Buenos_Aires"
```

Aplicar los cambios y observar los logs:

```bash
sudo systemctl daemon-reload
sudo systemctl restart walle.service
sudo journalctl -u walle.service -n 60 --no-pager
```

Probar primero «¿Qué eventos tengo mañana?» y luego «Crea un evento de prueba
el 30 de octubre de 2026 de 15 a 15:30 con título Prueba PIUU».

## Avisos proactivos diez minutos antes

El usuario eligió avisos de próximos eventos **10 minutos antes**. Esta primera
versión consulta un solo calendario cada **120 segundos**, con una ventana de
24 horas y hasta 1.000 eventos por página. Recorre las páginas necesarias, con
límite de diez; no reemplaza la agenda por una respuesta incompleta si se supera
el límite o falla una página. Omite eventos cancelados, de día completo y eventos
rechazados por la propia cuenta. Los de día completo requieren definir una hora
de aviso y se dejan para otra etapa.

Un reloj local evalúa cada segundo los eventos ya descargados. Un evento que ya
está en la caché no espera a la próxima consulta para disparar su aviso. Los eventos
nuevos, movidos o cancelados se detectan en la siguiente consulta, normalmente en
hasta 120 segundos más un pequeño jitter. En caso de fallos se aplica espera
progresiva, hasta 30 minutos. Una caché más antigua que el máximo entre 300 segundos
y dos intervalos de consulta no produce avisos.

Con una página por consulta, 120 segundos equivale aproximadamente a **720
solicitudes diarias** para ese calendario, más paginación y consultas por voz.
No garantiza una cuota disponible: los límites efectivos se verifican en Google
Cloud para el proyecto y usuario. El intervalo se puede aumentar; el mínimo
aceptado es 60 segundos. La lectura periódica no abre una sesión Gemini: Gemini
se conecta solamente al conversar o al entregar un aviso a CORE.

Variables adicionales en `[Service]`, una vez que CORE tenga el firmware nuevo:

```ini
Environment="PIUU_CALENDAR_REMINDERS_ENABLED=1"
Environment="PIUU_CALENDAR_POLL_SEC=120"
Environment="PIUU_CALENDAR_REMINDER_MINUTES=10"
Environment="PIUU_CALENDAR_REMINDER_CALENDAR=primary"
Environment="PIUU_CALENDAR_REMINDER_STATE=/home/denis/apps/Server-Wall-e-PCM-OPUS/.state/calendar-reminders.json"
```

Los avisos requieren Gemini y `PIUU_CALENDAR_MCP_ENABLED=1`. Permanecen desactivados
por defecto. El servicio debe poder crear y escribir `.state/`; el archivo guarda
solo claves opacas y fechas para deduplicar avisos, con permisos `600`. `.state/`
está excluido de Git. No borrar este archivo al actualizar el backend.

### Activación de PIUU_CORE

Se agregó la capacidad opcional UDP v3 `CAP_NOTIFICATIONS=0x0010` y el paquete
`NOTICE=28`. El backend ofrece el aviso solo con CORE conectado e inactivo y la
capacidad negociada. CORE elige el ID de conversación y responde con un START
cuyo cuerpo contiene el número de solicitud NOTICE (uint32 big-endian).
Los START manuales mantienen el cuerpo vacío y tienen prioridad en una carrera.

El backend abre Gemini, espera READY y su confirmación de CORE, y envía el
recordatorio como texto. Gemini produce el audio para la salida Opus normal.
CORE mantiene el micrófono silenciado al abrir el aviso; después de reproducirlo
retoma la escucha con la protección de eco habitual. Si no llega una respuesta en
20 segundos, CORE aborta la apertura del aviso. No se interrumpe una conversación
activa. Los avisos esperan mientras CORE esté ocupado o desconectado, hasta la
hora de inicio del evento; después se omiten.

La marca de deduplicación se guarda **antes de enviar el texto a Gemini**. Se
prioriza no repetir avisos ante reinicios o resultados inciertos: una marca prueba
un intento de envío, no que el usuario escuchó el audio. Si falla después de esa
marca, no se repite automáticamente. Los logs `NOTICE_SUBMITTED` y
`CALENDAR_REMINDER_SUBMITTED` tampoco equivalen a una confirmación de reproducción;
esta corresponde al flujo `PLAYBACK_FINISHED`.

**Actualizar backend y firmware CORE para usar los avisos.** La extensión es
opcional: la integración Calendar por voz funciona con CORE anterior, pero no
puede despertarlo de forma proactiva. No se desplegó Calendar al VPS ni se cargó
firmware en el dispositivo durante esta implementación.

### Webhooks después

Cuando haya un dominio HTTPS, se puede sustituir la consulta periódica por
`events.watch` y una sincronización al recibir cambios. El temporizador local
seguirá siendo necesario para avisar diez minutos antes. Los canales de Google
expiran y requieren renovación; las notificaciones de cambio no son una alarma
a la hora del evento. Esta versión todavía no publica endpoints ni registra
webhooks.

## Verificación y diagnóstico

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

**Validación local: 197 pruebas backend, tests host de CORE e interoperabilidad
Python/C++ pasaron; el firmware `cyd-uart0` compiló correctamente.**

Se prueban paginación, recurrencias consultadas, zona horaria, FreeBusy con errores,
fechas exclusivas de día completo, creación incierta sin reintentos, OAuth separado,
MCP Gmail+Calendar real por stdio, avisos diferidos, caducados, cancelados y
persistidos, y negociación NOTICE/START/READY con Gemini simulado. No se crean
eventos reales ni se envían correos desde los tests.

Los logs `MCP_SERVER_READY`, `MCP_TOOL_STARTED` y `MCP_TOOL_FINISHED` identifican
servidor, herramienta y estado sin imprimir argumentos, cuerpos ni tokens.
`CALENDAR_REMINDERS_SYNCED` informa la actualización de la agenda.
`calendar_api_disabled` requiere habilitar Calendar API; `calendar_auth_required`
requiere autorizar el token. `calendar_access_denied` exige revisar el permiso
sobre el calendario y los scopes concedidos.

Fuentes: [OAuth Calendar](https://developers.google.com/workspace/calendar/api/quickstart/python),
[scopes](https://developers.google.com/workspace/calendar/api/auth),
[creación de eventos](https://developers.google.com/workspace/calendar/api/v3/reference/events/insert),
[FreeBusy](https://developers.google.com/workspace/calendar/api/v3/reference/freebusy/query),
[cuotas](https://developers.google.com/workspace/calendar/api/guides/quota),
[webhooks](https://developers.google.com/workspace/calendar/api/guides/push).
