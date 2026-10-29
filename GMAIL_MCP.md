# Gmail MCP de PIUU

Para autorizar Gmail, Contactos, Calendar, Tasks y archivos de Drive juntos,
seguir [GOOGLE_AUTH.md](GOOGLE_AUTH.md). Permite usar un único token en Gmail y
Calendar; los pasos individuales de esta guía siguen disponibles como alternativa.

Flujo implementado: **PIUU → Gemini Live → Tool Manager → Gmail MCP → Gmail API**.
El Tool Manager descubre herramientas con `tools/list`, agrega el prefijo `gmail_`
y enruta las llamadas mediante `tools/call`. La conexión MCP usa stdio y un
subproceso Python con el mismo entorno virtual del bridge. `ServerSpec` permite
registrar servidores adicionales. Calendar está implementado; ver
[CALENDAR_MCP.md](CALENDAR_MCP.md). Home sigue pendiente.

Herramientas disponibles. La autorización solicita `gmail.readonly`, `gmail.send`
y `contacts` (lectura y escritura de contactos); las operaciones de correo solo
requieren los dos primeros:

- `gmail_search_messages(query="in:inbox", max_results=5, page_token="")`:
  búsqueda con sintaxis Gmail, metadatos y paginación. Máximo 20 mensajes.
- `gmail_read_message(message_id)`:
  metadatos y cuerpo `text/plain`, hasta 12.000 caracteres. Omite adjuntos y HTML;
  `body_available=false` indica que no hay texto plano disponible, y el snippet
  puede servir como vista previa. Leer no marca el mensaje como leído.

- `gmail_send_message(to, subject, body)`:
  envía un correo de texto plano a una dirección explícita (por ejemplo,
  `persona@gmail.com`), con asunto y cuerpo. Devuelve `status=sent`, ID del mensaje
  e ID del hilo cuando Gmail acepta el envío. No incluye adjuntos, CC ni BCC.

- `gmail_search_contacts(query, max_results=10)`:
  busca contactos guardados por prefijo de nombre, correo o teléfono, hasta 30
  resultados. Devuelve nombres, direcciones con etiquetas y teléfonos.
- `gmail_get_contact(resource_name)`:
  consulta un contacto guardado usando el `resource_name` devuelto por la búsqueda.

- `gmail_create_contact(given_name, family_name="", email="", phone="")`:
  crea un contacto guardado con nombre y al menos un correo o teléfono, únicamente
  cuando el usuario lo pide. Devuelve `status=created` y el contacto con su
  `resource_name`. La creación no envía correos.

El MCP permite consultar y crear contactos. No modifica ni elimina contactos
existentes y no incluye «Otros contactos» generados automáticamente ni
directorios corporativos.
La búsqueda de People API no es una búsqueda libre por subcadenas: usa prefijos.
Si devuelve `possibly_truncated=true`, acotar el nombre o correo; este endpoint
no ofrece paginación. Si hay varias personas o direcciones posibles, PIUU pide
que el usuario elija antes de enviar. No deduce ni inventa direcciones.

La búsqueda hace una solicitud vacía para actualizar la caché de Google, espera
cinco segundos y realiza la consulta, según la documentación de People API. Esa
espera ocurre en el worker de la herramienta y mantiene `NON_BLOCKING` en Gemini.
Los datos de contacto se devuelven a Gemini como datos no confiables y no se
escriben en los logs de ciclo de herramientas.

No incluye borradores, cambios de etiquetas ni eliminación.

## Instalación y autorización

Desde el directorio `PIUU_SERVER`:

```bash
.venv/bin/python -m pip install -r src/requirements-mcp.txt
```

En Google Cloud, habilitar **Gmail API** y **People API**, configurar la pantalla de consentimiento
OAuth y crear credenciales de tipo **Desktop app**. Si la aplicación está en
modo de pruebas, agregar la cuenta como usuario de prueba. Guardar el JSON
descargado en `.secrets/gmail-client.json`.

Ejecutar la autorización interactiva una vez:

```bash
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.gmail --auth
```

El comando abre el navegador para el consentimiento y recibe la respuesta OAuth
por un puerto local temporal. Guarda `.secrets/gmail-token.json` con permisos
`0600`. El proceso MCP normal nunca abre un navegador: si falta autorización,
la herramienta devuelve un error y debe ejecutarse nuevamente `--auth`.

Si ya existía un token de solo lectura, **volver a ejecutar `--auth`** para conceder
`gmail.send`. El comando reemplaza el token al terminar correctamente, sin tener
que borrarlo antes. Si la pantalla de consentimiento OAuth tiene una lista
explícita de permisos, agregar también `https://www.googleapis.com/auth/gmail.send`
en Google Cloud. Los tokens antiguos de solo lectura se detectan y se rechazan
con `gmail_auth_required` hasta completar la nueva autorización.

Para consultar y crear contactos, agregar `https://www.googleapis.com/auth/contacts`
a la pantalla de consentimiento si el proyecto mantiene una lista explícita de
scopes. Volver a ejecutar el comando `--auth` de Gmail desde la PC y copiar el token
actualizado al VPS como se describe más abajo. La autorización ahora incluye
correo y contactos, con el mismo cliente OAuth y archivo de token de Gmail.
Un token anterior con lectura y envío sigue sirviendo para el correo; solo las
herramientas de contactos devuelven `contacts_auth_required` hasta autorizar el
permiso adicional. Un token anterior con `contacts.readonly` sigue sirviendo para
buscar y leer contactos; solo la creación devuelve `contacts_write_auth_required`
hasta autorizar `contacts`. No hace falta conceder ambos scopes de contactos:
el de escritura también permite leer. El token de Calendar no cambia.

Los tokens se renuevan automáticamente mientras el refresh token sea válido.
`.secrets/` y `.env` están excluidos de Git. Se pueden cambiar las ubicaciones
con `PIUU_GMAIL_CREDENTIALS` y `PIUU_GMAIL_TOKEN` (preferir rutas absolutas).

Activar Gmail al iniciar el bridge:

```bash
export GEMINI_API_KEY='...'
export PIUU_VOICE_PROVIDER=gemini
export PIUU_GMAIL_MCP_ENABLED=1
.venv/bin/python src/main.py
```

Gmail está desactivado por defecto y las dependencias adicionales son opcionales.
La integración corresponde a Gemini en UDP v3; los adaptadores legacy y xAI
conservan su comportamiento. Para ejecutar el servidor de forma independiente:

```bash
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.gmail
```

## Servidor VPS: ruta y autorización desde la PC

La ruta confirmada del proyecto en el servidor es:

```text
/home/denis/apps/Server-Wall-e-PCM-OPUS
```

El acceso SSH es `denis@74.208.158.226`. `/apps/Server-Wall-e-PCM-OPUS` no existe:
el directorio `apps` está dentro de `/home/denis`.

**Conviene ejecutar `--auth` en la PC y copiar el token al servidor.** La
implementación usa `run_local_server(port=0)`: abre el navegador y recibe la
respuesta de Google por un puerto temporal de localhost. Ejecutarlo por SSH en
el VPS sin navegador requeriría además un túnel al puerto elegido; no basta con
abrir la URL en el navegador de la PC porque localhost apuntaría a la PC.
Ver el [flujo local de Google OAuth](https://googleapis.dev/python/google-auth-oauthlib/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_local_server).

### 1. Autorizar desde la PC

Con `credentials.json` en la raíz local de `PIUU_SERVER`, ejecutar en la PC:

```bash
cd /home/denis/Documents/PlatformIO/Projects/PIUU_SERVER
PIUU_GMAIL_CREDENTIALS="$PWD/credentials.json" \
PIUU_GMAIL_TOKEN="$PWD/.secrets/gmail-token.json" \
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.gmail --auth
```

Aceptar los permisos de lectura y envío en el navegador. El archivo
`.secrets/gmail-token.json` contiene la autorización para la cuenta elegida.
`credentials.json` identifica al cliente OAuth; copiarlo por sí solo no autoriza
el acceso a la cuenta. Usar el mismo cliente OAuth en la PC y el VPS.

### 2. Copiar las credenciales y el token

Desde la misma carpeta de la PC, copiar las credenciales al proyecto del VPS
(si ya se copiaron, este primer comando se puede omitir):

```bash
scp credentials.json denis@74.208.158.226:/home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json
ssh denis@74.208.158.226 'chmod 600 /home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json'
```

Crear el directorio privado y copiar el token generado:

```bash
ssh denis@74.208.158.226 'mkdir -p /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets && chmod 700 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets'
scp .secrets/gmail-token.json denis@74.208.158.226:/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json
ssh denis@74.208.158.226 'chmod 600 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json'
```

`credentials.json` y `.secrets/` están excluidos por el `.gitignore` del proyecto.
El usuario que ejecuta el servicio debe poder leer las credenciales y leer y
actualizar el token dentro de `.secrets/`; estos comandos dejan los archivos a
nombre de `denis`.

### Estado verificado de la copia (2026-10-29)

El token generado en la PC ya se copió mediante SCP al VPS
`denis@74.208.158.226`. Las credenciales OAuth ya estaban presentes. Se verificó
que el token no está vacío y se aplicaron estos permisos y propietario:

| Recurso en `/home/denis/apps/Server-Wall-e-PCM-OPUS` | Permisos | Propietario |
| --- | --- | --- |
| `.secrets/` | `700` | `denis` |
| `.secrets/gmail-token.json` | `600` | `denis` |
| `credentials.json` | `600` | `denis` |

No hace falta repetir la copia mientras el token siga siendo válido. Esta
verificación confirma la presencia y los permisos de los archivos; todavía no
se probó una llamada real a Gmail desde el VPS.

Queda pendiente verificar las variables del servicio, en particular
`PIUU_GMAIL_TOKEN=/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json`,
recargar systemd y reiniciar el servicio según el siguiente apartado. Durante
la copia no se modificó ni se reinició el servicio.

### 3. Configurar el servicio del VPS

Agregar estas variables en la sección `[Service]` del servicio systemd, junto
con `GEMINI_API_KEY` y la configuración existente:

```ini
Environment="PIUU_VOICE_PROVIDER=gemini"
Environment="PIUU_GMAIL_MCP_ENABLED=1"
Environment="PIUU_GMAIL_CREDENTIALS=/home/denis/apps/Server-Wall-e-PCM-OPUS/credentials.json"
Environment="PIUU_GMAIL_TOKEN=/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json"
```

Instalar también las dependencias MCP en el entorno del VPS, desde el servidor:

```bash
cd /home/denis/apps/Server-Wall-e-PCM-OPUS
.venv/bin/python -m pip install -r src/requirements-mcp.txt
```

Después de guardar cambios en la unidad systemd, ejecutar en el VPS (reemplazar
`NOMBRE_DEL_SERVICIO.service` por el nombre real):

```bash
sudo systemctl daemon-reload
sudo systemctl restart NOMBRE_DEL_SERVICIO.service
```

El servidor renueva el token automáticamente mientras sea válido; no necesita
un navegador durante el funcionamiento normal. Si Google revoca el permiso o
vence el refresh token, repetir `--auth` en la PC y volver a copiar el token.

## Activar y probar contactos

Primero habilitar [People API en Google Cloud](https://console.cloud.google.com/apis/library/people.googleapis.com)
en el proyecto que generó `credentials.json`. La autorización de Gmail no habilita
esta API por sí sola. Si falta, el resultado indica `people_api_disabled`.

Después de desplegar el backend actualizado, repetir en la PC:

```bash
cd /home/denis/Documents/PlatformIO/Projects/PIUU_SERVER
PIUU_GMAIL_CREDENTIALS="$PWD/credentials.json" \
PIUU_GMAIL_TOKEN="$PWD/.secrets/gmail-token.json" \
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.gmail --auth
```

Aceptar el permiso de contactos (`contacts`, que incluye escritura) y copiar el
token actualizado:

```bash
scp .secrets/gmail-token.json denis@74.208.158.226:/home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json
ssh denis@74.208.158.226 'chmod 600 /home/denis/apps/Server-Wall-e-PCM-OPUS/.secrets/gmail-token.json'
```

Reiniciar `walle.service` en el VPS. Las variables `PIUU_GMAIL_CREDENTIALS`,
`PIUU_GMAIL_TOKEN` y `PIUU_GMAIL_MCP_ENABLED` conservan sus valores. No hay nuevas
dependencias: People API usa las mismas librerías de Google ya instaladas.

Probar «Busca el correo de Ana en mis contactos» y luego, si corresponde,
«Envía un correo a Ana con asunto Prueba y cuerpo Hola desde PIUU». Resolver
cualquier ambigüedad antes del envío. La búsqueda por sí sola no autoriza enviar.
Para crear un contacto, por ejemplo: «Guarda un contacto llamado Ana Pérez con
correo ana@example.com y teléfono +54 11 1234». PIUU busca primero si ya existe
y usa solo los datos proporcionados; pide cualquier dato imprescindible que falte.
No hay actualización ni fusión automática de contactos existentes.

La herramienta serializa las creaciones y desactiva los reintentos de API. Si
`status=unknown`, un timeout o una cancelación dejan el resultado incierto, revisar
los contactos antes de solicitar otra creación. No se vuelve a ejecutar el mismo
ID de llamada cancelado dentro de la caché de sesión, pero no hay garantía contra
duplicados entre reinicios o llamadas con IDs distintos. Una búsqueda inmediata
sin resultados no es motivo para crear otra vez: conservar el `resource_name`
devuelto cuando la creación sí fue confirmada.

Esta implementación todavía no fue desplegada ni validada con contactos reales.

## Probar el envío desde PIUU

Tras autorizar de nuevo y reiniciar el bridge con `PIUU_GMAIL_MCP_ENABLED=1`,
se puede decir: «Envía un correo a MI_DIRECCION@gmail.com con asunto Prueba PIUU
y cuerpo Hola, este es un correo de prueba de PIUU». Reemplazar la dirección
por la cuenta donde se quiere recibir la prueba.

Gemini dispone de `gmail_send_message` y solicita los datos faltantes. El mensaje
se envía desde la cuenta autorizada. `status=sent` confirma que Gmail aceptó el
mensaje; no garantiza su entrega en el buzón de destino. Si `status=unknown` o
la herramienta informa timeout, revisar Enviados antes de solicitar otro envío.
No hay reintentos automáticos. La deduplicación por ID y la marca de llamadas
canceladas evitan repetir el mismo envío dentro de la caché de sesión; no ofrecen
una garantía de exactamente una entrega entre reinicios o IDs diferentes.
Cancelar la espera de la herramienta no deshace un envío aceptado por Gmail.

## Ejecución asíncrona

Las declaraciones Gmail usan `behavior: NON_BLOCKING` explícito; las respuestas
incluyen `response.scheduling: WHEN_IDLE`. La herramienta de expresiones mantiene
`BLOCKING` para preparar los gestos antes del audio. El soporte de ejecución
asíncrona depende del modelo Live seleccionado.

Las consultas MCP corren en tareas separadas del receptor WebSocket, con cola
acotada y timeout por llamada. El servidor procesa las consultas en orden.
Un `turnComplete` o una nueva intervención del usuario no descarta consultas
pendientes: sus resultados pueden llegar después. `toolCallCancellation` cancela
la espera de las llamadas indicadas y descarta sus resultados; cerrar la sesión
cancela todas las esperas y termina el subproceso MCP. Una consulta de Gmail ya
enviada puede continuar hasta su timeout aunque su resultado se haya descartado.

Las llamadas repetidas en curso se deduplican por ID; se conservan los últimos
128 resultados de sesión para responder repeticiones sin consultar Gmail otra vez.
Los errores MCP se devuelven con códigos genéricos, sin propagar credenciales.
El subproceso no hereda las claves Gemini/xAI. El prompt indica que el contenido
de los correos es información no confiable y no debe ejecutarse como instrucciones.

## Verificación

Con la integración de contactos, **209 pruebas backend pasaron**, sin leer
contactos reales ni enviar correos.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

`test_gmail_mcp.py` cubre MIME de lectura y envío, búsqueda, paginación, límites,
compatibilidad del token anterior de Gmail, permiso adicional de contactos,
búsqueda, detalle y creación People API simulados, compatibilidad con contactos de
solo lectura, creaciones inciertas sin reintentos, errores de envío y permisos del token,
planificación Gemini, deduplicación, cancelación, continuidad entre turnos y el
protocolo MCP real con subprocesos. Las pruebas no acceden a una cuenta de Gmail.
Sin el SDK MCP, las dos pruebas de integración stdio se omiten.

Referencias: [SDK MCP v1](https://py.sdk.modelcontextprotocol.io/v1/),
[OAuth Gmail en Python](https://developers.google.com/workspace/gmail/api/quickstart/python),
[lectura y búsqueda de contactos](https://developers.google.com/people/v1/contacts),
[creación de contactos y permiso requerido](https://developers.google.com/people/api/rest/v1/people/createContact),
[herramientas asíncronas en Gemini Live](https://ai.google.dev/gemini-api/docs/live-api/tools#asynchronous-function-calling).


## Estado posterior de Gmail

Tras habilitar Gmail API en Google Cloud, el usuario confirmó el funcionamiento
de Gmail desde PIUU. La integración de Calendar usa un token independiente y el
mismo cliente OAuth Desktop; no requiere reemplazar el token de Gmail.
