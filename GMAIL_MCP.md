# Gmail MCP de PIUU

Flujo implementado: **PIUU → Gemini Live → Tool Manager → Gmail MCP → Gmail API**.
El Tool Manager descubre herramientas con `tools/list`, agrega el prefijo `gmail_`
y enruta las llamadas mediante `tools/call`. La conexión MCP usa stdio y un
subproceso Python con el mismo entorno virtual del bridge. `ServerSpec` permite
registrar posteriormente servidores Calendar y Home; aún no están implementados.

Herramientas disponibles, con permisos `gmail.readonly` y `gmail.send`:

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

No incluye borradores, cambios de etiquetas ni eliminación.

## Instalación y autorización

Desde el directorio `PIUU_SERVER`:

```bash
.venv/bin/python -m pip install -r src/requirements-mcp.txt
```

En Google Cloud, habilitar **Gmail API**, configurar la pantalla de consentimiento
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

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

`test_gmail_mcp.py` cubre MIME de lectura y envío, búsqueda, paginación, límites,
rechazo del token antiguo, errores de envío sin reintentos, permisos del token,
planificación Gemini, deduplicación, cancelación, continuidad entre turnos y el
protocolo MCP real con subprocesos. Las pruebas no acceden a una cuenta de Gmail.
Sin el SDK MCP, las dos pruebas de integración stdio se omiten.

Referencias: [SDK MCP v1](https://py.sdk.modelcontextprotocol.io/v1/),
[OAuth Gmail en Python](https://developers.google.com/workspace/gmail/api/quickstart/python),
[herramientas asíncronas en Gemini Live](https://ai.google.dev/gemini-api/docs/live-api/tools#asynchronous-function-calling).
