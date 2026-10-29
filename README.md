# PIUU CORE ↔ Gemini Live / xAI Voice bridge

## Entorno Python y dependencias

Ejecutar estos comandos desde `PIUU_SERVER`. El entorno virtual se llama `.venv`;
el punto inicial hace que sea una carpeta oculta en muchos exploradores. Se
verificó con Python 3.12. Para preparar una instalación nueva:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements-mcp.txt
python -m pip check
```

`src/requirements-mcp.txt` incluye los requirements base y las dependencias de
MCP y OAuth de Gmail. Para usar solo el bridge de voz, instalar en su lugar
`src/requirements.txt`. Si `.venv` ya existe, basta con activarlo e instalar los
requirements; no hace falta recrearlo.

Para iniciar el backend con las variables del proveedor configuradas:

```bash
source .venv/bin/activate
python src/main.py
```

También se puede ejecutar `.venv/bin/python src/main.py` sin activar el entorno.
La instalación local de dependencias y sus imports se verificaron sin conflictos.

## Reconexión de CORE por MAC (2026-10-28)

CORE anuncia su MAC WiFi STA en los seis bytes del cuerpo HELLO, con
`CAP_DEVICE_MAC=0x0008`. El backend confirma ese bit en HELLO_ACK y asocia la primera
MAC declarada durante la vida del proceso. Otra MAC, o un HELLO sin MAC después
de esa asociación, no puede reemplazar al dispositivo admitido. Se confía en la
MAC declarada; la verificación mediante una clave compartida queda pendiente.
La asociación no se persiste y se pierde al reiniciar PIUU_SERVER.

Un HELLO de la misma MAC con un nonce de sesión nuevo permite actualizar la IP
y/o el puerto UDP sin esperar los diez segundos de ausencia. Si hay conversación
activa, primero se cierra el proveedor y se limpian audio y eventos; CORE reintenta
HELLO hasta completar el cierre. Después se genera un nuevo nonce de servidor y
se exige otro START. No se conserva ni se reanuda el historial de conversación.
Los paquetes deben coincidir con la sesión y dirección actuales; los últimos 32
nonces retirados no pueden restaurar una conexión anterior.

`CORE_ADMITTED` registra la admisión inicial y `CORE_REBOUND` la nueva sesión y
las direcciones anterior/nueva. `peer_restarted` significa cambio de identidad de
sesión; por sí solo no demuestra un reinicio físico de CORE. Los clientes legacy
con HELLO sin MAC conservan su restricción a la dirección de origen original.
Ver el [contrato de identidad MAC](PROTOCOLO_UDP_V3.md#identidad-estable-de-core-y-reconexión-udp).

**Desplegar primero PIUU_SERVER y después el firmware CORE actualizado.** CORE
exige la confirmación del bit MAC; un backend anterior no es compatible con ese
handshake. La corrección de coordinación WiFi está en CORE: un escaneo conectado
ya no simula una desconexión ni reinicia su identidad, y el worker coordina las
operaciones explícitas con los reintentos automáticos de Arduino.

Validación realizada: **161 pruebas backend offline**, pruebas host y de
interoperabilidad CORE/backend, y compilación del firmware `cyd-uart0`. Desde
PIUU_SERVER, repetir las pruebas del backend con:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

No se desplegó el backend ni se cargó firmware durante este chat. Falta validar
los cambios en el VPS y el hardware. CORE corrige además el bloqueo de lectura
UDP tras un fallo temporal del mutex: reintenta el datagrama pendiente y descarta
enteros los datagramas sobredimensionados o leídos parcialmente. Es una corrección
local, sin cambios adicionales al backend ni al protocolo; falta probarla con los
microcortes WiFi reales.

## Gemini Live (UDP v3)

El backend permite elegir `gemini` (predeterminado) o `xai` sin cambiar el firmware,
el protocolo UDP v3 ni el formato Opus de CORE. La selección se aplica al iniciar
el proceso; no se cambia de proveedor dentro de una conversación.

Desde la raíz del backend, con las dependencias de `src/requirements.txt` instaladas:

```bash
export PIUU_VOICE_PROVIDER=gemini
export GEMINI_MODEL=gemini-3.8-live
export GEMINI_VOICE=Leda
read -rsp 'GEMINI_API_KEY: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
.venv/bin/python src/main.py
```

La clave debe pertenecer a un proyecto con acceso a Gemini Live. No se guarda en
el repositorio ni se incluye en la URL WebSocket. El backend no carga archivos
`.env` automáticamente. Para volver a xAI, usar `PIUU_VOICE_PROVIDER=xai` y la
variable `XAI_API_KEY` habitual. No ejecutar dos servidores en el mismo puerto.

La voz base predeterminada es Leda. `BridgeConfig.gemini_voice_style` agrega
instrucciones de interpretación: un pajarito bebé de dibujos animados, con voz
aguda, dulce, brillante, alegre y juguetona, manteniendo claridad en español y
sin gritar. Esto orienta la voz predefinida; no reproduce una voz diseñada.
El prompt de conversación y las instrucciones de ojos se conservan.

No usar IDs `voice_...` ni `voices/...` en Gemini Live: el modo TTS personalizado
queda pendiente. Si el servicio conserva un ID personalizado, reemplazarlo por
`Environment="GEMINI_VOICE=Leda"` y reiniciar después de desplegar estos cambios.

Flujo implementado:

1. CORE abre la conversación mediante START. Se abre el WebSocket y se espera
   `setupComplete` antes de enviar audio o anunciar READY.
2. PCM16 mono a 16 kHz entra por `realtimeInput.audio`. Gemini usa detección de
   actividad automática; `voiceActivity` se traduce a eventos SPEECH_STARTED y
   SPEECH_STOPPED. No se envían commits ni comandos de xAI.
3. Gemini entrega PCM16 a 24 kHz. Un FIR con estado continuo convierte a 16 kHz
   antes de grabar/encodificar Opus. Mantiene fase entre fragmentos, rechaza
   aliasing y tiene 1,25 ms de retardo; al terminar se vacía una cola de filtro
   de hasta 2,5 ms. No se cambia el reloj de reproducción de CORE.
4. `set_expression_sequence` usa el mismo validador y transporte de ojos. Se
   declara BLOCKING para preparar el plan antes del audio. El resultado de la
   herramienta conserva la identidad de respuesta; no dispara `response.create`.
   Se rechazan planes tardíos, repetidos o inválidos. Su entrega sigue ligada al
   primer paquete Opus y a las capacidades negociadas con CORE.
5. `turnComplete` termina la respuesta; `generationComplete` por sí solo no
   produce END. Una interrupción invalida audio y ojos pendientes. El siguiente
   turno espera la barrera de confirmación de CORE.

Diferencias y límites de esta integración:

- La persona de PIUU se comparte con xAI; `BridgeConfig.gemini_prompt` permite
  reemplazarla. La configuración no incluye thinking, affective dialogue ni una
  desactivación de proactive audio. El modelo probado por contrato es
  `gemini-3.8-live`, no la variante `extended-thinking`.
- La interrupción por voz usa el mecanismo nativo de Gemini. No hay truncado
  remoto exacto por los milisegundos realmente reproducidos por CORE: el log
  `GEMINI_NATIVE_INTERRUPTION` informa `playback_history_truncated=False`.
  El historial del proveedor puede incluir audio que CORE todavía no reprodujo.
- Un INTERRUPT manual de CORE cierra la conversación de forma segura y requiere
  un nuevo START. No se simula una cancelación mediante texto ni se reutiliza un
  historial incierto. CANCEL conserva el cierre normal de conversación.
- Ante `goAway`, desconexión o error se cierra la conversación y se permite un
  nuevo START. Todavía no hay reanudación de sesión ni reconexión transparente:
  no se reenvían audio, herramientas ni respuestas de resultado incierto.
- Las herramientas de búsqueda web de xAI no se trasladan a Gemini. Esta
  integración habilita únicamente la herramienta visual.
- Las pruebas son offline. Falta validar credenciales/acceso, voz, latencia,
  interrupciones y ojos con Gemini real y el hardware. El ejemplo adjunto no
  constituye una garantía de duración de sesión o de cuota gratuita.

Diagnóstico: `BOOT provider=gemini`, `GEMINI_READY`, `GEMINI_FIRST_AUDIO`,
`GEMINI_USAGE`, `GEMINI_NATIVE_INTERRUPTION`, `GEMINI_GO_AWAY` y
`PROVIDER_DISCONNECTED provider=gemini`. Las transcripciones tienen etiquetas
`GEMINI_USER` y `GEMINI_ASSISTANT`. Mantener las precauciones habituales al
compartir grabaciones y transcripciones.

Pruebas sin conexión ni consumo de API:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Para la prueba integrada: activar WALLE, pedir una respuesta corta, comprobar
`GEMINI_READY` y audio inteligible; pedir una respuesta expresiva y comprobar
`EXPRESSION_PREPARED` y entrega del plan; interrumpir hablando y verificar que
se corta el audio anterior y llega una nueva respuesta. No hacen falta logs USB
de CORE: usar los del servidor y EARS, junto con la reproducción y los ojos.

Referencias oficiales:
[modelo](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live),
[protocolo Live](https://ai.google.dev/api/live),
[herramientas](https://ai.google.dev/gemini-api/docs/live-api/tools).


Backend Python para PCM16LE mono a **16 kHz** desde CORE y respuestas Opus de
20 ms hacia CORE. El micrófono entra a CORE desde C3_EARS por UART; el backend
no depende de ese driver. Bluetooth queda fuera de esta etapa.

## Ejecutar

Requiere Python 3.10+, `websockets` y la biblioteca del sistema `libopus`.
Desde esta carpeta, con `XAI_API_KEY` en el entorno:

```sh
.venv/bin/python src/main.py
```

Configuración en `src/bridge/config.py`: puerto UDP 9999, audio 16 kHz,
voz/prompt, codec, grabaciones y límites de entrada. No colocar claves en el código.
La sesión xAI usa server_vad. Solo se conecta al abrir una conversación y se
desconecta al cerrarla; no existe retención de cinco minutos.

## Transporte actual

**Actualizar CORE y todo el árbol src/ del backend juntos.** Se exige UDP v3
PIU3; los sobres v1/v2 se rechazan. El cuerpo audio conserva su formato anterior.

- Conversación continua, PCM también durante las respuestas y IDs por respuesta.
- START/CANCEL/INTERRUPT/PLAYED y eventos del servidor con ACK idempotentes.
- xAI abre bajo demanda; búfer inicial 300 frames y timeout de apertura 6 segundos.
- CORE cierra al vencer `CONVERSATION_IDLE_TIMEOUT_MS` (10 segundos en la configuración actual); touch cierra.
- El transporte UDP puede renegociarse; la conexión al proveedor requiere otro START y no retiene historial después de un cierre.
- Se mantienen recuperación PCM acotada, cola normal de 16 frames y silencio
  de 20 ms por frame irrecuperable, hasta 200 ms por hueco.

HELLO/PONG indica disponibilidad del backend para admitir una conversación.
READY confirma posteriormente que xAI ya configuró su sesión. Un único CORE
por ahora; cada ContinuousSession posee sus recursos y admite una futura registry.

Ver [contrato UDP v3 y responsabilidades](PROTOCOLO_UDP_V3.md).
Los módulos managed_session/session/session_protocol (v2 y legacy) quedan como
referencia y regresión; el endpoint activo usa continuous_session.

## Desconexión de CORE e interrupciones

`core_disconnect_timeout_sec` en `src/bridge/config.py` controla cuánto tiempo
puede permanecer abierta una conversación sin mensajes válidos del CORE admitido.
El valor inicial es **10 segundos** y se comprueba cada segundo mediante reloj
monotónico. HELLO del cliente admitido, PING válido, PCM válido y controles/ACK
válidos actualizan la presencia; actividad de xAI, paquetes malformados y otros
clientes no la renuevan. El límite es independiente del cierre por silencio de
`CONVERSATION_IDLE_TIMEOUT_MS` que decide CORE. Los ACK de eventos también pueden provocar un
cierre antes si agotan sus reintentos.

Al vencer se registra `CORE_DISCONNECTED` y se cierra la conversación: se cancela
entrada, reproducción y reparación pendiente; se descartan buffers y se cierra
el WebSocket xAI, cuyo cierre tiene un máximo de 2 segundos antes de abortar el
transporte. CORE debe abrir explícitamente otra conversación para volver a usar
xAI; no se conserva contexto ni se reconecta al proveedor en segundo plano.

`InterruptionCoordinator` unifica wake manual y VAD por ID de respuesta. CORE
confirma la posición reproducida; `VadXai` confirma recorte/eliminación de historial.
Una pausa de coordinación (`INPUT_PAUSED`/`INPUT_RESUMED`) no consume el timeout
real de envío de audio. Se conserva la cola limitada y el descarte al llenarse.

Los plazos son independientes y configurables: `history_repair_timeout_sec`
(4 s para toda la transacción), `history_item_timeout_sec` (1 s para conocer el
mensaje interrumpido) y `history_confirmation_timeout_sec` (1 s por confirmación).
Con un mensaje identificado se intenta reparar sin exigir antes `response.done`.
Sin mensaje se espera su llegada o el fin de respuesta. Si vence ese primer plazo
para una interrupción VAD, se envía `response.cancel` con el ID de esa respuesta
y se espera como máximo otro `history_item_timeout_sec`, dentro del límite total
de 4 s. Una cancelación manual no repite ese envío. El fin sin mensajes confirma
un historial vacío; si el evento final incluye mensajes, también se reparan.
Una interrupción confirmada conserva la conversación, el contexto escuchado y
la entrada nueva; no envía `input_audio_buffer.clear`. La falta de confirmación
sigue cerrando la sesión por historial incierto.
Los mensajes no se retiran del registro antes de confirmar su reparación.

Una respuesta nueva antes de confirmar el historial, o un mensaje nuevo tardío
de una respuesta ya reparada, cierra la sesión con motivo explícito; bloquear
solo su reproducción no garantiza que se haya generado con el contexto correcto.
Logs útiles: `INTERRUPT_BEGIN`, `INTERRUPT_MERGED`, `INTERRUPT_PLAYBACK_ACK`,
`INTERRUPT_COMPLETE`, `XAI_CANCEL_FALLBACK`, `HISTORY_EMPTY_CONFIRMED`,
`HISTORY_REPAIR_FAILED` (fase y tiempo),
`RESPONSE_DURING_HISTORY_REPAIR`, `HISTORY_LATE_ITEM`, `CONVERSATION_CLOSED`.

## Próxima evolución: modelos y proveedores seleccionables

**Propuesta pendiente; no implementada.** El objetivo cercano es seleccionar
modelos de distintos proveedores desde CORE, inicialmente mediante una definición
en tiempo de compilación, sin añadir UI. El backend actual continúa usando xAI.

CORE indicaría un perfil lógico al abrir la conversación, por ejemplo:

```cpp
#define AI_PROFILE AI_PROFILE_OPENAI_REALTIME_MINI
```

El backend resolvería ese perfil mediante un registro explícito de proveedor,
modelo y configuración. Ejemplos iniciales: `XAI_VOICE` → xAI /
`grok-voice-latest`, y `OPENAI_REALTIME_MINI` → OpenAI /
`gpt-realtime-2.1-mini`. Las claves, URL, voces, prompt y opciones específicas
permanecerían en el servidor. Un perfil desconocido se rechazaría explícitamente;
la selección quedaría fija durante cada conversación.

Cambios previstos:

1. Extraer una interfaz de proveedor para conectar, enviar PCM, emitir eventos
   normalizados de voz/respuesta, interrumpir, recortar historial y cerrar.
   `ContinuousSession` instancia hoy `VadXai` y accede a sus estructuras internas
   de historial: hay que sustituir ese acoplamiento por operaciones de la interfaz.
2. Mantener transporte UDP, IDs/ACK, recuperación por NACK, silencio insertado,
   estados de conversación y codificación Opus independientes del proveedor.
3. Incorporar un adaptador OpenAI con autenticación, configuración de sesión,
   voces, eventos, métricas de uso y manejo de errores propios. La compatibilidad
   conceptual de VAD y recorte no implica que ambos proveedores tengan esquemas
   ni garantías idénticos; deben verificarse cancelación y confirmaciones.
4. Extender de forma coordinada CORE/backend el contrato de apertura para enviar
   el perfil. Actualmente START exige cuerpo vacío; definir su formato y política
   de compatibilidad antes de cambiarlo.
5. Añadir pruebas por adaptador y de selección/rechazo de perfiles, remuestreo
   continuo, interrupciones, audio tardío y recorte del historial.

Para `gpt-realtime-2.1-mini`, OpenAI Realtime admite WebSocket y requiere PCM
mono a **24 kHz**. La propuesta conserva EARS, CORE y UDP a **16 kHz**, con
remuestreo exclusivamente en Python:

```text
CORE → PCM 16 kHz → remuestreo 24 kHz → proveedor OpenAI
CORE ← Opus 16 kHz ← remuestreo 16 kHz ← proveedor OpenAI
```

El remuestreador debe conservar estado entre bloques, aplicar filtrado adecuado
y reiniciarse al cambiar conversación o descartar una respuesta. Las duraciones
usadas para recortar historial deben calcularse con la frecuencia correspondiente,
sin asumir que todo PCM del proveedor tiene 16 kHz. Esto evita añadir procesamiento
al ESP32 y conserva el contrato de audio con CORE.

Orden recomendado: interfaz de proveedor, adaptador OpenAI/remuestreo y pruebas,
y finalmente selección desde CORE. Bluetooth y UI siguen fuera de este trabajo.
Verificar disponibilidad y requisitos del modelo al implementar.

Referencias oficiales:
- [gpt-realtime-2.1-mini](https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini).
- [Realtime: formatos, VAD y recorte del historial](https://developers.openai.com/api/reference/resources/realtime/client-events).
- [Conexión WebSocket](https://developers.openai.com/api/docs/guides/realtime-websocket).

## Pacing de salida y redes variables

La salida Opus usa `AudioPacer` (`src/bridge/audio/pacing.py`) con plazos absolutos del
reloj monotónico `asyncio.loop.time()`. Codificación y preparación UDP quedan
incluidas en el intervalo nominal de 20 ms: no se añade una espera fija de 20 ms
tras cada frame. Esto evita la deriva observada de unos 45 frames/s frente a los
50 frames/s que consume CORE.

El estado del pacer se reinicia por respuesta/cancelación. Antes de enviar un
frame después de una espera se vuelve a validar su identidad y el pacer vigente.
Un retraso mayor que `output_pacing_max_lag_sec=0.04` reajusta el calendario;
los pequeños atrasos se compensan con separación mínima de medio intervalo
(10 ms por defecto). No se salta audio ni se retransmite Opus, ni se descarga de
inmediato toda la deuda de tiempo en una ráfaga. Una fuente o CPU sostenidamente
más lenta que tiempo real aún puede causar cortes: el pacer no crea audio faltante.

`OUTPUT_PACING_STATS` cada cinco segundos y `OUTPUT_PACING` al finalizar informan
frames, span_ms, fps, rebases y max_late_ms. Son métricas de envíos locales, no de
llegada a CORE. Con audio disponible y procesamiento inferior a 20 ms se espera
aproximadamente 50 fps. El backend sigue usando las colas limitadas existentes.

La red se trata como variable, con jitter, reordenamiento y pérdidas aisladas o
en ráfaga. Se conservan el búfer inicial de CORE de 1 s y rebuffer de 800 ms;
aumentarlos no corrige deriva de envío y añade latencia. El heartbeat y el fin
explícito por END permanecen separados del pacing; perder END puede terminar en
cancelación por stall en CORE, no en una falsa reproducción completa.

La recuperación uplink usa una caché en CORE reservada una sola vez al iniciar de 32 frames (640 ms de
PCM, TTL 640 ms desde la captura), reordenamiento normal de 40 ms y una espera
máxima de recuperación de 400 ms desde la solicitud NACK. Los valores se ajustan
en `SessionClient::CACHE_FRAMES/CACHE_TTL_MS` y `BridgeConfig`. Cada frame admite
un único intento de retransmisión; esto no cubre interrupciones de varios segundos.
La caché aumenta unos 13,1 KiB respecto de los 12 frames anteriores y también
retiene audio mientras llega START_ACK. Una caché circular puede sobrescribir un
frame antes del TTL si ingresan ráfagas de audio.

El backend separa los 32 frames de reordenamiento (`input_reorder_frames`) de los
16 de admisión al envío xAI. Reserva una posición adicional para el frame que
falta al principio del hueco. Solicita frames ausentes aunque haya audio en cola,
excluyendo los ya recibidos o encolados. La espera adicional afecta solo a huecos;
el PCM contiguo se envía de inmediato. Sin recuperación sigue insertando hasta
200 ms de silencio por hueco y descarta audio ante saturación de la cola xAI.

Para evaluar hardware: `V3_STATS` incluye `retx` (envíos exitosos), `retx_fail`,
`cache_miss`, `cache_expired` (subconjunto de misses) y `nack_age_max_ms` (máximo
para frames encontrados en caché, incluidos vencidos). `INPUT_STATS.nack_recovered`
cuenta frames solicitados que llegaron a tiempo al reordenador; puede incluir un
original retrasado y no demuestra por sí solo que la copia retransmitida llegó.
Los contadores de CORE son acumulados desde el arranque; los del backend pertenecen
a la entrada de la conversación. Comparar deltas, `missing` y `silence_frames`.
La prueba cruzada offline usa el cliente C++ real y el trabajador Python con
300 ms de RTT simulado; aún se requiere validar RAM y recuperación en hardware.

Las pruebas offline incluyen 30 s de pacing con procesamiento simulado de 2 ms,
Opus real con reloj simulado, pausas y atrasos del scheduler, cancelación durante
la espera, jitter de 20–140 ms, reordenamiento, pérdidas aisladas/en ráfaga y
ACK retrasado con primera transmisión perdida. Las ventanas finitas pueden
descartar paquetes tardíos: no se garantiza audio continuo bajo cualquier red.
Estas simulaciones no ejecutan la reproducción física de CORE ni sustituyen la
prueba integrada en Wi-Fi real.

## Logs para pruebas integradas

El entrypoint habilita logs INFO con fecha UTC, IDs CORE y response_id de xAI.
Incluyen ACK/reintentos observados, cancelación/recorte confirmado, colas y pérdidas.
No imprimen frames de audio ni transcripciones. Resúmenes cada cinco segundos.
Ver [captura e interpretación](DIAGNOSTICOS_TEST_INTEGRADO.md).

## Verificación sin hardware

```sh
.venv/bin/python -m unittest discover -s tests -v
```

64 pruebas locales (incluyen v3 y regresiones legacy): serialización, codecs/grabaciones, colas/reordenamiento,
controles duplicados, aislamiento de IDs, recuperación, silencio limitado,
cancelación y confirmaciones de recorte con un WebSocket simulado. No usan
credenciales, sockets UDP ni sesiones xAI reales.

No se cargó firmware ni se conectó a xAI. Faltan pruebas en placa de radio,
latencia, DMA y recursos, y validación del recorte con el proveedor. La cola PCM
de salida v3 está limitada a 128 chunks; saturación y consumo de memoria deben
evaluarse con respuestas reales. No hay retransmisión de Opus.

Las grabaciones WAV opcionales representan audio aceptado/procesado por el
bridge; incluyen silencio insertado en uplink. No miden lo escuchado físicamente.

## Startup audio backpressure

The startup transfer counts both queued PCM and frames held for reordering before
admitting more audio to the normal 16-frame pipeline. A missing sequence must not
cause the retained startup audio to flood the reorder buffer. The existing recovery
and silence deadlines still apply. `STARTUP_TRANSFER_COMPLETE` reports remaining
buffered frames, normal pipeline drops and startup drops separately.

Audio still uses 20 ms frames and individual provider appends. Provider batching
(e.g. 80 ms) is a separate possible optimization; it does not define a minimum
speech duration or change server VAD endpointing/interruption policy.

### Interrupciones tempranas y errores del proveedor

Una pausa natural puede hacer que server_vad empiece una respuesta y la cancele
cuando continúa la grabación. El cambio Thinking → Listening representa esa
interrupción; no debe cerrar por sí solo la conversación.

Las operaciones de cancelación y edición del historial llevan `event_id` para
correlacionar rechazos con su transacción. `XAI_ERROR` registra código, mensaje
(hasta 512 caracteres), parámetro, operación y método de correlación. Si el
proveedor no devuelve el ID solicitado, se puede atribuir el error a una única
operación pendiente. Un error de operación identificado despierta la transacción
inmediatamente; el receptor WebSocket continúa leyendo eventos y confirmaciones.

La respuesta explícita «no hay respuesta activa» a `response.cancel`, y «el item
no existe» a `conversation.item.delete`, satisfacen esas operaciones idempotentes.
`XAI_OPERATION_RECOVERED` registra estos casos; se retoma la entrada sin limpiar
el nuevo audio del usuario. No se interpreta un `invalid_request_error` genérico
como éxito. Los rechazos ambiguos y plazos sin confirmación conservan el cierre
por `history_uncertain`, evitando continuar con historial sin verificar.

Los límites de reparación y la política de descarte ante saturación no cambian.
Las pruebas offline cubren ambos resultados recuperables, continuación con una
nueva respuesta, errores no recuperables y supervivencia del receptor. En el VPS
se requiere una nueva captura para conocer el mensaje exacto de los rechazos
anteriores y comprobar que corresponden a los casos manejados.
Referencia: https://docs.x.ai/developers/rest-api-reference/inference/voice

## Bounded downlink recovery and reliable audio END

Deploy this backend and the matching CORE firmware together. UDP v3 HELLO_ACK
advertises downlink recovery with flag `0x02` (flag `0x01` retains admission
readiness). CORE refuses a server without this capability. New envelope kinds:
`DOWNLINK_NACK=22`, `AUDIO_END=23`, `AUDIO_END_ACK=24`. Legacy v1/v2 and the
reference managed-session implementation are unchanged.

`DownlinkRecovery` owns only the bounded resend cache and END receipt tracking.
The audio pipeline owns encoding/pacing; `ContinuousSession` validates identity,
services recovery every 20 ms and retains lifecycle/cancellation ownership.
NACK carries a big-endian uint32 base and uint16 mask (eight valid bits); the
header sequence must equal its base. All recovery traffic is scoped to the
active client/server/conversation/response. Cancel/reset clears pending work.

Defaults: 128 cached Opus frames, original-send TTL 3 seconds, one retransmit
attempt per frame, at most two retransmissions per 20 ms cycle, pending queue
32 entries. CORE retains its current jitter buffer; it waits 40 ms for reordering
and requests only useful future frames (at least 160 ms lead during normal
playback). Late recovery falls back to existing FEC/PLC. A lost NACK is not retried.
Cache RAM is on the server; CORE adds bounded sequence metadata, not audio copies.

AUDIO_END carries the existing Opus END body and exclusive final sequence.
The server resends it every 150 ms until AUDIO_END_ACK. Receipt does not mean
playback completion: PLAYED/PLAYED_ACK still confirms completion after estimated
DMA drain. END retries expire after 5 seconds with explicit
`output_end_unconfirmed`, never a fabricated completion. CORE allows 400 ms for
missing tail recovery before concealment through the known boundary. Without
END, a stalled stream remains a failure. Heartbeat limits are unchanged.

Tune server bounds in `BridgeConfig`: `output_cache_frames`,
`output_cache_ttl_sec`, `output_end_retry_sec`, `output_end_timeout_sec`.
`OUTPUT_RECOVERY_STATS` and `OUTPUT_RECOVERY_COMPLETE` report requested frames,
retransmissions, cache misses/expiry, repeated requests, queue saturation, local
send failures and END sends/ACKs. Correlate with CORE `DOWNLINK_RECOVERY`,
`OUTPUT_END_ACK`, `OUTPUT_END_TIMEOUT`, and `PLAYBACK_FINISHED`.

Offline tests cover bounded retransmission, cache age/eviction, stale identities,
loss/retry of END and its ACK, cancellation, tail grace and real Opus pipeline
integration. Hardware validation still needs packet loss/reordering tests and
confirmation that playback completes when the first END or ACK is lost.


### Provider input retention during history repair

Continuous VAD input now has an independent ordered PCM sender. UDP admission,
reordering, uplink NACK and bounded silence insertion continue while the provider
adapter pauses appends to repair interrupted assistant history. A valid interruption
preserves retained user audio; closing or replacing the conversation cancels the
sender and clears its pending frames before another conversation starts.

`BridgeConfig` controls the bounds:

- `provider_input_buffer_sec = 5.0`: 250 pending 20 ms PCM frames at 16 kHz mono
  signed 16-bit, up to 160,000 payload bytes on the **backend**, plus Python metadata
  and one in-flight frame. CORE RAM and jitter settings are unchanged.
- `provider_input_max_age_sec = 5.0`: expire pending frames after five seconds from
  admission to this queue, using a monotonic clock. Expiry is checked on admission
  and after the provider repair barrier, before sending.
- `provider_input_send_interval_sec = 0.01`: space appends by at least 10 ms after
  each send completes, allowing catch-up at no more than twice real time. Normal
  20 ms arrivals are sent immediately when the provider is available; there is no
  requirement to accumulate five seconds and no unlimited catch-up burst.

When capacity is exhausted, discard new frames and retain the existing ordered
prefix. Expired or rejected frames are counted explicitly; no extra silence is
inserted for these policy drops. This queue does not change the 200 ms uplink gap
silence cap or the four-second history repair deadline. Startup transfer respects
both ingress and provider queue capacity instead of flooding them.

`PROVIDER_INPUT_STATS` (periodic) and `PROVIDER_INPUT_CLOSED` (final) report queued
frames/ms, oldest age, peak occupancy, sent frames, capacity drops, expiry, cleared
frames, send failures and maximum send age. `INPUT_STATS` still describes UDP ingress
and recovery; a transport-recovered packet can separately expire before provider
sending. Input WAV recordings contain audio actually handed to the provider sender,
not every retained frame. These counters are always enabled with existing backend
event logging. Catch-up behavior and voice quality still require an integrated test.

Deploy this change by restarting the backend only; no firmware upload is required.


### Backend module layout

```text
src/
├── main.py                         # Stable entrypoint and legacy class exports
└── bridge/
    ├── app.py                      # Logging, configuration and endpoint startup
    ├── config.py                   # Shared settings
    ├── transport/
    │   ├── udp.py                  # Endpoint routing
    │   ├── ingress.py              # Uplink admission, reordering and NACK
    │   ├── downlink_recovery.py    # Opus resend cache and reliable audio END
    │   ├── reliable_events.py      # Event delivery and ACK tracking
    │   └── protocols/              # Audio, conversation v3 and session v2 formats
    ├── conversation/
    │   ├── session.py              # Current v3 coordinator
    │   ├── state.py                # Conversation state and response identity
    │   └── interruption.py         # Cancellation and history repair lifecycle
    ├── audio/
    │   ├── pipeline.py             # PCM/Opus output pipeline
    │   ├── codec.py                # libopus binding
    │   ├── pacing.py               # Audio transmission timing
    │   ├── provider_input.py       # Bounded provider input retention and sending
    │   ├── startup_audio.py        # Audio retained during provider startup
    │   └── recording.py            # WAV recording
    ├── providers/
    │   ├── operation_errors.py     # Provider operation error classification
    │   └── xai/
    │       ├── realtime.py         # WebSocket connection and basic API events
    │       ├── tagged.py           # Response identity and history operations
    │       └── vad.py              # Continuous VAD and history repair barrier
    ├── diagnostics/logging.py      # Structured events and logging setup
    └── legacy/                    # Reference v1/v2 coordinators and their tests
```

The current path is `app → transport.udp → conversation.session`; the coordinator
connects transport, audio and provider components. Protocol contracts contain no
provider logic. Package initializers are deliberately lightweight: import classes
from their defining modules, without starting connections or tasks on import.

`python src/main.py` and the public class exports from `main` remain available.
Internal `bridge.*` import paths have moved to the locations above; update custom
scripts that import them directly. No old-path shim modules are retained.
This reorganization changes no wire formats, timing, buffer limits or provider
behavior, and does not yet add a generic multi-provider interface. Existing offline
tests retain their names and now import the modules by responsibility.


## PIUU expression sequences and optional UDP delivery

The continuous UDP v3 provider now exposes `set_expression_sequence` through xAI
custom function calling. The backend validates a plan, preserves its response
identity across tool continuation, and sends an atomic binary plan when CORE
advertises eye-plan support. Existing CORE audio remains compatible; clients
without that capability receive no new packets. No MCP or serial writer is used.

PIUU_CORE now implements kind-26/27 negotiation, receipt ACKs and a scheduler
using an estimated consumed-I2S-sample clock. Hardware validation remains pending. See
[PIUU eye-plan wire contract](PROTOCOLO_EYES_V1.md) for capability negotiation,
exact bytes/IDs, ACK/retry rules and CORE scheduling behavior. No firmware or deployment
change is included here. Preview logs are not proof of CYD presentation.

### Personality and prompt

`BridgeConfig.xai_prompt` defines PIUU as a warm, curious little bird speaking
concise Rioplatense Spanish. Questions are encouraged only when useful; it does
not force every answer to end in a question. Tool-specific guidance lives in
`bridge/expressions/plan.py` and is appended only when the continuous adapter
advertises the tool. Set `expression_tools_enabled=False` in `BridgeConfig` to
disable the tool without removing the personality prompt or built-in web search.
Legacy v1/v2 providers do not advertise this function.

The policy requests at most one sequence before audio when visual expression
adds value, omits it for short confirmations, and continues normally if validation
fails. It does not ask the model to predict exact audio duration or speak control
commands. Routine blinking/gaze remains a device-side responsibility.

### Contract

```json
{
  "sequence": [
    {"emotion": "happy", "gesture": "soft_emphasis", "duration_ms": 2400},
    {"emotion": "curious", "gesture": "question", "duration_ms": 1800}
  ]
}
```

- Exactly one top-level `sequence` array with 1–3 steps; no extra properties.
- Every step requires `emotion`, `gesture` and `duration_ms`.
- Emotions: neutral, happy, curious, calm, concerned, sad, excited.
- Gestures: none, soft_emphasis, emphasis, question, acknowledge, glance, blink,
  doubleblink, sway.
- Duration is an integer from 1200 to 4000 ms (maximum plan 12000 ms). Booleans,
  floats, numeric strings, malformed JSON and oversized arguments are rejected,
  not silently clamped. Arguments are limited to 4096 UTF-8 bytes.
- Offsets are cumulative, relative to the first played audio sample. Gesture is
  a one-shot request at a step boundary, not a command for every audio packet.
- A short plan falls back to neutral; a short answer cuts off the remaining plan.
  Approximate durations do not align gestures to particular words/questions.

Tool results report `status: prepared`, never claim the eyes have already moved.
The schema and validation constants are shared in `bridge/expressions/plan.py`.

### Tool response lifecycle

1. The first xAI `response.created` obtains the ordinary v3 response identity.
2. Function calls are collected until `response.done`, including calls present
   only in its terminal output snapshot. Duplicate call IDs are ignored. A missing
   response ID is resolved only through an explicit earlier function-item/call-ID
   association, never by guessing the only active response.
3. The server validates every call and returns all `function_call_output` items.
   At most one plan is accepted. Invalid/extra/unsupported calls receive errors.
4. Exactly one `response.create` continues a successful tool-only response after
   all outputs are sent. A tool-only response does not enqueue an empty audio END.
5. The continuation reuses the same v3 identity and output gate. It does not reset
   audio playback or cancel its own plan. The first valid audio delta publishes
   the prepared plan to the conversation coordinator.
6. Existing output pacing, Opus encoding, END/recovery and playback acknowledgments
   continue normally. A tool arriving after audio began is rejected and causes no
   extra continuation.

One tool-only round is allowed per spoken response. A repeated tool-only round
fails the provider session rather than looping indefinitely. This is a protocol
violation guard, not ordinary argument validation. A continuation arriving after
its identity was invalidated is also rejected; the existing provider recovery
closes the uncertain conversation rather than associating old metadata with a new
user turn. Both behaviors need observation in live xAI testing.

VAD interruption, manual history repair, output reset and provider close discard
pending expression data. The original late-item guard remains intact. Function
call traces are not counted as audible assistant history; existing truncation
continues to operate on assistant audio items. Calls/plans/duplicate caches are
bounded and there is no expression task with wall-clock sleeps.

### Diagnostic preview and UDP transport

- `EXPRESSION_PREPARED`: valid plan received from the model.
- `EXPRESSION_REJECTED`: bounded reason code, without logging raw arguments.
- `EXPRESSION_PLAN`: plan bound to the response identity, with offsets and total.
- `EXPRESSION_PREVIEW clock=opus_send`: current visual step at the outgoing media
  position; emitted only on step changes, not per frame. Skipped media positions
  skip expired gestures rather than replaying a burst.
- `EXPRESSION_RESET`: outgoing audio ended, response reset, cancellation or close.

`ExpressionTimeline` remains a backend preview, not a hardware clock. The separate
`EyePlanDelivery` sends one complete plan immediately before the first Opus frame,
after `RESPONSE_STARTED` is acknowledged. The plan carries cumulative offsets from
playback start, so CORE can execute it despite network/jitter-buffer delay.

`BridgeConfig.expression_udp_enabled=True` enables negotiation (HELLO offer and
HELLO_ACK bit `0x0004`). Existing clients offering zero keep the previous flags
and audio behavior. With no negotiated support, only the previous preview runs.
A client must keep its capability offer stable until it creates a new session nonce.

The packet is at most **66 bytes including the v3 envelope**, with 1–3 combined
emotion/gesture steps. ACK confirms receipt only. Delivery uses at most three
attempts, at least 100 ms apart, and expires after 500 ms. Neither a missing ACK
nor a local visual send error blocks/fails audio. The existing recovery loop polls
retries; there is no per-expression task or wall-clock animation loop.

`EYES_PLAN_SENT`, `EYES_PLAN_ACK`, `EYES_PLAN_UNCONFIRMED`, `EYES_PLAN_SEND_ERROR`
and `EYES_PLAN_STATS` expose delivery. A short audio response ends its visual plan
at actual playback completion; Opus send completion does not send a premature
neutral command. Cancellation/reset/close discards retries and rejects stale ACKs.

Durations include transitions and define a continuous playback timeline. CORE must
not translate them to legacy `emotion NAME duration` holds that start after each
transition. Exact duplicate plans must be ACKed without restarting any gesture;
late plans skip expired gestures. See the wire contract for the full scheduler and
EARS/automatic-gesture arbitration requirements.

### Review of the supplied prototype

Kept the function name, emotion/gesture vocabulary, short sequences and separation
of voice from eye controls. Replaced the provider-owned `eye_writer`/sleep task,
which started before audio, outlived playback and was cancelled by the next
`response.created`. Validation now rejects malformed values instead of reporting
`ok` before a background task fails. Parallel tool calls are completed before one
continuation; stale calls cannot start new gestures. The existing VAD late-event
return was preserved: moving it inside the unknown-item condition would allow
known stale audio to rebuild local state.

### Validation

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
The expression tests cover bounds/types, schema consistency, timeline expiration,
parallel and duplicate calls, terminal-only calls, missing-ID correlation, late
calls, cancellation, stale continuations, disabled tools and the real v3 Opus
pipeline. `test_eye_transport.py` adds wire vectors, negotiation, retries, ACK
validation, tool-to-wire integration and nonblocking audio. No live provider
request, paid voice session or device write is part of
the offline tests. Before deployment, measure first-audio latency, validate event
ordering with the current xAI model and exercise interruptions around tool calls.

References checked 2026-10-26:
- https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech
- https://docs.x.ai/developers/rest-api-reference/inference/voice

## Gmail MCP propio (2026-10-29)

Integración opcional **PIUU → Gemini Live → Tool Manager → Gmail MCP → Gmail API**.
El servidor MCP propio corre en un subproceso por stdio. El Tool Manager descubre
las herramientas y las expone a Gemini con el prefijo `gmail_`:

- `gmail_search_messages`: búsqueda con sintaxis Gmail, metadatos y paginación.
- `gmail_read_message`: lectura del cuerpo de texto plano y metadatos.
- `gmail_send_message(to, subject, body)`: envío de texto plano a una dirección.

Las herramientas Gmail usan `NON_BLOCKING` y resultados `WHEN_IDLE`; las consultas
pendientes sobreviven al cierre de un turno de audio. La herramienta de
expresiones conserva `BLOCKING`. Los envíos no se reintentan automáticamente:
un timeout puede dejar el resultado incierto y debe revisarse Enviados antes
de solicitar otro envío.

Tras instalar `src/requirements-mcp.txt`, habilitar Gmail API en Google Cloud y
guardar las credenciales OAuth de tipo Desktop app en `.secrets/gmail-client.json`:

```bash
PYTHONPATH=src .venv/bin/python -m bridge.mcp_servers.gmail --auth
export PIUU_VOICE_PROVIDER=gemini
export PIUU_GMAIL_MCP_ENABLED=1
.venv/bin/python src/main.py
```

El inicio requiere también `GEMINI_API_KEY`, como se describe en la sección de
Gemini Live. OAuth solicita `gmail.readonly` y `gmail.send`. Si había un token de
solo lectura, volver a ejecutar `--auth` para conceder el permiso de envío.
Gmail permanece desactivado por defecto; Calendar y Home MCP quedan pendientes.

Validación: **173 pruebas backend pasaron**, incluidos intercambios MCP reales por
stdio y pruebas de envío con API simulada. No se enviaron correos reales; falta
configurar las credenciales, autorizar la cuenta y probar Gmail en vivo.

Instalación, autorización, límites y ejemplo de envío por voz en
[GMAIL_MCP.md](GMAIL_MCP.md).
