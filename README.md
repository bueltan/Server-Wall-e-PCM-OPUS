# PIUU CORE ↔ xAI Voice bridge

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
- CORE informa silencio y cierra por 30 segundos de inactividad; touch cierra.
- No hay reconexión automática ni historial retenido después de un cierre.
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
30 segundos que decide CORE. Los ACK de eventos también pueden provocar un
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

La salida Opus usa `AudioPacer` (`src/bridge/pacing.py`) con plazos absolutos del
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
