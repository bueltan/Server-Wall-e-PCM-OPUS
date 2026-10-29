# Conversación continua y UDP v3

Estado: implementación local y pruebas sin hardware. Actualizar CORE y backend
juntos; v3 (`PIU3`, versión 3) rechaza sobres v1/v2. UART EARS/KEYWORD no cambia.
La frecuencia de todo el pipeline continúa en 16000 Hz, PCM16 mono.

## Comportamiento

- Wake o touch abre una conversación. CORE muestra Connecting mientras el backend
  abre xAI; no hay conexión xAI permanente cuando CORE está inactivo.
- El backend confirma START inmediatamente y conserva hasta 300 frames PCM
  (6 segundos / 197400 bytes de paquetes, más overhead Python) durante la apertura.
  Timeout de apertura xAI: 6 segundos. Saturación descarta nuevos frames y lo registra.
- Al confirmar xAI su configuración, el backend anuncia READY. CORE lo confirma;
  entonces el backend entrega el audio acumulado en orden al ingreso habitual.
- `server_vad` detecta inicio/fin de voz y genera respuestas automáticamente.
  No se envían COMMIT ni response.create en este flujo.
- PCM continúa durante procesamiento y reproducción para permitir interrupciones.
  SpeechStarted invalida la respuesta anterior, limpia reproducción y obtiene
  los milisegundos reproducidos de CORE. xAI cancela automáticamente por VAD;
  el backend recorta/elimina el historial no escuchado sin limpiar la nueva pregunta.
- Wake durante una respuesta interrumpe sin cerrar la conversación. Durante
  Connecting/Listening, wakes repetidos son idempotentes. Touch siempre cierra.
- CORE cierra por 30 segundos de silencio mientras espera una nueva intervención.
  No cierra por ese temporizador durante Connecting, WaitingResponse ni Playing,
  ni mientras xAI mantiene una intervención de voz abierta.
- Cerrar detiene UART hacia CORE, invalida audio local, confirma CANCEL y cierra
  el WebSocket xAI. No hay espera de cinco minutos, reconexión automática en idle
  ni conservación del historial entre aperturas. Reabrir usa una sesión nueva.
- Una pérdida de conexión aborta la conversación. Hace falta un nuevo wake/touch.

## Responsabilidades

| Módulo | Propietario / responsabilidad |
|---|---|
| CORE app/conversation | Estados, touch/wake, temporizador de cierre, eventos de reproducción |
| CORE audio/silence_monitor | Nivel medio absoluto PCM; no clasifica palabras ni modifica audio |
| CORE app/ears_session | Adaptación UART a entrada de conversación y observación de nivel |
| CORE network/session_client | Identidades, ACK, retransmisión, caché PCM; sin UI ni DSP |
| CORE audio/audio_downlink | Opus, jitter, reproducción, estimación DAC y epochs |
| backend conversation | Estado e identidades independientes del proveedor |
| backend continuous_session | Admisión UDP y coordinación del ciclo de conversación |
| backend reliable_events | Cola limitada de eventos confirmados y reintentos |
| backend startup_audio | Búfer limitado durante la apertura, sin red ni tareas propias |
| backend ingress | Orden, NACK, pérdidas y silencio limitado |
| backend audio_pipeline | Grabaciones, Opus y ritmo de salida |
| backend vad_xai | Traducción xAI, eventos VAD y recorte confirmado del historial |

Los módulos v1/v2 del backend quedan como referencia y regresión; la factory UDP
usa exclusivamente ContinuousSession. No se implementaron Bluetooth ni LVGL.

## Sobre binario

Se conserva el tamaño de encabezado de 38 bytes; enteros big-endian:
`!4sBBHQQIIIH`: magic, version, kind, flags, client_nonce u64, server_nonce u64,
conversation_id u32, response_id u32, sequence u32, payload_size u16.
En los tipos heredados el nombre de campo `turn` representa conversation_id.
El cuerpo audio conserva su encabezado de 18 bytes anterior. Máximo cuerpo: 658.

- PCM/MISSING/NACK usan response_id=0. La secuencia PCM no reinicia entre respuestas.
- Cada respuesta tiene response_id creciente, con secuencia Opus comenzando en 0.
- client/server nonce separan reinicios; IDs viejos no reactivan conversaciones.

| Kind | Valor | Cuerpo / semántica |
|---|---:|---|
| HELLO / HELLO_ACK | 1 / 2 | Admisión backend; no implica xAI conectado |
| START / START_ACK | 3 / 4 | Abrir conversación, cuerpo vacío; ACK flags bit0=aceptado |
| COMMIT / COMMIT_ACK | 5 / 6 | Reservados; no usados en v3 |
| CANCEL / CANCEL_ACK | 7 / 8 | Cerrar conversación; cuerpo played_ms u32 |
| PCM / OPUS | 9 / 10 | Cuerpo audio |
| NACK / MISSING | 11 / 12 | Recuperación PCM acotada, igual a v2 |
| ERROR | 13 | Abortar estado incierto |
| PING / PONG | 14 / 15 | Reachability/admisión backend |
| EVENT / EVENT_ACK | 16 / 17 | Evento servidor ordenado / ACK con played_ms u32 |
| INTERRUPT / INTERRUPT_ACK | 18 / 19 | Wake manual durante respuesta; played_ms u32 |
| PLAYED / PLAYED_ACK | 20 / 21 | Reproducción terminada; played_ms u32 |

EVENT: code u8 + utterance_id u32. Códigos: 1 SpeechStarted, 2 SpeechStopped,
3 ResponseStarted, 4 Closed reservado, 5 Ready. Sequence identifica el evento,
crece desde 1 por conversación. El servidor espera cada ACK antes del siguiente
control. Reintenta cada 150 ms hasta 8 envíos; al agotar, aborta la conversación.
CORE conserva 16 ACK recientes y devuelve el mismo played_ms en duplicados.
ResponseStarted se confirma solo cuando el reproductor procesó el nuevo epoch;
el backend retiene la salida hasta ese ACK. Opus no se retransmite.

Los controles de CORE reintentan cada 150 ms hasta 48 envíos; START/CANCEL,
INTERRUPT y PLAYED tienen efectos idempotentes. La caché de controles del backend
es de 64 entradas. Cola de controles: 8 backend / 4 pendientes CORE.
PCM mantiene caché CORE de 12 frames, TTL 240 ms, un reintento y presupuesto 2
frames por poll. Backend: orden 40 ms, recuperación 120 ms; después silencio
20 ms/frame perdido, máximo 200 ms por hueco. Cola normal 16 frames.

## Cierre por silencio

`CONVERSATION_IDLE_TIMEOUT_MS=30000` y `CONVERSATION_ACTIVITY_LEVEL=160` son
configurables como build_flags de CORE. El nivel es la media de |PCM16| de cada
frame (0..32768), calculada con enteros, sin tareas ni buffers nuevos de audio.
Se reinicia el plazo al observar nivel >= umbral, eventos de voz o fin de reproducción.
Esto mide actividad acústica, no distingue voz/ruido: el umbral requiere ajuste
con el micrófono real. Ruido/eco por encima del umbral pueden posponer el cierre.

`[CONV_STATS]` informa estado, nivel, quiet_ms, umbral y plazo cada cinco segundos.
`[CONV] silence_close` identifica la decisión de CORE. Correlacionar con
`CONTROL_ACK kind=CANCEL_ACK`, `CONVERSATION_CLOSED provider_closed=True` y
el siguiente `CONVERSATION_OPEN` / `PROVIDER_READY` del backend.

## Pruebas y límites

Pruebas offline: ACK duplicados/perdidos, frontera de epochs, audio continuo,
IDs de respuestas, interrupción por voz/wake, recorte confirmado sin borrar nueva
entrada, apertura diferida, búfer saturado, cierre y reapertura sin contexto,
rechazo v2, timeout/ausencia de reconexión y nivel/silencio con rollover de millis.
No utilizan credenciales ni conexiones xAI reales. Compilar no carga firmware.

Pendiente hardware: eco, umbral, latencia de apertura, reproducción/interrupción,
y cierre automático con audio real. xAI VAD puede interpretar eco como voz;
no hay AEC. La reproducción estimada sigue siendo conservadora para el DAC;
Bluetooth requerirá su propio reloj/estimador.

## Facturación

En server_vad xAI publica facturación por duración de sesión: US$0,08/minuto.
Los 30 segundos de espera activa pueden costar US$0,04. Se abre solo bajo demanda
para evitar sesiones facturables en reposo. La facturación efectiva debe verificarse
con el uso de xAI, no únicamente con nuestros temporizadores.
Se registra XAI_USAGE con billable_audio_seconds cuando el proveedor lo incluye.
Fuentes: https://docs.x.ai/developers/models/speech-to-speech y
https://docs.x.ai/developers/rest-api-reference/inference/voice.


## Extensión opcional de ojos

El backend implementa `EYES_PLAN=26` y `EYES_PLAN_ACK=27`, negociados mediante
bit `0x0004` en HELLO/HELLO_ACK. `EYES=25` queda reservado para los comandos
inmediatos de CORE; este emisor usa planes binarios y no envía ese tipo.
Los clientes actuales con HELLO flags=0 mantienen su handshake y audio.
El cuerpo máximo del plan es 28 bytes (66 con cabecera), con hasta tres pasos
atómicos de emoción/gesto y offsets de reproducción. La falta de ACK visual no
bloquea ni cancela audio. CORE ya implementa recepción/planificación y anuncia soporte; queda validar
la sincronía y el uso de recursos en hardware. Contrato completo y vectores: [PROTOCOLO_EYES_V1.md](PROTOCOLO_EYES_V1.md).

## Identidad estable de CORE y reconexión UDP

`CAP_DEVICE_MAC=0x0008` en HELLO indica un cuerpo de exactamente seis bytes:
la MAC WiFi STA de CORE, en orden de red (ejemplo `02 12 34 56 78 9a`). Se
rechazan MAC cero, multicast, cuerpos de otra longitud y cuerpos sin el bit.
HELLO conserva `server_nonce=turn=response=0`; `client_nonce` sigue siendo la
identidad variable de la sesión, no la MAC. HELLO_ACK no tiene cuerpo y confirma
el bit `0x0008`. El firmware que anuncia MAC exige esa confirmación.

El backend admite un dispositivo por proceso. El primer HELLO fija la MAC;
los siguientes deben declarar la misma. Esa declaración es la regla de admisión
actual, sin firma ni clave compartida: todavía no es autenticación criptográfica.
La asociación no se guarda en disco y se pierde al reiniciar el backend.

Un HELLO con la misma MAC y un `client_nonce` nuevo permite cambiar IP y/o puerto
sin esperar el timeout de ausencia. Si hay una conversación activa, primero se
cierra el proveedor y se limpian audio, reparaciones y eventos; CORE reintenta
HELLO hasta que termina ese cierre. Entonces el backend retira el nonce anterior,
genera otro `server_nonce`, cambia la dirección y reinicia la numeración de
conversaciones. Debe llegar un START nuevo; no se conserva historial del proveedor.
No se redirige la sesión por un PING, PCM ni HELLO con el mismo nonce desde otra
dirección. CORE renueva su nonce al perder el enlace, incluso estando inactivo
(6 s), para recuperar también un cambio de puerto NAT sin reiniciar el equipo.
Los últimos 32 nonces retirados se rechazan para evitar que HELLO retrasados
restauren una sesión anterior; los demás paquetes deben coincidir con la sesión
actual y su dirección.

Los clientes anteriores, con HELLO sin MAC, conservan el handshake restringido
al origen original. No pueden reemplazar un dispositivo ya asociado por MAC.
Actualizar primero PIUU_SERVER y después CORE. Diagnóstico: `CORE_ADMITTED`
identifica la primera admisión; `CORE_REBOUND` registra MAC y direcciones anterior
y nueva. La recepción de CORE reintenta el datagrama pendiente si falla el mutex
entre parse y read; descarta enteros los datagramas sobredimensionados o leídos
parcialmente. Esa corrección local no modifica el contrato UDP ni agrega buffers;
su validación durante microcortes WiFi reales sigue pendiente.
