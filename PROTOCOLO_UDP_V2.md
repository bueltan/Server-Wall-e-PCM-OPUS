# CORE ↔ servidor: UDP v2

Estado: implementado en ambos extremos; pruebas locales, sin carga de firmware
ni sesión xAI real. Requiere actualizar firmware y árbol `src/` del backend juntos.
No hay fallback a datagramas v1 sin sobre. EARS UART no cambia en esta etapa.

## Formato

Sobre `PIU2`, versión 2, big-endian, **38 bytes** (`!4sBBHQQIIIH`):

| Campo | Bytes | Uso |
| --- | ---: | --- |
| magic / version / kind | 4 / 1 / 1 | `PIU2`, 2, tipo de mensaje |
| flags | 2 | Bit 0: ACK exitoso o backend xAI listo |
| client_nonce / server_nonce | 8 / 8 | Identidad de la conexión en ambos extremos |
| turn / response | 4 / 4 | Turno monotónico; respuesta 1 dentro del turno |
| sequence | 4 | Secuencia de audio o ID de operación de control |
| body_size | 2 | Tamaño exacto del cuerpo, máximo 658 |

El nonce de CORE nace de `esp_random` y cambia también al reiniciar su transporte;
el servidor genera el suyo aleatoriamente. La identidad completa aísla turnos,
respuestas y reconexiones. No constituye autenticación ni cifrado.

| Kind | Valor | Cuerpo |
| --- | ---: | --- |
| HELLO / HELLO_ACK | 1 / 2 | Vacío; petición con server/turn/response=0 |
| START / START_ACK | 3 / 4 | Vacío |
| COMMIT / COMMIT_ACK | 5 / 6 | Petición: secuencia final y PTS final, dos u32 |
| CANCEL / CANCEL_ACK | 7 / 8 | Petición: reproducción estimada en ms, u32 |
| PCM / OPUS | 9 / 10 | Cuerpo de audio con encabezado de 18 bytes |
| NACK | 11 | base u32 + máscara u16; ocho bits bajos válidos |
| MISSING | 12 | Vacío; sequence identifica el frame ausente de caché |
| ERROR | 13 | Vacío; sesión/turno no utilizable |
| PING / PONG | 14 / 15 | Vacío; turn/response=0 |

El cuerpo de audio reutiliza la serialización `PCM!`/`OPUS` existente, con
versión interna 1 y encabezado de 18 bytes. Esa versión describe el **cuerpo**,
no habilita el transporte antiguo. PCM16LE mono, 16 kHz, 320 muestras/640 bytes
por frame de 20 ms. END lleva secuencia final sin muestras ni payload. La
secuencia exterior coincide con la interior. Tamaño máximo del datagrama: 696 bytes.
No confundir con `PCM!` UART de EARS: su encabezado tiene 10 bytes.

## Controles fiables y turnos

- HELLO establece sesión. El indicador listo requiere `session.updated` de xAI;
  PONG también informa esa disponibilidad, además del alcance del bridge.
- START debe ser confirmado antes de transmitir PCM retenido. CORE conserva
  hasta 12 frames de arranque; no bloquea el loop esperando el ACK.
- START, COMMIT y CANCEL usan ID de operación y ACK. CORE reintenta cada 150 ms,
  hasta ocho intentos. El servidor coalesce pendientes y conserva los últimos
  64 ACK. Repetir COMMIT no vuelve a solicitar una respuesta a xAI.
- COMMIT incluye el límite final para reconocer pérdidas del final aunque se
  pierda END. El backend sella la entrada antes de solicitar la respuesta.
- Durante Listening, touch/STOP termina entrada. Durante espera/reproducción,
  touch o wake cancela y abre otro turno. CORE invalida la identidad anterior
  inmediatamente; CANCEL precede al START nuevo. El backend invalida audio al
  admitir CANCEL, antes de esperar operaciones remotas.
- Al finalizar normalmente se libera el turno local. Una nueva entrada no se
  abre por la mera llegada de PCM: requiere START con identidad nueva.
- Cambiar configuración o reconectar renueva el nonce local. Desde el mismo
  endpoint el backend limpia primero el estado remoto anterior; después acepta
  HELLO nuevo. Cambiar IP/puerto requiere reiniciar el bridge por ahora.

## Pérdidas, NACK y silencio

CORE tiene un pool estático de 12 frames (~8 KiB), edad máxima 240 ms y un solo
reintento por frame. El backend solicita por NACK hasta ocho secuencias faltantes.
CORE vacía como máximo dos frames retenidos/reintentos por poll. Una solicitud
sin copia disponible responde MISSING; no extiende indefinidamente la espera.

El reorder tiene ventana base de 40 ms. Al pedir recuperación, espera hasta
120 ms para ese hueco. La ventana de cierre también es fija de 120 ms; duplicados
no la renuevan. Con cola de entrada muy ocupada se suprimen NACK para no amplificar
la congestión. Hay 16 posiciones de audio más dos reservadas para END/COMMIT.

**Política actual aprobada:** después de intentar recuperación, insertar 20 ms de
PCM cero por frame faltante, con **máximo 200 ms por hueco**. Los huecos mayores
se contabilizan completos, pero solo se rellenan sus primeros 200 ms. Un único
append acotado envía el relleno, que también queda en el WAV de entrada. El
silencio conserva parcialmente el tiempo; no recupera fonemas perdidos.

Al saturar se descartan frames nuevos y se conserva el turno. Un append WebSocket
fallido/ambiguo no se reintenta ni se rellena automáticamente: no sabemos si xAI
llegó a recibirlo. Los contadores distinguen pérdidas, relleno y errores de envío.
No hay ARQ de Opus de salida; permanecen jitter/FEC/PLC. Tampoco se recupera con
NACK audio que se perdió antes de que CORE lo empaquetara, incluido UART.

Ajustes CORE: constantes de `network/session_client.h`. Backend:
`input_queue_frames`, `input_reorder_window_sec`, `input_recovery_window_sec`,
`input_silence_max_ms`, `input_send_timeout_sec` en `bridge/config.py`.

## Cancelación y recorte de historial

Cada `response_id` de xAI se asocia con la identidad local. Los deltas y finales
viejos no pueden salir bajo la identidad nueva. CANCEL solicita detener la
respuesta, espera `response.done`, recorta el item parcialmente escuchado y
elimina los items del asistente todavía no escuchados. Espera confirmaciones de
truncate/delete y de `input_audio_buffer.clear` antes de confirmar CANCEL.

La posición enviada por CORE es conservadora: muestras entregadas al DAC menos
la capacidad total DMA (8×256 muestras = 128 ms), expresadas sobre la línea de
tiempo del audio. **No es una medición acústica**: puede quitar del historial
parte de lo ya escuchado. Al cambiar de época, el worker de reproducción detiene,
limpia y reinicia DMA. Al terminar normalmente espera el drenaje conservador del
DMA antes de informar Finished. Bluetooth requerirá otra estimación de su propia
cola/latencia detrás de `audio_output`.

Si una operación remota no se confirma, se cierra la conexión xAI; no se abre
una entrada contra un buffer remoto incierto. Una reconexión pierde el historial
remoto de esa conexión. Es un límite explícito de esta implementación.

Referencia de operaciones: [xAI Voice Realtime](https://docs.x.ai/developers/rest-api-reference/inference/voice).

## Validación y pendientes

Suite host CORE: codec v2 C++/Python, bytes/casos inválidos, ACK perdido,
orden START/PCM/COMMIT, caché/NACK/reintento único, cancelación/turnos viejos,
reutilización normal y timeouts; se conservan pruebas UART/UI/NVS/servicios.
Backend: regresión del cuerpo legacy y pruebas de sesiones v2 con WebSocket
simulado, recorte confirmado, pérdidas recuperadas y relleno limitado.

Falta medir en placa: latencia real de interrupción/limpieza DMA, heap mínimo,
stacks, carga UDP/Opus y recuperación bajo radio congestionada. Falta validar
con xAI real el recorte y las confirmaciones. La cola PCM de salida del backend
conserva el comportamiento sin límite del legacy; su control de presión es
pendiente. Se atiende un único endpoint: la factory de sesiones mantiene el
estado separado para un futuro registro multiusuario, que aún no está habilitado.
