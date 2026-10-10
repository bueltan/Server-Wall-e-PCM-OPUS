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
La sesión xAI configura detección manual de turno; CORE decide cuándo cerrar.

## Transporte actual

**Actualizar CORE y todo el árbol `src/` del backend juntos.** Se exige sobre
UDP v2 `PIU2`; los datagramas v1 sin sobre se rechazan. El cuerpo PCM/Opus conserva
su serialización anterior de 18 bytes, dentro de ese sobre.

- Identidad de conexión/turno/respuesta en audio y controles.
- START, COMMIT y CANCEL confirmados e idempotentes.
- Touch/wake durante respuesta cancela y abre un turno nuevo; el backend recorta
  el historial remoto usando la reproducción conservadora estimada por CORE.
- Recuperación PCM por NACK y un reintento por frame. Si falla, silencio de
  20 ms por frame faltante, hasta 200 ms por hueco. Huecos mayores se registran.
- Cola de entrada acotada: 16 frames y espacio reservado para END/COMMIT.
  Saturación descarta frames nuevos conservando el turno.
- HELLO/PONG informa si xAI ya confirmó la configuración de la sesión.

Un único endpoint por ahora. Cada sesión concentra su xAI/codec/estado; un
registro multiusuario puede añadirse detrás de la factory UDP. Cambiar IP/puerto
de CORE requiere reiniciar el bridge. Reiniciar el transporte CORE desde el
mismo endpoint limpia la sesión anterior antes de adoptar su nuevo nonce.

Ver [contrato UDP v2](PROTOCOLO_UDP_V2.md) y [responsabilidades](REFACTOR.md).

## Logs para pruebas integradas

El entrypoint habilita logs INFO con fecha UTC, IDs CORE y response_id de xAI.
Incluyen ACK/reintentos observados, cancelación/recorte confirmado, colas y pérdidas.
No imprimen frames de audio ni transcripciones. Resúmenes cada cinco segundos.
Ver [captura e interpretación](DIAGNOSTICOS_TEST_INTEGRADO.md).

## Verificación sin hardware

```sh
.venv/bin/python -m unittest discover -s tests -v
```

37 pruebas locales: serialización, codecs/grabaciones, colas/reordenamiento,
controles duplicados, aislamiento de IDs, recuperación, silencio limitado,
cancelación y confirmaciones de recorte con un WebSocket simulado. No usan
credenciales, sockets UDP ni sesiones xAI reales.

No se cargó firmware ni se conectó a xAI. Faltan pruebas en placa de radio,
latencia, DMA y recursos, y validación del recorte con el proveedor. La cola PCM
de salida del backend conserva el comportamiento sin límite del legacy; falta
incorporar control de presión. No hay retransmisión de Opus.

Las grabaciones WAV opcionales representan audio aceptado/procesado por el
bridge; incluyen silencio insertado en uplink. No miden lo escuchado físicamente.
