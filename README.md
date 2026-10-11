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

## Logs para pruebas integradas

El entrypoint habilita logs INFO con fecha UTC, IDs CORE y response_id de xAI.
Incluyen ACK/reintentos observados, cancelación/recorte confirmado, colas y pérdidas.
No imprimen frames de audio ni transcripciones. Resúmenes cada cinco segundos.
Ver [captura e interpretación](DIAGNOSTICOS_TEST_INTEGRADO.md).

## Verificación sin hardware

```sh
.venv/bin/python -m unittest discover -s tests -v
```

54 pruebas locales (incluyen v3 y regresiones legacy): serialización, codecs/grabaciones, colas/reordenamiento,
controles duplicados, aislamiento de IDs, recuperación, silencio limitado,
cancelación y confirmaciones de recorte con un WebSocket simulado. No usan
credenciales, sockets UDP ni sesiones xAI reales.

No se cargó firmware ni se conectó a xAI. Faltan pruebas en placa de radio,
latencia, DMA y recursos, y validación del recorte con el proveedor. La cola PCM
de salida del backend conserva el comportamiento sin límite del legacy; falta
incorporar control de presión. No hay retransmisión de Opus.

Las grabaciones WAV opcionales representan audio aceptado/procesado por el
bridge; incluyen silencio insertado en uplink. No miden lo escuchado físicamente.
