# Backend: responsabilidades y estado — 2026-10-10

El entrypoint es `python src/main.py`. Desplegar el árbol `src/` completo junto
con firmware CORE UDP v2; no basta con copiar `main.py`.

| Módulo | Responsabilidad |
| --- | --- |
| `config` | Audio, proveedor y límites configurables |
| `session_protocol` | Sobre UDP v2 e identidades, sin dependencias de red |
| `protocol` | Cuerpo PCM/Opus de 18 bytes, independiente del sobre |
| `codec` / `recording` | libopus y WAV |
| `ingress` | Cola de entrada, reorder, NACK, relleno limitado y cierre |
| `managed_session` | Sesión v2, controles fiables, identidad y salida etiquetada |
| `tagged_xai` | Asociación de respuestas, cancelación y recorte confirmado |
| `udp` | Endpoint y factory de sesiones; producción usa ManagedSession |
| `app` | Arranque/cierre del transporte |
| `session` / `xai` | Bases extraídas del legacy; helpers y regresión de referencia |

El runtime UDP no acepta controles/audio desnudos v1. Las clases legacy que
`main.py` reexporta se conservan para referencia/compatibilidad de imports; no
son el endpoint activo. ManagedSession reutiliza helpers de codec/recording,
pero sustituye admisión, control, eventos xAI y envío de respuestas.

## Decisiones aplicadas

- Se conserva 16 kHz mono PCM16, Opus de 20 ms, voz/prompt y pacing existentes.
- IDs completos aíslan conexiones, turnos y respuestas. START explícito abre
  entrada; PCM tardío nunca abre automáticamente otra entrada de producción.
- START/COMMIT/CANCEL llevan IDs de operación y ACK cacheados. COMMIT repetido
  no genera otra respuesta. La secuencia final del COMMIT cubre END perdido.
- Se invalida el turno inmediatamente al admitir CANCEL. Deltas de xAI y cada
  frame UDP se comprueban contra la identidad antes de transmitirlos.
- CANCEL detiene generación, recorta/elimina contenido no escuchado y limpia
  entrada remota, esperando confirmaciones. Si queda ambiguo, se cierra xAI.
- Reorder base 40 ms; recuperación/cierre hasta 120 ms. Caché CORE de 12 frames,
  edad 240 ms, un reintento. No ARQ de salida Opus.
- Política revisada por el usuario: los huecos no recuperados insertan 20 ms
  de silencio por frame, máximo 200 ms por hueco, también en grabación WAV.
- Saturación conserva el turno y descarta frames nuevos. Append WS fallido no
  se reintenta ni se rellena: puede haberse enviado antes del fallo.
- Un endpoint ahora; estado por sesión prepara un registro multiusuario futuro.

Detalles del formato, límites de reproducción, reinicio y recuperación en
[PROTOCOLO_UDP_V2.md](PROTOCOLO_UDP_V2.md).

## Verificación y pendientes

37 pruebas offline con asyncio y WebSocket simulado, sin red/credenciales.
La suite host CORE valida bytes C++/Python en ambos sentidos y el cliente fiable.
Compilan `cyd` y `cyd2usb`. No se cargó firmware ni se abrió sesión xAI real.

Pendientes: validación real de truncate/delete, afinado de ventanas y caché con
radio congestionada, latencia de interrupción/DMA, recursos ESP32, control de
presión de salida del backend, registro/admisión multiusuario y estimación de
reproducción específica cuando se implemente Bluetooth. Los nonces aíslan
sesiones pero no autentican equipos. Un cambio de IP/puerto requiere reinicio.
