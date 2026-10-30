# PIUU → Codex remoto

Integración opcional para Gemini, independiente del audio y del firmware. Python
3.11+, `asyncio`, SQLite, OpenSSH, bubblewrap y systemd de usuario y el SDK oficial MCP v1.
No requiere VS Code ni controlar una conversación abierta de la aplicación.

## Arquitectura y decisiones

```text
VPS: Gemini → ToolManager independiente → MCP stdio → cliente autenticado
                                      ↓ ws://127.0.0.1:4500
                                túnel SSH inverso
                                      ↓
ThinkPad: gateway autenticado → cola/SQLite local → Codex App Server privado stdio
                                              → proyecto autorizado
```

**El puerto 4500 expone operaciones de tareas, no el protocolo completo de Codex.**
Exponer directamente App Server permitiría al cliente remoto elegir rutas y
permisos, saltándose la política de proyectos. El servicio `piuu-codex.service`
supervisa el gateway; cada ejecución inicia y cierra su propio App Server privado.
La cola sigue viva aunque se cierre MCP, termine una conversación o caiga SSH.

Inspección: backend en `src/bridge`, MCP existente en `tools/manager.py`, registro
Gemini en `providers/gemini/live.py`, sesiones en `conversation/session.py`,
logging en `diagnostics/logging.py` y recordatorios en `notifications/calendar.py`.
Se reutilizan el cliente stdio, descubrimiento, validación JSON Schema y eventos.
El proveedor xAI no tiene integración Codex en esta versión; habilitarla con xAI
falla al iniciar en lugar de anunciar herramientas inexistentes. El servicio del
VPS es `walle.service`; no había servicios locales PIUU Codex.

Se verificó el ejecutable con `command -v codex`, `codex --version`,
`codex app-server --help` y los esquemas generados por el binario. Versión probada:
**0.162.0-alpha.2**, binario nativo de la aplicación; no requiere Node.js.
Para instalaciones npm/nvm, usar rutas absolutas y un `PATH` explícito de systemd,
porque no se carga el perfil interactivo del shell. App Server soporta stdio y
WebSocket autenticado; aquí stdio mantiene el protocolo privado. CLI `exec` admite
JSON y reanudación; el SDK oficial es una alternativa, pero no se añade Node.js
ni un segundo adaptador mientras App Server cubre sesiones, eventos y cancelación.

Referencias oficiales consultadas:
[App Server](https://developers.openai.com/codex/app-server),
[configuración](https://developers.openai.com/codex/config-reference),
[SDK Python MCP](https://github.com/modelcontextprotocol/python-sdk).
La documentación actual y los esquemas locales difieren en controles de lectura:
se usa un perfil nombrado `piuu`, verificado en `activePermissionProfile`, en lugar
de enviar campos `readOnlyAccess` ausentes en el protocolo instalado. La API del
perfil es experimental: verificar de nuevo después de actualizar Codex.

## Seguridad y alcance

- Gateway exclusivamente `127.0.0.1`, bearer de al menos 32 caracteres, fichero
  regular propiedad del usuario y modo 0600. Rechaza Origin de navegador,
  operaciones desconocidas y campos adicionales. No recibe comandos ni rutas.
- Registro JSON local privado con IDs de proyectos. Verifica rutas canónicas y
  enlaces simbólicos al listar y ejecutar. Cambiar el registro requiere reiniciar
  el servicio. Las rutas nunca son elegidas por Gemini.
- Perfil Codex: lectura mínima del sistema, escritura solo en el proyecto,
  lectura adicional solo en raíces configuradas, red de comandos deshabilitada.
  La metadata `.git` es de solo lectura. Se bloquean secretos habituales `.env*`, `.secrets`, claves PEM/KEY y
  archivos de credenciales. No es un detector universal de secretos: no registrar
  proyectos que contengan secretos con nombres distintos. Las raíces adicionales
  deben contener exclusivamente herramientas públicas, sin credenciales.
- Sin herencia de variables del backend; HOME del shell es el proyecto.
  App Server usa un CODEX_HOME dedicado, sin `config.toml`, `hooks.json` ni
  `AGENTS.md` personales. Codex puede generar sus cachés, skills del sistema y
  directorios de plugins; su mera existencia no significa habilitarlos.
- Se rechazan ampliaciones de permisos y aprobaciones de comandos fuera del
  sandbox; solicitudes de herramientas externas/interacción no se aprueban.
  No hay modo sin sandbox ni aprobación automática de escapes.
- El launcher usa bubblewrap para montar un
  `/dev` sintético sin dispositivos físicos. Se mantiene además el sandbox Codex.
  Bubblewrap activa `NoNewPrivs=1` antes de ejecutar Python. El gateway verifica
  ese indicador, rechaza UID 0 y exige únicamente los dispositivos virtuales
  esperados antes de leer configuración o abrir el puerto. La unidad no fija
  NoNewPrivileges antes del launcher: Ubuntu puede impedir con ello la transición
  AppArmor necesaria para crear el sandbox. No se permite una caída silenciosa
  a ejecución sin aislamiento. Red de comandos deshabilitada impide push/despliegues
  externos. El modelo sigue accediendo a OpenAI mediante el harness autenticado.
- La autonomía permite editar y ejecutar código dentro del proyecto. No es posible
  garantizar la conservación de todos sus archivos mientras se permite ejecutar
  código arbitrario allí. Para aislamiento de cambios, registrar un **worktree
  creado y revisado por el usuario**. No se crean ramas ni commits automáticamente;
  la metadata Git permanece fuera del acceso del agente. Un usuario Linux dedicado
  o contenedor aporta una frontera adicional recomendable para proyectos sensibles.
- Una tarea activa por proyecto o por raíces superpuestas (padre/hija), hasta cuatro proyectos independientes simultáneos configurables;
  cola máxima 100. Las tareas interrumpidas bloquean nuevas ejecuciones del mismo
  proyecto hasta reconciliar su resultado. No hay botón que olvide incertidumbre.

## Instalación local

Desde el checkout `PIUU_SERVER`:

```bash
.venv/bin/python -m pip install -r src/requirements-mcp.txt
command -v codex
codex --version
codex app-server generate-json-schema --experimental --out /tmp/piuu-codex-schema
mkdir -p ~/.config/piuu-codex ~/.local/share/piuu-codex/codex-home
chmod 700 ~/.config/piuu-codex ~/.local/share/piuu-codex
```

Crear un CODEX_HOME dedicado y autenticarlo con `codex login` según la instalación.
Si se comparte el login CLI existente, puede apuntarse `auth.json` del home dedicado
al archivo de autenticación ya existente mediante un enlace simbólico privado.
No copiar su contenido a `.env`, al VPS ni a Git. La renovación del login queda a
cargo de Codex; el bridge no gestiona credenciales OpenAI ni sesiones de la app.

Copiar y editar `.env.example` como `~/.config/piuu-codex/local.env`: usar rutas
absolutas reales, puerto disponible y `PIUU_CODEX_BINARY` descubierto. Las variables
se leen del entorno; no hay cargador dotenv implícito. `PIUU_CODEX_ENABLED` solo
controla las herramientas del backend, no es necesario en el gateway local.

Copiar `scripts/codex-projects.example.json` a un fichero privado, sustituir las
rutas y aplicar `chmod 600`. Preferir raíces específicas; no autorizar `/home` ni `/`. Una carpeta amplia
como Documents requiere autorización explícita y exclusiones de los controles del bridge. Las raíces de herramientas en `readable_roots` son
**solo lectura**; por ejemplo el entorno Python de PlatformIO y sus paquetes.

### Carpetas personales y exclusiones

También se admiten carpetas para tareas de archivos, aunque no contengan código ni
`platformio.ini`. En la ThinkPad se autorizaron explícitamente lectura y escritura
para los IDs `documents`, `downloads` y `music`, junto a CORE/EARS. Las rutas reales
son Documents, Downloads y Music, respectivamente. Cada tarea trabaja dentro de
la raíz elegida; registrar varias no concede automáticamente todas a cada tarea.

Para bloquear partes de una raíz amplia, agregar `excluded_paths` en el registro:

```json
{
  "id": "documents",
  "name": "Documentos",
  "path": "/absolute/path/Documents",
  "readable_roots": [],
  "excluded_paths": ["PlatformIO/Projects/PIUU_SERVER", "another-project/.venv"]
}
```

Son rutas **relativas** a esa raíz, sin `..`, comodines ni la raíz `.`. Se bloquea
lectura y escritura y no se recorre su contenido al validar enlaces. Un enlace
externo fuera de una exclusión sigue rechazando el workspace completo. No usar
exclusiones para ampliar acceso a destinos externos. Se omite también un
`platformio.ini` excluido, incluido un enlace cuyo destino esté excluido.

En la configuración local de Documents se excluyen el bridge completo (código,
estado y registro que controlan el servicio) y nueve entornos `.venv` con enlaces
que salen de Documents. La configuración personal permanece en `.state/projects.json`,
fuera de Git. Downloads y Music no tienen esas exclusiones de subcarpetas; se
mantienen los bloqueos habituales de secretos y el sandbox.

El Codex instalado no puede montar denegaciones hijas dentro de un directorio ya
denegado. Para raíces con exclusiones se expanden los patrones de secretos sobre
los archivos existentes al iniciar cada tarea y se eliminan denegaciones hijas
redundantes. Los archivos nuevos dentro de subcarpetas no se vuelven a escanear
por cada comando; no introducir secretos nuevos durante una tarea. Las raíces
sin exclusiones mantienen los patrones originales del perfil de Codex.
La validación usa `scandir`, evita seguir enlaces y poda las exclusiones, para no
hacer una llamada `stat` por cada archivo de carpetas grandes.

Después de editar el JSON, validar permisos 0600 y reiniciar solamente
`piuu-codex.service`; hacerlo con el agente inactivo porque interrumpe tareas en
curso. El túnel y el VPS no necesitan cambios por una modificación del registro.

Generar el bearer sin mostrarlo y fuera de Git:

```bash
(umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(48))' \
  > ~/.config/piuu-codex/gateway.token)
```

No ejecutar como root. Instalar las unidades después de revisar la configuración:

```bash
systemctl --user link "$PWD/systemd/piuu-codex.service"
systemctl --user link "$PWD/systemd/piuu-codex-tunnel.service"
systemctl --user daemon-reload
systemctl --user enable --now piuu-codex.service
```

Para iniciar antes del login y mantenerlo tras logout, habilitar linger para el
usuario con `loginctl enable-linger "$USER"` (puede requerir administrador).

### Ubuntu y AppArmor

Si `bwrap` falla con `setting up uid map: Permission denied`, revisar el bloqueo
`kernel.apparmor_restrict_unprivileged_userns`. No deshabilitarlo globalmente.
El servicio necesita un perfil que permita crear sus namespaces. En la ThinkPad
se autorizó explícitamente utilizar el perfil `chatgpt` ya instalado por la app,
mediante un drop-in de **esta unidad solamente**:

```ini
[Service]
AppArmorProfile=chatgpt
```

Ese perfil es `unconfined` y añade `userns`: no constituye la frontera de acceso
a archivos. La frontera de las herramientas es el perfil restringido de Codex y
el `/dev` privado del launcher. Revisar y autorizar este uso en cada instalación;
no copiar el override a servicios ajenos. Aplicar con `systemctl --user edit
piuu-codex.service`, daemon-reload y restart. No requiere editar el perfil de la app.

Como alternativa preparada (no instalada en la ThinkPad), un administrador puede
revisar `systemd/piuu-codex.apparmor` e instalar el wrapper root-owned
`scripts/piuu-codex-bwrap` en `/usr/local/libexec/piuu-codex-bwrap`, cargar el perfil
con `apparmor_parser` y configurar `PIUU_CODEX_BWRAP` con esa ruta. Requiere sudo,
ABI AppArmor compatible y una validación posterior; el bridge no aplica cambios
globales de kernel ni instala esa política automáticamente.

Los reinicios son de systemd. Cambiar pausas mediante `systemctl --user edit`:

```ini
[Service]
RestartSec=30
```

No se interpolan variables de entorno en `RestartSec`.

## Túnel SSH

Revisar `~/.config/piuu-codex/tunnel.env`, usando las variables de `.env.example`.
El script admite host/usuario/clave/puerto SSH y puertos local/remoto configurables.
No modifica `~/.ssh/config`, known_hosts, firewall ni configuración global SSH.
Verificar la huella del VPS por un canal confiable e incorporarla a known_hosts
antes de activar: `StrictHostKeyChecking=yes` y `BatchMode=yes` son obligatorios.
No iniciar una sesión tmux para el túnel; SSH permanece en primer plano.

```bash
systemctl --user enable --now piuu-codex-tunnel.service
```

En el VPS usar preferiblemente un usuario y una clave exclusivos. Ejemplo de
opciones de `authorized_keys` para esa clave (administrador debe revisar):

```text
restrict,port-forwarding,permitlisten="127.0.0.1:4500",command="/bin/false" ssh-ed25519 PUBLIC_KEY
```

`command=/bin/false` impide shells; `ssh -N` no solicita comando. Conservar
`AllowTcpForwarding remote`, `GatewayPorts no`, `PermitListen 127.0.0.1:4500`
y `PermitTTY no` en un bloque Match del usuario dedicado; validar con `sshd -t`
antes de recargar. No aplicar un bloque restrictivo al usuario administrativo
compartido sin separar antes sus otras claves y usos.

Comprobar en VPS con `ss -lnt` que 4500 esté ligado **solo a 127.0.0.1**. SSH usa
keepalive 15s/3, timeout 10s y `ExitOnForwardFailure=yes`. Los eventos CONNECTING,
CONNECTED y DISCONNECTED van a journald; `Restart=always`, pausa inicial 20s y
sin límite acumulado permiten recuperar Wi-Fi/IP sin bucles rápidos.

## Activación en el backend/VPS

Actualizar el checkout y usar el **intérprete real de ExecStart** de walle.service;
no asumir que es `.venv` del checkout. Instalar dependencias MCP allí si faltan.
Transferir únicamente el bearer del gateway, por SSH, a un archivo 0600 del mismo
usuario del backend. Nunca transferir credenciales OpenAI al VPS.

Agregar al entorno existente, sin reemplazar los secretos o valores anteriores:

```dotenv
PIUU_CODEX_ENABLED=1
PIUU_CODEX_HOST=127.0.0.1
PIUU_CODEX_PORT=4500
PIUU_CODEX_TOKEN_FILE=/absolute/private/path/gateway.token
PIUU_CODEX_EVENTS_STATE=/absolute/private/path/codex-events.cursor
```

Reiniciar `walle.service`. Cada conexión Gemini descubre ocho herramientas con
prefijo `codex_`, en un ToolManager independiente de Google y Device. Las llamadas
execute/continue devuelven inmediatamente un ID persistido; no esperan al modelo.
La cancelación de la espera de voz no cancela la tarea local. Se conservan
marcas de llamadas canceladas para no repetir mutaciones por el mismo call ID.

## Herramientas y ejemplos

En MCP stdio los nombres son `status`, `projects`, `execute`, `task_status`,
`task_result`, `cancel`, `task_list` y `continue`; PIUU agrega `codex_`.

```json
{"name":"codex_projects","arguments":{}}
```

```json
{"name":"codex_execute","arguments":{"project":"piuu_ears","instruction":"Revisar el VAD y ejecutar pruebas sin subir firmware","mode":"autonomous","idempotency_key":"voice_request_001"}}
```

```json
{"name":"codex_task_status","arguments":{"task_id":"task_RETURNED_ID"}}
```

```json
{"name":"codex_task_result","arguments":{"task_id":"task_RETURNED_ID","include_details":false}}
```

```json
{"name":"codex_continue","arguments":{"task_id":"task_RETURNED_ID","instruction":"Continuar con la revisión de los tests","idempotency_key":"voice_request_002"}}
```

```json
{"name":"codex_cancel","arguments":{"task_id":"task_RETURNED_ID"}}
```

Una nueva intención requiere una nueva clave; un reintento de la misma solicitud
requiere exactamente la misma clave y argumentos. Reutilizarla con otros datos
falla. Las claves son globales al gateway, no por conversación. Si se pierde una
respuesta, consultar task_list o reintentar explícitamente con la misma clave:
el cliente no repite mutaciones por su cuenta. Los nombres/rutas libres no forman
parte de la API.

Los estados son queued/running/completed/failed/cancelled/interrupted. Cancelar
solicita `turn/interrupt`; `cancelled` requiere confirmación. Una carrera puede
terminar en completed. Timeout, cierre de proceso o resultado incierto producen
interrupted, nunca un éxito inventado ni una repetición automática.
Al reiniciar se conservan queued y se reconcilian running usando thread/turn IDs
persistidos y `thread/read`. Un cierre antes de guardar el turn ID puede impedir
reconciliación automática; mantener interrupted y revisar manualmente. No repetir
la instrucción si pudo modificar archivos. La reanudación de contexto crea un
nuevo turno solo tras una solicitud continue y el fin confirmado del anterior.

El resumen es texto reportado por Codex, limitado a 2000 caracteres. El contador
de archivos proviene de eventos fileChange y puede ser null/incompleto si editó
mediante shell. `tests_passed`/`build_success` solo se derivan de códigos de salida
de comandos literales reconocidos; de lo contrario son null. No implican revisión
humana ni corrección funcional. `include_details=true` devuelve hasta 30 comandos,
códigos de salida, duración y cola de salida limitada a 2000 caracteres/comando.
Son datos privados del proyecto, no se escriben a journald. El resumen normal
omite los comandos y las salidas extensas.

## PlatformIO

projects incluye environments leídos de `platformio.ini`. Pedir `pio run` o
`pio run -e ENV` usando esos nombres. En esta fase no hay herramientas para
flashing. `pio test` puede subir firmware: para tests host, preferir el runner
local correspondiente o un entorno `native` expresamente preparado.

Las dependencias deben estar instaladas previamente: el sandbox no descarga
paquetes ni abre puertos. Revisar `PIUU_CODEX_DEV_PATH` y `readable_roots` para el
intérprete/toolchain. Si PlatformIO necesita escribir en su directorio global,
preparar un core/cache **dentro del proyecto autorizado** en lugar de conceder
escritura a todo el home. Un build no ejecutado se devuelve desconocido, no exitoso.

## Eventos, persistencia y diagnóstico

SQLite local guarda tareas e historial de eventos; no borrar mientras existan
tareas inciertas. Un lock de proceso impide dos gestores de la misma base.
La base, WAL y directorio de estado deben permanecer privados. No rotar el fichero
SQLite con un rename mientras el servicio esté activo. Para copias coherentes usar
la API backup de SQLite o detener el servicio.

El backend consume `events(after, limit)` del gateway con un cursor persistente.
`notifications/codex.py` entrega task_started/completed/failed/cancelled al callback
asíncrono `deliver`; por defecto solo registra `CODEX_NOTIFICATION`. **No genera
audio ni interrumpe conversaciones.** Una futura política de voz puede conectar ese
callback al mecanismo de notificaciones existente cuando la sesión esté ociosa.
Entrega al menos una vez; deduplicar por sequence. El cursor se actualiza solo tras
una entrega exitosa. Si el VPS estuvo apagado, recupera eventos del historial.

```bash
systemctl --user status piuu-codex.service piuu-codex-tunnel.service
journalctl --user -u piuu-codex.service -u piuu-codex-tunnel.service -n 60 --no-pager
systemctl show walle.service --property=ExecStart
journalctl -u walle.service -n 60 --no-pager
```

Los logs de PIUU contienen timestamp, task/project, estado y duración en los
cambios de estado; el progreso usa frases fijas. No registran instrucciones,
comandos, tokens ni excepciones remotas arbitrarias. App Server stderr no se
reenvía porque puede contener datos privados. `codex_rpc_rejected` requiere revisar
versión/esquema y configuración local; no intentar ampliar permisos a ciegas.

- `codex_transport_unavailable`: comprobar ambos servicios, loopback remoto y bearer.
  No asumir que la tarea dejó de ejecutarse.
- `private_file_required`: corregir propietario, modo 0600 y archivo regular.
- `dedicated_codex_home_not_clean`: separar configuración personal del servicio.
- `codex_permission_profile_mismatch`: incompatibilidad/configuración que no conserva
  el perfil. Se rechaza antes de iniciar el turno.
- `project_symlink_escape`/`project_path_changed`: revisar registro y árbol local.
- `session_not_ready`: esperar estado terminal confirmado antes de continuar.
- Fallo de forward: puerto ocupado o restricciones SSH. No cambiar GatewayPorts a yes.
- Error bwrap: revisar soporte de sandbox del host y versión; nunca deshabilitarlo.

Para desactivar, quitar PIUU_CODEX_ENABLED y reiniciar backend; detener las unidades
locales interrumpe tareas en curso y requiere reconciliar. No elimina cambios ni
historial.

## Validación reproducible

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pip check
bash -n scripts/piuu-codex-start.sh scripts/piuu-codex-tunnel.sh
systemd-analyze --user verify systemd/piuu-codex.service systemd/piuu-codex-tunnel.service
```

Tests automáticos usan agentes simulados y un circuito MCP stdio → WS loopback real;
no modifican firmware ni usan OpenAI. Cubren autenticación/Origin, proyectos,
traversal/symlinks, idempotencia, cola, cancelación, continuación, timeout, historial,
reconciliación, caída/reconexión del transporte y cursores de notificaciones.

Prueba explícita con modelo real (consume la cuenta configurada), solo en directorio
nuevo desechable, con PIUU_CODEX_HOME y PIUU_CODEX_BINARY configurados:

```bash
PYTHONPATH=src .venv/bin/python scripts/piuu-codex-smoke.py
```

La aceptación física de reinicio de Linux y pérdida real de Wi-Fi requiere pruebas
operativas adicionales; las pruebas de reconexión simulada no equivalen a apagar
el equipo. La voz y la compilación de cada firmware deben comprobarse por separado.

### Resultados de la validación de esta versión

- Python 3.12 local: **290 tests** completos aprobados; compilación de sintaxis
  Python/Bash, pip check, systemd-analyze y revisión de diff aprobadas.
- Python 3.11 del VPS: **31 tests Codex** aprobados. La suite completa remota
  no pudo terminar: se bloquea en
  `test_ingress.InboundTests.test_coordination_pause_does_not_consume_send_timeout`.
  El mismo bloqueo se reprodujo en una copia del commit anterior `7256ef2` con
  timeout de 20 segundos; el intento completo actualizado se detuvo a los 120
  segundos. No se cambió audio para resolver un problema ajeno a esta integración.
- App Server real: edición y unittest en proyectos temporales, aislamiento con
  archivos señuelo y metadata Git de solo lectura.
- Servicio real local: gateway y túnel activos/habilitados, linger activado, historial
  SQLite preservado al reiniciar.
- MCP real desde VPS: ocho herramientas descubiertas, tarea creada/resultados
  recuperados y reintentos deduplicados. Una continuación conservó el thread y
  siguió ejecutándose durante un corte intencional del túnel; completó la espera
  de 40 segundos y sus pruebas después de la reconexión automática.
- Cancelación real confirmada de un comando de espera, sin esperar los 120 segundos
  solicitados por la prueba. Ninguna prueba modificó los proyectos de firmware.

Siguen pendientes las pruebas de voz con PIUU, compilaciones concretas de los
firmwares autorizados, reinicio físico de Linux/VPS y pérdida física de Wi-Fi.
No equivalen a los tests de transporte y reinicio de servicio ejecutados. La
restricción de la clave SSH dedicada descrita arriba requiere un usuario/clave
separados; no se modificó la configuración SSH global ni la clave administrativa
existente durante estas pruebas.
