"""MCP stdio connections owned by a single task (including context teardown)."""
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from bridge.diagnostics.logging import event
from bridge.mcp_servers.diagnostics import SAFE_ERROR_CODES, exception_type


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def google_server_spec(name, extra_env=None):
    # Each subprocess gets only its own Google integration settings, never API keys.
    prefix = 'PIUU_' + name.upper() + '_'
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'SSL_CERT_FILE', 'SSL_CERT_DIR')
           or key.startswith(prefix)}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    env.update(extra_env or {})
    return ServerSpec(name, sys.executable, ('-m', 'bridge.mcp_servers.' + name), env)


def gmail_server_spec():
    return google_server_spec('gmail')


def calendar_server_spec(time_zone=None):
    return google_server_spec('calendar', {'PIUU_CALENDAR_TIME_ZONE': time_zone} if time_zone else None)


def device_server_spec(config):
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'PIUU_LOG_LEVEL')}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               PIUU_DEVICE_CONTROL_PORT=str(config.device_control_port),
               PIUU_DEVICE_CONTROL_TOKEN=config.device_control_token)
    return ServerSpec('device', sys.executable, ('-m', 'bridge.mcp_servers.device'), env)


class ToolManager:
    def __init__(self, servers=(), timeout=15.0):
        self.servers = tuple(servers)
        self.timeout = timeout
        self.declarations = []
        self.routes = {}
        self.queue = asyncio.Queue(8)
        self.worker = None
        self.ready = None
        self.validators = {}
        self.call_sequence = 0

    def _finished(self, name, call_id, started, payload, identity=None):
        data = payload.get('data')
        data = data if isinstance(data, dict) else {}
        code = payload.get('error') or data.get('error')
        code = code if isinstance(code, str) and code in SAFE_ERROR_CODES else ('mcp_tool_error' if code else None)
        kind = payload.get('error_kind') or data.get('error_kind')
        kind = kind if kind in ('api', 'auth', 'validation', 'exception', 'timeout', 'transport',
                                'remote', 'shutdown', 'cancelled', 'storage') else None
        status = data.get('status')
        status = status if isinstance(status, str) and status in ('failed', 'unknown', 'created', 'sent', 'confirmed') else None
        http = data.get('http_status')
        exception = payload.get('exception_type') or data.get('exception_type')
        event('MCP_TOOL_FINISHED', identity, tool=name, call_id=call_id,
              duration_ms=round((time.monotonic() - started) * 1000),
              ok=payload['ok'], error=code, status=status, error_kind=kind,
              http_status=http if type(http) is int and 100 <= http <= 599 else None,
              retry_safe=data.get('retry_safe') if type(data.get('retry_safe')) is bool else None,
              exception_type=exception if exception in ('TimeoutError', 'ConnectionError', 'OSError',
                  'ValueError', 'TypeError', 'KeyError', 'RuntimeError', 'Exception') else None,
              validation=payload.get('validation'))

    def _schema_failure(self, name, args):
        validator = self.validators.get(name)
        if validator is None:
            return None
        errors = []
        for error in validator.iter_errors(args):
            # Only schema-defined top-level fields and fixed validator names;
            # ValidationError.message/instance can contain private argument data.
            root = next(iter(error.absolute_path), None)
            field = root if root in validator.schema.get('properties', {}) else None
            rule = error.validator if error.validator in (
                'required', 'type', 'enum', 'format', 'additionalProperties',
                'minimum', 'maximum', 'minLength', 'maxLength', 'pattern') else 'schema'
            if rule == 'required' and not error.absolute_path:
                missing = [key for key in error.validator_value
                           if key in validator.schema.get('properties', {}) and key not in args]
                errors.extend({'field': key, 'rule': rule} for key in missing)
            else:
                errors.append({'field': field, 'rule': rule})
            if len(errors) >= 5:
                errors = errors[:5]
                break
        if errors:
            return {'ok': False, 'error': 'invalid_tool_arguments',
                    'error_kind': 'validation', 'validation': errors}
        return None

    async def start(self):
        if self.worker is not None:
            return
        self.ready = asyncio.get_running_loop().create_future()
        self.worker = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), self.timeout)
        except TimeoutError:
            event('MCP_SERVER_ERROR', server=None, phase='startup', error='mcp_startup_timeout')
            await self.close()
            raise
        except BaseException:
            await self.close()
            raise

    async def _run(self):
        active_server = None
        phase = 'dependencies'
        try:
            # Lazy imports keep MCP optional when no server is enabled.
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from jsonschema.validators import validator_for
            async with AsyncExitStack() as stack:
                for spec in self.servers:
                    active_server = spec.name
                    phase = 'connect'
                    streams = await stack.enter_async_context(stdio_client(
                        StdioServerParameters(command=spec.command, args=list(spec.args), env=spec.env)))
                    session = await stack.enter_async_context(ClientSession(*streams))
                    phase = 'initialize'
                    await session.initialize()
                    phase = 'discovery'
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        for tool in page.tools:
                            name = spec.name + '_' + tool.name
                            if name in self.routes:
                                raise ValueError('duplicate_mcp_tool')
                            self.routes[name] = (session, tool.name)
                            validator_type = validator_for(tool.inputSchema)
                            validator_type.check_schema(tool.inputSchema)
                            self.validators[name] = validator_type(tool.inputSchema)
                            self.declarations.append({
                                'name': name, 'description': tool.description or name,
                                'parametersJsonSchema': tool.inputSchema,
                                'behavior': 'NON_BLOCKING'})
                        cursor = page.nextCursor
                        if not cursor:
                            break
                    event('MCP_SERVER_READY', server=spec.name,
                          tools=sum(name.startswith(spec.name + '_') for name in self.routes))
                self.ready.set_result(None)
                phase = 'worker'
                while True:
                    name, args, future, call_id, queued, identity = await self.queue.get()
                    if future.cancelled():
                        self._finished(name, call_id, queued, {
                            'ok': False, 'error': 'tool_cancelled', 'error_kind': 'cancelled'}, identity)
                        continue
                    started = time.monotonic()
                    event('MCP_TOOL_STARTED', identity, tool=name, call_id=call_id,
                          queue_ms=round((started - queued) * 1000))
                    try:
                        session, remote_name = self.routes[name]
                        payload = self._schema_failure(name, args)
                        if payload is not None:
                            self._finished(name, call_id, started, payload, identity)
                            if not future.done():
                                future.set_result(payload)
                            continue
                        result = await asyncio.wait_for(session.call_tool(remote_name, args), self.timeout)
                        payload = {'ok': not result.isError}
                        if result.isError:
                            # Unknown/external MCP error text is untrusted and may
                            # include credentials. Owned servers return safe codes.
                            payload.update(error='mcp_tool_error', error_kind='remote')
                        elif result.structuredContent is not None:
                            payload['data'] = result.structuredContent
                        else:
                            texts = [part.text for part in result.content if part.type == 'text']
                            payload['data'] = texts
                            if len(texts) == 1:
                                try:
                                    payload['data'] = json.loads(texts[0])
                                except ValueError:
                                    pass
                        data = payload.get('data')
                        if isinstance(data, dict) and (data.get('error') or data.get('status') in ('failed', 'unknown')):
                            payload['ok'] = False
                            payload['error'] = data.get('error') or 'mcp_tool_error'
                    except asyncio.CancelledError:
                        self._finished(name, call_id, started, {
                            'ok': False, 'error': 'tool_cancelled', 'error_kind': 'cancelled'}, identity)
                        raise
                    except TimeoutError:
                        payload = {'ok': False, 'error': 'tool_timeout', 'error_kind': 'timeout',
                                   'exception_type': 'TimeoutError'}
                    except Exception as exc:
                        payload = {'ok': False, 'error': 'tool_unavailable', 'error_kind': 'transport',
                                   'exception_type': exception_type(exc)}
                    self._finished(name, call_id, started, payload, identity)
                    if not future.done():
                        future.set_result(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event('MCP_SERVER_ERROR', server=active_server,
                  phase=phase, error='mcp_server_unavailable', exception_type=exception_type(exc))
            if not self.ready.done():
                self.ready.set_exception(RuntimeError('MCP startup failed; check dependencies and server configuration'))
        finally:
            self.routes.clear()
            self.declarations.clear()
            self.validators.clear()
            while not self.queue.empty():
                name, _, future, call_id, queued, identity = self.queue.get_nowait()
                if not future.done():
                    payload = {'ok': False, 'error': 'tool_unavailable', 'error_kind': 'shutdown'}
                    self._finished(name, call_id, queued, payload, identity)
                    future.set_result(payload)

    async def call(self, name, args, identity=None):
        self.call_sequence += 1
        call_id, queued = self.call_sequence, time.monotonic()
        def rejected(code):
            payload = {'ok': False, 'error': code}
            # Do not log unknown tool names supplied by a provider.
            self._finished(name if name in self.routes else None, call_id, queued, payload, identity)
            return payload
        if name not in self.routes or self.worker is None or self.worker.done():
            return rejected('unsupported_tool')
        if not isinstance(args, dict):
            return rejected('invalid_tool_arguments')
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((name, args, future, call_id, queued, identity))
        except asyncio.QueueFull:
            return rejected('tool_busy')
        try:
            return await asyncio.wait_for(future, self.timeout * 9)
        except TimeoutError:
            return rejected('tool_timeout')

    async def close(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
