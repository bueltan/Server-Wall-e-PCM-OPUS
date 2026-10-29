"""Private local WebSocket API over the session-owned device transport."""
import hmac
import json
from http import HTTPStatus
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from bridge.transport.device_control import failure, request_tag
from bridge.diagnostics.logging import event
from bridge.mcp_servers.diagnostics import exception_type


class DeviceGateway:
    def __init__(self, session, token):
        if not token:
            raise ValueError('Device control requires an ephemeral token')
        self.session, self.token = session, token
        self.connections = 0

    def authenticate(self, connection, request):
        values = request.headers.get_all('Authorization')
        if len(values) != 1 or not hmac.compare_digest(values[0].encode(), ('Bearer ' + self.token).encode()):
            event('DEVICE_WS_REJECTED', reason='authentication')
            return connection.respond(HTTPStatus.UNAUTHORIZED, 'Unauthorized\n')
        if request.headers.get_all('Origin'):
            event('DEVICE_WS_REJECTED', reason='origin')
            return connection.respond(HTTPStatus.FORBIDDEN, 'Forbidden\n')

    async def handle(self, websocket):
        self.connections += 1
        connection = self.connections
        binding = self.session.conversation.identity
        event('DEVICE_WS_CONNECTED', binding, connection=connection)
        try:
            await self._handle(websocket, binding, connection)
        except ConnectionClosed:
            # MCP child teardown can close the socket without a close frame.
            pass
        except Exception as exc:
            event('DEVICE_WS_ERROR', binding, connection=connection, phase='connection',
                  exception_type=exception_type(exc))
            await websocket.close(code=1011, reason='device_unavailable')
        finally:
            event('DEVICE_WS_CLOSED', binding, connection=connection)

    async def _handle(self, websocket, binding, connection):
        # A connection cannot silently acquire a subsequent conversation/device.
        seen = set()
        async for raw in websocket:
            request_id = None
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError()
                request_id = request.get('request_id')
                if (not isinstance(request_id, str) or not 1 <= len(request_id) <= 64 or
                        set(request) != {'request_id', 'operation', 'arguments'}):
                    raise ValueError()
                if request_id in seen:
                    result = failure('duplicate_device_request')
                elif len(seen) >= 256:
                    result = failure('device_connection_expired')
                else:
                    seen.add(request_id)
                    result = await self.session.volume.call(request['operation'], request['arguments'], binding,
                                                           request_tag(request_id))
            except (ValueError, TypeError):
                result = failure('invalid_volume_arguments')
            except Exception as exc:
                result = failure('device_internal_error', True, 'exception')
                event('DEVICE_WS_ERROR', binding, connection=connection, phase='request',
                      request_id=request_tag(request_id), exception_type=exception_type(exc))
            event('DEVICE_WS_RESULT', binding, connection=connection, request_id=request_tag(request_id),
                  status=result['status'], error=result.get('error'))
            await websocket.send(json.dumps({'request_id': request_id, 'result': result}))

    async def start(self, port):
        try:
            server = await serve(self.handle, '127.0.0.1', port,
                                 process_request=self.authenticate, origins=[None],
                                 max_size=4096, max_queue=4, close_timeout=1,
                                 ping_interval=15, ping_timeout=15)
        except Exception as exc:
            event('DEVICE_WS_ERROR', phase='startup', port=port, exception_type=exception_type(exc))
            raise RuntimeError('Device gateway startup failed') from None
        event('DEVICE_WS_READY', port=server.sockets[0].getsockname()[1])
        return server
