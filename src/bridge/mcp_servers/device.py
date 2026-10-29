"""Device MCP: stdio tools translated to the bridge's private WebSocket API."""
import asyncio
import json
import os
import uuid
from typing import Annotated
from pydantic import Field, StrictBool
from websockets.asyncio.client import connect
from bridge.transport.device_control import failure, volume_body
from bridge.diagnostics.logging import configure_logging, event
from bridge.mcp_servers.diagnostics import exception_type


Percent = Annotated[int, Field(strict=True, ge=0, le=100)]
Delta = Annotated[int, Field(strict=True, ge=-100, le=100)]


class DeviceClient:
    def __init__(self, port, token):
        self.url = f'ws://127.0.0.1:{port}'
        self.token = token
        self.websocket = None
        self.calls = 0
        self.lock = asyncio.Lock()

    async def close(self):
        websocket, self.websocket = self.websocket, None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as exc:
                event('DEVICE_CLIENT_ERROR', phase='close', exception_type=exception_type(exc))

    async def call(self, operation, arguments):
        try:
            volume_body(operation, arguments)
        except (ValueError, TypeError):
            return failure('invalid_volume_arguments')
        async with self.lock:
            sent = False
            phase = 'connect'
            request_id = uuid.uuid4().hex
            try:
                # Rotate bounded per-connection request tracking before it fills.
                if self.calls >= 128:
                    await self.close()
                if self.websocket is None:
                    self.websocket = await connect(self.url,
                        additional_headers={'Authorization': 'Bearer ' + self.token},
                        open_timeout=2, close_timeout=1, max_size=4096, max_queue=4)
                    self.calls = 0
                    event('DEVICE_CLIENT_CONNECTED')
                # Treat a failed send as uncertain too: bytes may have escaped.
                sent = True
                async with asyncio.timeout(4):
                    phase = 'send'
                    await self.websocket.send(json.dumps({'request_id': request_id,
                        'operation': operation, 'arguments': arguments}))
                    self.calls += 1
                    phase = 'receive'
                    reply = json.loads(await self.websocket.recv())
                phase = 'decode'
                if (not isinstance(reply, dict) or reply.get('request_id') != request_id or
                        not isinstance(reply.get('result'), dict)):
                    raise ValueError('invalid_reply')
                return reply['result']
            except asyncio.CancelledError:
                event('DEVICE_CLIENT_CANCELLED', phase=phase, request_id=request_id)
                await self.close()
                raise
            except Exception as exc:
                event('DEVICE_CLIENT_ERROR', phase=phase, request_id=request_id,
                      exception_type=exception_type(exc), status='unknown' if sent else 'failed')
                await self.close()
                # Never reconnect and replay a possibly accepted adjustment.
                return failure('volume_not_confirmed' if sent else 'device_unavailable', sent)


def create_server(client=None):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    server = FastMCP('PIUU Device')
    client = client or DeviceClient(int(os.environ['PIUU_DEVICE_CONTROL_PORT']),
                                    os.environ['PIUU_DEVICE_CONTROL_TOKEN'])

    async def invoke(operation, arguments):
        try:
            return await client.call(operation, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event('DEVICE_CLIENT_ERROR', phase='tool', operation=operation,
                  exception_type=exception_type(exc))
            return failure('device_internal_error', True, 'exception')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                           idempotentHint=True, openWorldHint=False))
    async def get_volume() -> dict:
        """Read CORE's confirmed live playback volume and saved level."""
        return await invoke('get_volume', {})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                           idempotentHint=True, openWorldHint=False))
    async def set_volume(percent: Percent, persist: StrictBool = False) -> dict:
        """Set playback to 0..100. Persist only on an explicit request to save the default."""
        return await invoke('set_volume', {'percent': percent, 'persist': persist})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                           idempotentHint=False, openWorldHint=False))
    async def adjust_volume(delta: Delta) -> dict:
        """Add signed percentage points to CORE volume, clamped to 0..100. Default up/down is +20/-20."""
        return await invoke('adjust_volume', {'delta': delta})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                           idempotentHint=True, openWorldHint=False))
    async def set_mute(muted: StrictBool) -> dict:
        """Mute playback, or unmute by restoring CORE's last nonzero level."""
        return await invoke('set_mute', {'muted': muted})

    return server


if __name__ == '__main__':
    configure_logging()
    create_server().run(transport='stdio')
