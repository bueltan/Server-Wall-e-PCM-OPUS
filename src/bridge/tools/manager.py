"""MCP stdio connections owned by a single task (including context teardown)."""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def gmail_server_spec():
    # Do not pass Gemini/xAI credentials to the Gmail subprocess.
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'SSL_CERT_FILE', 'SSL_CERT_DIR')
           or key.startswith('PIUU_GMAIL_')}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    return ServerSpec('gmail', sys.executable,
                      ('-m', 'bridge.mcp_servers.gmail'), env)


class ToolManager:
    def __init__(self, servers=(), timeout=15.0):
        self.servers = tuple(servers)
        self.timeout = timeout
        self.declarations = []
        self.routes = {}
        self.queue = asyncio.Queue(8)
        self.worker = None
        self.ready = None

    async def start(self):
        if self.worker is not None:
            return
        self.ready = asyncio.get_running_loop().create_future()
        self.worker = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), self.timeout)
        except BaseException:
            await self.close()
            raise

    async def _run(self):
        try:
            # Lazy imports keep MCP optional when no server is enabled.
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            async with AsyncExitStack() as stack:
                for spec in self.servers:
                    streams = await stack.enter_async_context(stdio_client(
                        StdioServerParameters(command=spec.command, args=list(spec.args), env=spec.env)))
                    session = await stack.enter_async_context(ClientSession(*streams))
                    await session.initialize()
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        for tool in page.tools:
                            name = spec.name + '_' + tool.name
                            if name in self.routes:
                                raise ValueError('duplicate_mcp_tool')
                            self.routes[name] = (session, tool.name)
                            self.declarations.append({
                                'name': name, 'description': tool.description or name,
                                'parametersJsonSchema': tool.inputSchema,
                                'behavior': 'NON_BLOCKING'})
                        cursor = page.nextCursor
                        if not cursor:
                            break
                self.ready.set_result(None)
                while True:
                    name, args, future = await self.queue.get()
                    if future.cancelled():
                        continue
                    try:
                        session, remote_name = self.routes[name]
                        result = await asyncio.wait_for(session.call_tool(remote_name, args), self.timeout)
                        payload = {'ok': not result.isError}
                        if result.isError:
                            payload['error'] = 'mcp_tool_error'
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
                    except TimeoutError:
                        payload = {'ok': False, 'error': 'tool_timeout'}
                    except Exception:
                        payload = {'ok': False, 'error': 'tool_unavailable'}
                    if not future.done():
                        future.set_result(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.ready.done():
                self.ready.set_exception(RuntimeError('MCP startup failed; check dependencies and server configuration'))
        finally:
            self.routes.clear()
            self.declarations.clear()
            while not self.queue.empty():
                _, _, future = self.queue.get_nowait()
                if not future.done():
                    future.set_result({'ok': False, 'error': 'tool_unavailable'})

    async def call(self, name, args):
        if name not in self.routes or self.worker is None or self.worker.done():
            return {'ok': False, 'error': 'unsupported_tool'}
        if not isinstance(args, dict):
            return {'ok': False, 'error': 'invalid_tool_arguments'}
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((name, args, future))
        except asyncio.QueueFull:
            return {'ok': False, 'error': 'tool_busy'}
        try:
            return await asyncio.wait_for(future, self.timeout * 9)
        except TimeoutError:
            return {'ok': False, 'error': 'tool_timeout'}

    async def close(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
