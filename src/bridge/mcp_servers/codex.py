"""Official MCP SDK stdio facade exposing only authenticated task operations."""
from typing import Annotated, Literal
from pydantic import Field
from bridge.codex.client import TaskClient
from bridge.diagnostics.logging import configure_logging

Key = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')]
Instruction = Annotated[str, Field(min_length=1, max_length=12000)]


def create_server(client=None):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    server = FastMCP('PIUU Codex')
    client = client or TaskClient.environment()
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)

    @server.tool(annotations=read)
    async def status() -> dict:
        """Check local Codex availability. A tunnel failure does not cancel tasks."""
        return await client.call('status', {})

    @server.tool(annotations=read)
    async def projects() -> dict:
        """List authorized project IDs and their PlatformIO environments."""
        return await client.call('projects', {})

    @server.tool(annotations=write)
    async def execute(project: Key, instruction: Instruction, idempotency_key: Key,
                      mode: Literal['autonomous'] = 'autonomous') -> dict:
        """Queue a user's development request. Use the SAME unique key on retries.
        Only authorized projects. Never flash, deploy, commit or push.
        """
        return await client.call('execute', dict(project=project, instruction=instruction,
                                                idempotency_key=idempotency_key, mode=mode))

    @server.tool(annotations=read)
    async def task_status(task_id: Key) -> dict:
        """Get short progress; interrupted means an uncertain outcome."""
        return await client.call('task_status', {'task_id': task_id})

    @server.tool(annotations=read)
    async def task_result(task_id: Key, include_details: bool = False) -> dict:
        """Retrieve reported summary and command evidence. Null check results mean unknown."""
        return await client.call('task_result', {'task_id': task_id, 'include_details': include_details})

    @server.tool(annotations=write)
    async def cancel(task_id: Key) -> dict:
        """Request cancellation. Confirm only after cancelled; edits are not undone."""
        return await client.call('cancel', {'task_id': task_id})

    @server.tool(annotations=read)
    async def task_list(limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict:
        """List recent tasks, including those running after a reconnection."""
        return await client.call('task_list', {'limit': limit})

    @server.tool(name='continue', annotations=write)
    async def continue_task(task_id: Key, instruction: Instruction, idempotency_key: Key) -> dict:
        """Continue a terminal task's saved context with a NEW idempotency key."""
        return await client.call('continue', dict(task_id=task_id, instruction=instruction,
                                                 idempotency_key=idempotency_key))

    return server


if __name__ == '__main__':
    configure_logging()
    create_server().run(transport='stdio')
