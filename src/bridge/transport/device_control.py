"""Session-owned volume transactions. UDP receipt alone never confirms success."""
import asyncio
import struct
import time
from bridge.transport.protocols.conversation import Identity, Kind
from bridge.diagnostics.logging import event
from bridge.mcp_servers.diagnostics import exception_type


OPERATIONS = {'get_volume': 0, 'set_volume': 1, 'adjust_volume': 2, 'set_mute': 3}


def request_tag(value):
    return value if isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value) else None


def failure(error, unknown=False, kind=None):
    return {'status': 'unknown' if unknown else 'failed', 'error': error,
            'retry_safe': False, 'error_kind': kind or (
                'validation' if error == 'invalid_volume_arguments' else
                'storage' if error == 'volume_save_failed' else
                'timeout' if error == 'volume_not_confirmed' else 'transport')}


def volume_body(operation, arguments):
    if not isinstance(arguments, dict) or not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError('invalid_volume_arguments')
    fields = {'get_volume': (), 'set_volume': ('percent', 'persist'),
              'adjust_volume': ('delta',), 'set_mute': ('muted',)}[operation]
    if set(arguments) - set(fields):
        raise ValueError('invalid_volume_arguments')
    persist = arguments.get('persist', False)
    if type(persist) is not bool:
        raise ValueError('invalid_volume_arguments')
    if operation == 'set_mute':
        value = arguments.get('muted')
        if type(value) is not bool:
            raise ValueError('invalid_volume_arguments')
        value = int(value)
    elif operation == 'get_volume':
        value = 0
    else:
        value = arguments.get('percent' if operation == 'set_volume' else 'delta')
        low = 0 if operation == 'set_volume' else -100
        if type(value) is not int or not low <= value <= 100:
            raise ValueError('invalid_volume_arguments')
    return struct.pack('!BbB', OPERATIONS[operation], value, int(persist))


class VolumeTransactions:
    def __init__(self, session, timeout=2.0, retry=.15):
        self.session, self.timeout, self.retry = session, timeout, retry
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.pending = None

    def reset(self):
        # Keep sequence increasing even across peer resets; no stale reuse.
        if self.pending is not None and not self.pending[2].done():
            self.pending[2].set_result(failure('device_session_changed', True))

    def receive(self, packet):
        pending = self.pending
        if (pending is None or packet.flags or len(packet.body) != 4 or
                packet.identity != pending[0] or packet.sequence != pending[1]):
            return False
        status, level, restore, saved = packet.body
        if status > 2 or level > 100 or not 1 <= restore <= 100 or saved > 100:
            return False
        result = {'status': 'confirmed', 'volume_percent': level, 'muted': level == 0,
                  'restore_percent': restore, 'saved_percent': saved}
        if status:
            result.update(failure('volume_save_failed' if status == 2 else 'invalid_volume_arguments'))
            result['applied'] = status == 2
        if not pending[2].done():
            pending[2].set_result(result)
        return True

    async def call(self, operation, arguments, binding, request_id=None):
        started = time.monotonic()
        operation_name = operation if isinstance(operation, str) and operation in OPERATIONS else None
        tag = request_tag(request_id)
        event('DEVICE_VOLUME_STARTED', binding, operation=operation_name, request_id=tag)
        try:
            result = await self._call(operation, arguments, binding, tag)
        except asyncio.CancelledError:
            event('DEVICE_VOLUME_CANCELLED', binding, operation=operation_name, request_id=tag)
            raise
        except Exception as exc:
            # A transport send may have reached CORE before raising.
            code = 'device_transport_error' if isinstance(exc, (OSError, ConnectionError)) else 'device_internal_error'
            result = failure(code, True, 'transport' if code == 'device_transport_error' else 'exception')
            result['exception_type'] = exception_type(exc)
        event('DEVICE_VOLUME_FINISHED', binding, operation=operation_name, request_id=tag,
              status=result['status'], error=result.get('error'), error_kind=result.get('error_kind'),
              exception_type=result.get('exception_type'), volume=result.get('volume_percent'),
              duration_ms=round((time.monotonic() - started) * 1000))
        return result

    async def _call(self, operation, arguments, binding, request_id):
        try:
            body = volume_body(operation, arguments)
        except ValueError:
            return failure('invalid_volume_arguments')
        # Bound work across all local clients; never accumulate delayed commands.
        if self.lock.locked():
            return failure('device_busy')
        async with self.lock:
            session = self.session
            if binding is None or binding != session.conversation.identity:
                return failure('device_session_changed')
            if not session.volume_capable:
                return failure('volume_unsupported')
            if (session.closing or session.transport is None or
                    time.monotonic() - session.last_peer_seen >= session.config.core_disconnect_timeout_sec):
                return failure('device_unavailable')
            if self.sequence == 0xffffffff:
                return failure('device_unavailable')
            self.sequence += 1
            identity = Identity(binding.client, binding.server, binding.turn, 0)
            future = asyncio.get_running_loop().create_future()
            self.pending = (identity, self.sequence, future)
            deadline = asyncio.get_running_loop().time() + self.timeout
            attempts = 0
            try:
                while True:
                    if session.conversation.identity != binding or session.closing:
                        return failure('device_session_changed', True)
                    attempts += 1
                    event('DEVICE_VOLUME_SEND' if attempts == 1 else 'DEVICE_VOLUME_RETRY', identity,
                          sequence=self.sequence, attempt=attempts, request_id=request_id)
                    session._send(Kind.VOLUME, identity, self.sequence, body)
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        return failure('volume_not_confirmed', True)
                    try:
                        return await asyncio.wait_for(asyncio.shield(future), min(self.retry, remaining))
                    except TimeoutError:
                        if asyncio.get_running_loop().time() >= deadline:
                            return failure('volume_not_confirmed', True)
            finally:
                self.pending = None
                if not future.done():
                    future.cancel()
