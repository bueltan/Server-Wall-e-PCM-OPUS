"""UDP endpoint routing; client state is isolated in ClientSession.

Single-session admission for now. A future registry can create one ClientSession
per peer, each with its own xAI connection, queues, encoder and recordings.
"""
import asyncio
from .config import BridgeConfig
from .managed_session import ManagedSession

class UdpAudioBridgeProtocol(asyncio.DatagramProtocol):
    def __init__(self, config: BridgeConfig, session_factory=ManagedSession):
        self.session = session_factory(config)

    def connection_made(self, transport):
        self.session.connection_made(transport)

    def datagram_received(self, data, addr):
        self.session.datagram_received(data, addr)

    def connection_lost(self, exc):
        self.session.connection_lost(exc)
