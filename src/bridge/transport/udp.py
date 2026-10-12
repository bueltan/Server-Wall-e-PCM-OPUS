"""UDP endpoint routing; client state is isolated in ContinuousSession.

Single-session admission for now. A future registry can create one ContinuousSession
per peer, each with its own xAI connection, queues, encoder and recordings.
"""
import asyncio
from bridge.config import BridgeConfig
from bridge.conversation.session import ContinuousSession

class UdpAudioBridgeProtocol(asyncio.DatagramProtocol):
    def __init__(self, config: BridgeConfig, session_factory=ContinuousSession):
        self.session = session_factory(config)

    def connection_made(self, transport):
        self.session.connection_made(transport)

    def datagram_received(self, data, addr):
        self.session.datagram_received(data, addr)

    def connection_lost(self, exc):
        self.session.connection_lost(exc)
