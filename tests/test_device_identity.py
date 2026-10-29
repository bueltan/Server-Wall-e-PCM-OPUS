"""Device MAC binding, address migration and retired-session isolation, offline."""
import unittest
import asyncio
from dataclasses import replace
from bridge.config import BridgeConfig
from bridge.conversation.session import ContinuousSession
from bridge.transport.protocols.conversation import CAP_DEVICE_MAC, Identity, Kind, Packet, encode, decode
from test_v3 import settle

MAC = bytes.fromhex('02123456789a')
OTHER = bytes.fromhex('02123456789b')
OLD = ('192.0.2.1', 12000)
NEW = ('198.51.100.2', 54321)

class Transport:
    def __init__(self): self.sent = []
    def sendto(self, data, addr): self.sent.append((decode(data), addr))

class DeviceIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.s = ContinuousSession(replace(BridgeConfig(), voice_provider='xai',
            save_input_wav=False, save_output_wav=False))
        self.s.transport = Transport()
        self.s.events.start()
        self.closed = 0
        async def close(): self.closed += 1
        self.s.xai.close = close
        self.hello()

    async def asyncTearDown(self):
        self.s.connection_lost(None)
        await settle()

    def hello(self, client=11, addr=OLD, mac=MAC, flags=CAP_DEVICE_MAC):
        self.s.datagram_received(encode(Packet(Kind.HELLO, flags,
            Identity(client, 0, 0, 0), 1, mac)), addr)

    def test_initial_binding_and_keepalive(self):
        self.assertEqual(self.s.device_mac, MAC)
        reply, addr = self.s.transport.sent[-1]
        self.assertEqual((reply.kind, addr), (Kind.HELLO_ACK, OLD))
        self.assertTrue(reply.flags & CAP_DEVICE_MAC)
        nonce = self.s.server_nonce
        self.hello()
        self.assertEqual(self.s.server_nonce, nonce)

    async def test_idle_migration_rotates_session_and_discards_old_packets(self):
        old_server = self.s.server_nonce
        self.s.last_conversation = 99
        self.hello(12, NEW)
        self.assertEqual((self.s.remote_addr, self.s.client_nonce), (NEW, 12))
        self.assertEqual(self.s.device_mac, MAC)
        self.assertNotEqual(self.s.server_nonce, old_server)
        self.assertEqual(self.s.last_conversation, 0)
        self.assertIsNone(self.s.conversation.identity)  # A fresh START is required.
        self.assertEqual(self.s.transport.sent[-1][1], NEW)
        seen = self.s.last_peer_seen
        count = len(self.s.transport.sent)
        self.hello(11, OLD)
        self.s.datagram_received(encode(Packet(Kind.PING, 0,
            Identity(11, old_server, 0, 0), 2)), OLD)
        self.assertEqual(len(self.s.transport.sent), count)
        self.assertEqual(self.s.last_peer_seen, seen)
        self.assertEqual(self.s.remote_addr, NEW)

    async def test_active_migration_closes_provider_before_admission(self):
        identity = Identity(11, self.s.server_nonce, 1, 0)
        self.s.conversation.open(identity)
        self.s.startup.admit(b'old audio')
        self.hello(12, NEW)
        self.assertEqual(self.s.remote_addr, OLD)
        self.assertTrue(self.s.peer_resetting)
        await settle()
        self.assertGreater(self.closed, 0)
        self.assertIsNone(self.s.conversation.identity)
        self.assertFalse(self.s.startup.pending)
        self.hello(12, NEW)
        self.assertEqual(self.s.remote_addr, NEW)
        self.assertEqual(self.s.client_nonce, 12)

    async def test_queued_control_cannot_reopen_old_session_during_rebind(self):
        self.s.conversation.open(Identity(11, self.s.server_nonce, 1, 0))
        start = Packet(Kind.START, 0, Identity(11, self.s.server_nonce, 2, 0), 2)
        self.s.datagram_received(encode(start), OLD)
        self.s.tasks.append(asyncio.create_task(self.s.run_controls()))
        self.hello(12, NEW)
        await settle()
        self.assertFalse(self.s.pending)
        self.assertIsNone(self.s.conversation.identity)
        self.assertTrue(any(p.kind==Kind.START_ACK and not p.flags for p, _ in self.s.transport.sent))
        self.hello(12, NEW)
        self.assertEqual(self.s.remote_addr, NEW)

    async def test_retries_do_not_redirect_traffic_before_provider_close_finishes(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def close():
            entered.set()
            await release.wait()
        self.s.xai.close = close
        self.s.conversation.open(Identity(11, self.s.server_nonce, 1, 0))
        self.hello(12, NEW)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.hello(12, NEW)
            self.assertEqual(self.s.remote_addr, OLD)
            self.assertFalse(any(p.kind==Kind.HELLO_ACK and addr==NEW for p, addr in self.s.transport.sent))
        finally:
            release.set()
            await settle()
        self.hello(12, NEW)
        self.assertEqual(self.s.remote_addr, NEW)

    def test_different_or_missing_mac_cannot_replace_bound_device(self):
        self.s.conversation.open(Identity(11, self.s.server_nonce, 1, 0))
        count = len(self.s.transport.sent)
        for addr in (OLD, NEW):
            self.hello(12, addr, OTHER)
            self.hello(12, addr, b'', 0)
        self.assertEqual(len(self.s.transport.sent), count)
        self.assertIsNotNone(self.s.conversation.identity)
        self.assertFalse(self.s.peer_resetting)

    def test_address_change_requires_new_session_nonce(self):
        self.hello(11, NEW)
        self.assertEqual(self.s.remote_addr, OLD)

    async def test_port_only_and_ip_only_changes(self):
        for nonce, addr in ((12, (OLD[0], 54321)), (13, ('198.51.100.2', 54321))):
            with self.subTest(addr=addr):
                self.hello(nonce, addr)
                self.assertEqual(self.s.remote_addr, addr)
                self.assertEqual(self.s.client_nonce, nonce)

    def test_malformed_mac_does_not_refresh_presence(self):
        seen = self.s.last_peer_seen
        count = len(self.s.transport.sent)
        for mac in (b'', bytes(6), b'12345', b'1234567', bytes.fromhex('01123456789a')):
            self.hello(12, OLD, mac)
        self.hello(12, OLD, MAC, 0)
        self.assertEqual(len(self.s.transport.sent), count)
        self.assertEqual(self.s.last_peer_seen, seen)

    def test_mac_cannot_change_on_same_session(self):
        self.hello(11, OLD, OTHER)
        self.assertEqual(self.s.device_mac, MAC)
        self.assertEqual(self.s.client_nonce, 11)
