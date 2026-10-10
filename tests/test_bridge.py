"""Offline regression checks. No UDP sockets or xAI connections are opened."""
import asyncio
import importlib
import struct
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge.config import BridgeConfig
from bridge.protocol import PcmPacketFormat, OpusPacketFormat
from bridge.recording import RollingWavRecorder
from bridge.xai import XaiRealtimeBridge
from bridge.udp import UdpAudioBridgeProtocol
from bridge.session import ClientSession

class ProtocolTests(unittest.TestCase):
    def test_core_wire_contract_and_rollover_values(self):
        fmt=PcmPacketFormat(BridgeConfig())
        payload=struct.pack('<hh',123,-456)*160
        data=fmt.pack_audio_packet(0xffffffff,0xffffff00,320,payload)
        self.assertEqual(data[:18],struct.pack('!4sBBIIHH',b'PCM!',1,0,0xffffffff,0xffffff00,320,640))
        packet=fmt.unpack_packet(data);fmt.validate_packet(packet)
        self.assertEqual(packet['payload'],payload)
        self.assertFalse(fmt.is_end_packet(packet))
        end=fmt.unpack_packet(fmt.pack_end_packet(7,2240));fmt.validate_packet(end)
        self.assertTrue(fmt.is_end_packet(end));self.assertEqual(end['payload'],b'')

    def test_bad_magic_version_truncation_length(self):
        fmt=PcmPacketFormat(BridgeConfig());valid=fmt.pack_audio_packet(0,0,320,b'\0'*640)
        for data in (valid[:17],b'BAD!'+valid[4:],valid[:4]+b'\2'+valid[5:],valid[:-1],valid+b'\0'):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                fmt.validate_packet(fmt.unpack_packet(data))

    def test_legacy_entrypoint_exports(self):
        main=importlib.import_module('main')
        self.assertIs(main.PcmPacketFormat,PcmPacketFormat)
        self.assertIs(main.UdpAudioBridgeProtocol,UdpAudioBridgeProtocol)

    def test_recording_format_and_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=RollingWavRecorder(16000,1,2,Path(directory))
            recorder.start('test');path=recorder.path
            recorder.write(b'\0'*640);recorder.close();recorder.close()
            with wave.open(str(path),'rb') as wav:
                self.assertEqual((wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getnframes()),(16000,1,2,320))

class FakeEncoder:
    def __init__(self, config): self.frames=[];self.closed=False
    def encode_pcm16_frame(self, frame): self.frames.append(frame);return b'opus'
    def close(self): self.closed=True

class FakeTransport:
    def __init__(self): self.sent=[]
    def sendto(self, data, addr): self.sent.append((data,addr))

class FakeXai:
    def __init__(self): self.frames=[];self.commits=0
    async def send_audio_append(self, data): self.frames.append(data);return True
    async def send_commit(self): self.commits+=1;return True

class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.config=replace(BridgeConfig(),save_input_wav=False,save_output_wav=False,
            recordings_dir=self.directory.name,output_send_interval_sec=0,output_end_repeat_interval=0)
        with patch('bridge.session.LibOpusEncoder',FakeEncoder): self.bridge=ClientSession(self.config)
        self.bridge.transport=FakeTransport();self.bridge.xai=FakeXai()
        self.addr=('127.0.0.1',12000)

    async def asyncTearDown(self):
        self.bridge.connection_lost(None);self.directory.cleanup()

    async def test_pcm_end_commit_wire_compatibility(self):
        bridge=self.bridge;fmt=bridge.pcm_packet_format;payload=b'\1\0'*320
        bridge.input.start()
        bridge.datagram_received(fmt.pack_audio_packet(0,0,320,payload),self.addr)
        for _ in range(3): bridge.datagram_received(fmt.pack_end_packet(1,320),self.addr)
        bridge.datagram_received(b'COMMIT',self.addr)
        async with asyncio.timeout(1):
            while bridge.xai.commits != 1: await asyncio.sleep(.001)
        self.assertEqual(bridge.xai.frames,[payload]);self.assertFalse(bridge.inbound_started)
        self.assertEqual(bridge.input.stats.duplicate_controls,2)

    async def test_malformed_audio_never_reaches_xai(self):
        bridge=self.bridge
        bridge.datagram_received(b'PCM!',self.addr)
        bridge.datagram_received(bridge.pcm_packet_format.pack_audio_packet(0,0,320,b'\0'*2),self.addr)
        self.assertEqual(bridge.xai.frames,[]);self.assertIsNone(bridge.remote_addr)
        self.assertEqual(bridge.input.stats.invalid,2)

    async def test_ping_reply_without_xai(self):
        self.bridge.datagram_received(b'PING',self.addr)
        self.assertEqual(self.bridge.transport.sent,[(b'PONG',self.addr)])
        self.assertIsNone(self.bridge.remote_addr)

    async def test_peer_ownership_and_router_session_boundary(self):
        bridge=self.bridge;other=('127.0.0.2',12000)
        data=bridge.pcm_packet_format.pack_audio_packet(0,0,320,b'\0'*640)
        bridge.datagram_received(data,self.addr)
        bridge.datagram_received(b'PING',other)
        bridge.datagram_received(data,other);bridge.datagram_received(b'COMMIT',other)
        self.assertEqual(bridge.remote_addr,self.addr)
        self.assertEqual(bridge.input.stats.accepted,1)
        self.assertFalse(bridge.input.commit_admitted)
        router=UdpAudioBridgeProtocol(self.config,session_factory=lambda config: bridge)
        self.assertIs(router.session,bridge)

    async def test_chunking_padding_and_end_repeats(self):
        bridge=self.bridge;bridge.remote_addr=self.addr;bridge.output_active=True
        bridge.output_pending_pcm.extend(b'\1\0'*325)
        await bridge._flush_pending_pcm(final_flush=True)
        self.assertEqual(bridge.opus_encoder.frames,[b'\1\0'*320,b'\1\0'*5+b'\0'*630])
        await bridge._send_output_end_packets()
        packets=[bridge.opus_packet_format.unpack_packet(data) for data,_ in bridge.transport.sent]
        self.assertEqual([p['seq'] for p in packets[:2]],[0,1])
        self.assertEqual([p['pts_samples'] for p in packets[:2]],[0,320])
        self.assertEqual(len(packets),2+self.config.output_end_repeat_count)
        for p in packets[2:]:
            self.assertTrue(bridge.opus_packet_format.is_end_packet(p));self.assertEqual(p['seq'],2)
        self.assertFalse(bridge.output_active)

    async def test_turn_reset_discards_local_pending_output(self):
        bridge=self.bridge;bridge.output_pending_pcm.extend(b'stale');bridge.output_sequence=9
        bridge.output_pcm_queue.put_nowait(b'stale');bridge.output_pcm_queue.put_nowait(None)
        bridge._reset_downlink_turn_state()
        self.assertEqual(bridge.output_sequence,0);self.assertFalse(bridge.output_pending_pcm)
        self.assertTrue(bridge.output_pcm_queue.empty())

    async def test_xai_session_and_send_order_without_network(self):
        client=XaiRealtimeBridge(self.config,asyncio.Queue());sent=[]
        async def send(payload): sent.append(payload)
        client._safe_send=send;client.connected_event.set()
        await client._send_session_update()
        self.assertEqual(sent[0]['session']['audio']['input']['format']['rate'],16000)
        self.assertEqual(sent[0]['session']['audio']['output']['format']['rate'],16000)
        self.assertIsNone(sent[0]['session']['turn_detection'])
        await client.send_audio_append(b'\1\0'*320);await client.send_commit()
        self.assertEqual([p['type'] for p in sent[1:]],['input_audio_buffer.append','input_audio_buffer.commit','response.create'])
        client.connected_event.clear()
        self.assertFalse(await client.send_commit());self.assertEqual(len(sent),4)

if __name__=='__main__': unittest.main()
