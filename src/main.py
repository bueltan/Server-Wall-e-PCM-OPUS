"""Compatibility entrypoint: python src/main.py.

Public legacy class names remain importable here; implementations live in bridge.
"""
import asyncio
from bridge.config import BridgeConfig
from bridge.transport.protocols.audio import BasePacketFormat, PcmPacketFormat, OpusPacketFormat
from bridge.audio.recording import RollingWavRecorder
from bridge.audio.codec import OpusError, LibOpusBase, LibOpusEncoder
from bridge.providers.xai.realtime import XaiRealtimeBridge
from bridge.transport.udp import UdpAudioBridgeProtocol
from bridge.app import async_main

if __name__ == "__main__":
    asyncio.run(async_main())
