"""Compatibility entrypoint: python src/main.py.

Public legacy class names remain importable here; implementations live in bridge.
"""
import asyncio
from bridge.config import BridgeConfig
from bridge.protocol import BasePacketFormat, PcmPacketFormat, OpusPacketFormat
from bridge.recording import RollingWavRecorder
from bridge.codec import OpusError, LibOpusBase, LibOpusEncoder
from bridge.xai import XaiRealtimeBridge
from bridge.udp import UdpAudioBridgeProtocol
from bridge.app import async_main

if __name__ == "__main__":
    asyncio.run(async_main())
