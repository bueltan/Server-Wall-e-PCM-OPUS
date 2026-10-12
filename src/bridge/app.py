import asyncio
from bridge.config import BridgeConfig
from bridge.diagnostics.logging import configure_logging, event
from bridge.transport.udp import UdpAudioBridgeProtocol


async def async_main() -> None:
    configure_logging()
    config = BridgeConfig()
    event("BOOT", protocol=3, port=config.port, rate=config.input_sample_rate,
          reorder_ms=config.input_reorder_window_sec*1000, recovery_ms=config.input_recovery_window_sec*1000,
          silence_max_ms=config.input_silence_max_ms)
    loop = asyncio.get_running_loop()

    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: UdpAudioBridgeProtocol(config),
        local_addr=(config.host, config.port),
    )

    try:
        await asyncio.Future()
    finally:
        transport.close()
