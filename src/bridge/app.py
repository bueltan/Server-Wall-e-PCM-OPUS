import asyncio
import math
from bridge.config import BridgeConfig
from bridge.diagnostics.logging import configure_logging, event
from bridge.transport.udp import UdpAudioBridgeProtocol


async def async_main() -> None:
    configure_logging()
    config = BridgeConfig()
    event("BOOT", provider=config.voice_provider, protocol=3, port=config.port, rate=config.input_sample_rate,
          reorder_ms=config.input_reorder_window_sec*1000, recovery_ms=config.input_recovery_window_sec*1000,
          silence_max_ms=config.input_silence_max_ms)
    if config.calendar_reminders_enabled:
        if config.voice_provider != 'gemini' or not config.calendar_mcp_enabled:
            raise ValueError('Calendar reminders require Gemini and PIUU_CALENDAR_MCP_ENABLED=1')
        if not math.isfinite(config.calendar_poll_sec) or config.calendar_poll_sec < 60:
            raise ValueError('PIUU_CALENDAR_POLL_SEC must be finite and at least 60')
        if not 1 <= config.calendar_reminder_minutes <= 60:
            raise ValueError('PIUU_CALENDAR_REMINDER_MINUTES must be between 1 and 60')
    loop = asyncio.get_running_loop()

    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: UdpAudioBridgeProtocol(config),
        local_addr=(config.host, config.port),
    )

    reminders = None
    try:
        if config.calendar_reminders_enabled:
            from bridge.notifications.calendar import CalendarReminders
            reminders = CalendarReminders(config, _protocol.session.deliver_notification)
            reminders.start()
        await asyncio.Future()
    finally:
        if reminders is not None:
            await reminders.close()
        transport.close()
