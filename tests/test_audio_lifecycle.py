"""Deterministic connection/reset races at audio awaits; no live providers."""
import asyncio
import base64
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from bridge.audio.pipeline import AudioPipeline
from bridge.config import BridgeConfig
from bridge.providers.gemini.live import GeminiLive
from bridge.providers.xai.vad import VadXai
from bridge.transport.protocols.conversation import Identity


def audio_event():
    return {'serverContent': {'modelTurn': {'parts': [{'inlineData': {
        'mimeType': 'audio/pcm;rate=24000',
        'data': base64.b64encode(bytes(960)).decode()}}]}, 'turnComplete': True}}


class AudioLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def gemini(self):
        queue = asyncio.Queue()
        self.events = []
        self.current = None
        def callback(kind, message):
            self.events.append(kind)
            if kind == 'response.created':
                self.current = Identity(1, 2, 4, len(self.events))
                return self.current
        provider = GeminiLive(BridgeConfig(), queue,
                              lambda token: token is not None and token == self.current, callback)
        provider.ws = AsyncMock()
        provider.ready_event.set()
        provider.connected_event.set()
        return provider

    async def test_gemini_uplink_waiting_for_send_lock_cannot_use_replacement_socket(self):
        provider = self.gemini()
        old, fresh = provider.ws, AsyncMock()
        await provider.send_lock.acquire()
        task = asyncio.create_task(provider.send_audio_append(bytes(640)))
        try:
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            provider.ws = fresh
            provider.send_lock.release()
            with self.assertRaisesRegex(RuntimeError, 'websocket'):
                await task
            old.send.assert_not_called()
            fresh.send.assert_not_called()
            self.assertTrue(await provider.send_audio_append(bytes(640)))
            fresh.send.assert_awaited_once()
        finally:
            if provider.send_lock.locked(): provider.send_lock.release()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await provider.close()

    async def test_xai_uplink_revalidates_socket_and_connected_state_after_send_lock(self):
        for reconnect in (False, True):
            with self.subTest(reconnect=reconnect):
                provider = VadXai(BridgeConfig(), asyncio.Queue(), lambda token: True, lambda *args: None)
                old, fresh = AsyncMock(), AsyncMock()
                provider.ws = old
                provider.connected_event.set()
                await provider.send_lock.acquire()
                task = asyncio.create_task(provider.send_audio_append(bytes(640)))
                try:
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    if reconnect:
                        provider.ws = fresh
                    else:
                        provider.connected_event.clear()
                    provider.send_lock.release()
                    self.assertFalse(await task)
                    old.send.assert_not_called()
                    fresh.send.assert_not_called()
                    provider.ws = fresh
                    provider.connected_event.set()
                    self.assertTrue(await provider.send_audio_append(bytes(640)))
                    fresh.send.assert_awaited_once()
                finally:
                    if provider.send_lock.locked(): provider.send_lock.release()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await provider.close()

    async def test_cancelled_gemini_audio_send_is_not_replayed_after_unlock(self):
        provider = self.gemini()
        socket = provider.ws
        await provider.send_lock.acquire()
        task = asyncio.create_task(provider.send_audio_append(bytes(640)))
        try:
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            provider.send_lock.release()
            self.assertTrue(await provider.send_audio_append(bytes(640)))
            socket.send.assert_awaited_once()
        finally:
            if provider.send_lock.locked(): provider.send_lock.release()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await provider.close()

    async def test_gemini_close_invalidates_waiting_sends_before_tool_cleanup_finishes(self):
        provider = self.gemini()
        old = provider.ws
        entered, release = asyncio.Event(), asyncio.Event()
        async def cleanup():
            entered.set()
            await release.wait()
        provider.tool_manager.close = cleanup
        task = asyncio.create_task(provider.close())
        try:
            await entered.wait()
            with self.assertRaisesRegex(RuntimeError, 'websocket'):
                await provider._send({'realtimeInput': {'audio': {}}})
            old.send.assert_not_called()
        finally:
            release.set()
            await task
            old.close.assert_awaited_once()

    async def test_gemini_old_audio_or_tool_event_cannot_resume_into_new_connection(self):
        for with_tool in (False, True):
            with self.subTest(with_tool=with_tool):
                provider = self.gemini()
                provider.pause_input()
                message = audio_event()
                if with_tool:
                    message['toolCall'] = {'functionCalls': [
                        {'id': 'old-call', 'name': 'unknown', 'args': {}}]}
                task = asyncio.create_task(provider.handle_event(message))
                try:
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    await provider.close()
                    provider.ws = AsyncMock()
                    fresh = Identity(1, 2, 4, 99)
                    provider.token = self.current = fresh
                    provider.serial = 100
                    provider.audio_started = True
                    resampler = provider.resampler
                    provider.history_ready.set()
                    await task
                    self.assertEqual(self.events, [])
                    self.assertTrue(provider.output_pcm_queue.empty())
                    self.assertEqual(provider.token, fresh)
                    self.assertEqual(provider.serial, 100)
                    self.assertTrue(provider.audio_started)
                    self.assertIs(provider.resampler, resampler)
                    provider.ws.send.assert_not_called()
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await provider.close()

    async def test_obsolete_gemini_receiver_does_not_handle_audio_or_clear_new_readiness(self):
        for with_audio in (False, True):
            with self.subTest(with_audio=with_audio):
                provider = self.gemini()
                entered, release = asyncio.Event(), asyncio.Event()
                class OldSocket:
                    async def __aiter__(self):
                        entered.set()
                        await release.wait()
                        if with_audio:
                            yield json.dumps(audio_event())
                provider.ws = OldSocket()
                task = asyncio.create_task(provider.receive_events_forever())
                try:
                    await entered.wait()
                    provider.ws = AsyncMock()
                    release.set()
                    await task
                    self.assertEqual(self.events, [])
                    self.assertTrue(provider.output_pcm_queue.empty())
                    self.assertTrue(provider.ready_event.is_set())
                    self.assertTrue(provider.connected_event.is_set())
                finally:
                    release.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await provider.close()

    async def test_obsolete_xai_receiver_cannot_interrupt_new_audio_or_clear_readiness(self):
        for with_event in (False, True):
            with self.subTest(with_event=with_event):
                events = []
                provider = VadXai(BridgeConfig(), asyncio.Queue(), lambda token: True,
                                  lambda kind, message: events.append(kind))
                entered, release = asyncio.Event(), asyncio.Event()
                class OldSocket:
                    async def __aiter__(self):
                        entered.set()
                        await release.wait()
                        if with_event:
                            yield json.dumps({'type': 'input_audio_buffer.speech_started'})
                provider.ws = OldSocket()
                provider.ready_event.set()
                provider.connected_event.set()
                task = asyncio.create_task(provider.receive_events_forever())
                try:
                    await entered.wait()
                    provider.ws = AsyncMock()
                    release.set()
                    await task
                    self.assertEqual(events, [])
                    self.assertTrue(provider.ready_event.is_set())
                    self.assertTrue(provider.connected_event.is_set())
                finally:
                    release.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await provider.close()


class DownlinkLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_wait_cannot_finish_or_modify_stream_after_reset(self):
        for partial_frame in (False, True):
            with self.subTest(partial_frame=partial_frame):
                config = replace(BridgeConfig(), save_input_wav=False, save_output_wav=False)
                pipeline = AudioPipeline(config)
                token = Identity(1, 2, 3, 1)
                wire = []
                pipeline.output_valid = lambda identity: identity == token
                pipeline._send = lambda *args: wire.append(args)
                pipeline.counters = {'opus_frames': 0, 'stale_output': 0}
                pipeline.downlink = SimpleNamespace(reset=lambda: None, remember=lambda *args: None,
                                                    finish=lambda *args: wire.append(args))
                entered, release = asyncio.Event(), asyncio.Event()
                async def pacing():
                    entered.set()
                    await release.wait()
                pipeline.output_pacer.wait = pacing
                if partial_frame:
                    pipeline.output_pending_pcm.extend(bytes(config.opus_frame_bytes // 2))
                pipeline.output_pcm_queue.put_nowait((token, None))
                task = asyncio.create_task(pipeline._udp_output_sender())
                try:
                    await entered.wait()
                    pipeline.reset_output()
                    # Keep the response token to isolate the stream-generation
                    # guard; normal interruptions also invalidate the token.
                    pipeline.output_sequence = 7
                    pipeline.output_pts_samples = 2240
                    pipeline.output_active = True
                    release.set()
                    for _ in range(10): await asyncio.sleep(0)
                    self.assertEqual(wire, [])
                    self.assertEqual(pipeline.output_sequence, 7)
                    self.assertEqual(pipeline.output_pts_samples, 2240)
                    self.assertTrue(pipeline.output_active)
                    self.assertFalse(pipeline.output_finished)
                    self.assertFalse(task.done())
                    pipeline.output_pcm_queue.put_nowait((token, bytes(config.opus_frame_bytes)))
                    pipeline.output_pcm_queue.put_nowait((token, None))
                    async with asyncio.timeout(1):
                        while not pipeline.output_finished:
                            if task.done(): await task
                            await asyncio.sleep(.001)
                    self.assertEqual(len(wire), 2)
                    self.assertEqual(wire[0][2], 7) # new audio starts at its own sequence
                    self.assertEqual(wire[1][1], 8) # END follows the one new frame
                    self.assertEqual(pipeline.counters['opus_frames'], 1)
                finally:
                    release.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    pipeline.opus_encoder.close()
