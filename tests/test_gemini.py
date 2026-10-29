"""Offline Gemini wire, DSP and real UDP/Opus integration regressions."""
import asyncio
import base64
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch
import numpy as np
from bridge.audio.resample import Pcm24To16
from bridge.config import BridgeConfig
from bridge.providers import create_provider
from bridge.providers.gemini.live import GeminiLive
from bridge.providers.xai.vad import VadXai
from bridge.conversation.session import ContinuousSession
from bridge.transport.protocols.conversation import Identity, Kind, Event, Packet, encode
from test_v3 import Transport, settle


def audio(pcm=bytes(960)):
    return {'serverContent': {'modelTurn': {'parts': [{'inlineData': {
        'mimeType': 'audio/pcm;rate=24000', 'data': base64.b64encode(pcm).decode()}}]}}}


def tool(cid='call-1', **overrides):
    return {'toolCall': {'functionCalls': [dict({'id': cid, 'name': 'set_expression_sequence',
        'args': {'sequence': [{'emotion': 'happy', 'gesture': 'blink', 'duration_ms': 1200}]}}, **overrides)]}}


class ResamplerTests(unittest.TestCase):
    def test_chunk_boundaries_preserve_samples_phase_and_tail(self):
        source = np.random.default_rng(3).integers(-20000, 20000, 24001, dtype=np.int16).tobytes()
        whole = Pcm24To16()
        expected = whole.convert(source) + whole.finish()
        split = Pcm24To16()
        chunks = [split.convert(source[i:i+734]) for i in range(0, len(source), 734)]
        self.assertEqual(b''.join(chunks) + split.finish(), expected)
        self.assertEqual(split.finish(), b'')
        self.assertEqual(len(expected)//2, (24001*2+2)//3 + 40)

    def test_anti_alias_filter_preserves_voice_and_rejects_10khz(self):
        energies = []
        for freq in (1000, 10000):
            x = np.rint(10000*np.sin(2*np.pi*freq*np.arange(24000)/24000)).astype('<i2')
            y = np.frombuffer(Pcm24To16().convert(x.tobytes()), dtype='<i2')[100:].astype(float)
            energies.append(np.sqrt(np.mean(y*y)))
        self.assertGreater(energies[0], 6900)
        self.assertLess(energies[1], energies[0]*.01)

    def test_reject_partial_sample(self):
        with self.assertRaises(ValueError): Pcm24To16().convert(b'x')


class GeminiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.queue = asyncio.Queue(128)
        self.current = None
        self.serial = 0
        self.events = []
        def callback(kind, message):
            self.events.append((kind, message))
            if kind == 'response.created':
                self.serial += 1
                self.current = self.serial
                return self.current
            if kind == 'input_audio_buffer.speech_started':
                self.current = None
        self.g = GeminiLive(replace(BridgeConfig(), voice_provider='gemini'), self.queue,
                            lambda token: token == self.current, callback)
        self.g.ws = AsyncMock()

    async def test_setup_waits_for_confirmation_and_uses_header_auth(self):
        ws = AsyncMock()
        g = self.g
        g.config = replace(g.config, gemini_api_key='test-key')
        with patch('bridge.providers.gemini.live.websockets.connect', AsyncMock(return_value=ws)) as connect:
            await g.connect()
        self.assertNotIn('test-key', connect.call_args.args[0])
        self.assertEqual(connect.call_args.kwargs['additional_headers']['x-goog-api-key'], 'test-key')
        self.assertFalse(g.ready_event.is_set())
        self.assertFalse(await g.send_audio_append(bytes(640)))
        setup = json.loads(ws.send.call_args.args[0])['setup']
        self.assertEqual(setup['model'], 'models/gemini-3.8-live')
        self.assertEqual(setup['contextWindowCompression'], {'slidingWindow': {}})
        self.assertNotIn('thinkingConfig', setup['generationConfig'])
        self.assertEqual(setup['tools'][0]['functionDeclarations'][0]['behavior'], 'BLOCKING')
        await g.handle_event({'setupComplete': {}})
        self.assertTrue(await g.send_audio_append(bytes(640)))
        msg = json.loads(ws.send.call_args.args[0])
        self.assertEqual(msg['realtimeInput']['audio']['mimeType'], 'audio/pcm;rate=16000')

    async def test_tool_plan_precedes_audio_and_same_response_finishes_once(self):
        await self.g.handle_event(tool())
        self.assertTrue(self.queue.empty())
        await self.g.handle_event(audio())
        token, pcm = self.queue.get_nowait()
        self.assertEqual(len(pcm), 640)
        self.assertEqual([k for k, _ in self.events], ['response.created', 'expression.plan'])
        self.assertEqual(self.events[-1][1]['identity'], token)
        await self.g.handle_event({'serverContent': {'generationComplete': True}})
        self.assertTrue(self.queue.empty())
        await self.g.handle_event({'serverContent': {'turnComplete': True}})
        self.assertIsNotNone(self.queue.get_nowait()[1])  # FIR tail
        self.assertEqual(self.queue.get_nowait(), (token, None))
        await self.g.handle_event({'serverContent': {'turnComplete': True}})
        self.assertTrue(self.queue.empty())
        self.assertEqual(self.serial, 1)

    async def test_invalid_duplicate_and_late_tools(self):
        await self.g.handle_event(tool(args={'sequence': []}))
        first = json.loads(self.g.ws.send.call_args.args[0])
        self.assertFalse(first['toolResponse']['functionResponses'][0]['response']['ok'])
        await self.g.handle_event(tool(args={'sequence': []}))
        self.assertEqual(json.loads(self.g.ws.send.call_args.args[0]), first)
        await self.g.handle_event(audio())
        await self.g.handle_event(tool(cid='late'))
        result = json.loads(self.g.ws.send.call_args.args[0])['toolResponse']['functionResponses'][0]
        self.assertEqual(result['response']['error'], 'audio_already_started')
        self.assertFalse(any(k == 'expression.plan' for k, _ in self.events))

    async def test_interruption_drops_old_audio_and_plan_then_allows_next_turn(self):
        await self.g.handle_event(tool())
        token = self.current
        await self.g.handle_event({'voiceActivity': {'type': 'ACTIVITY_START'}})
        await self.g.handle_event(audio())
        self.assertTrue(self.queue.empty())
        repair = asyncio.create_task(self.g.repair_history(token, 20))
        await settle()
        self.assertFalse(repair.done())
        await self.g.handle_event({'voiceActivity': {'type': 'ACTIVITY_END'}})
        await self.g.handle_event({'serverContent': {'interrupted': True}})
        self.assertEqual(sum(k == 'input_audio_buffer.speech_started' for k, _ in self.events), 1)
        await repair
        await self.g.handle_event({'serverContent': {'turnComplete': True}})
        self.assertTrue(self.queue.empty())
        await self.g.handle_event(audio())
        self.assertNotEqual(self.queue.get_nowait()[0], token)
        self.assertFalse(any(k == 'expression.plan' for k, _ in self.events))

    async def test_new_response_waits_for_core_ack_barrier(self):
        self.g.pause_input()
        task = asyncio.create_task(self.g.handle_event(audio()))
        await settle()
        self.assertTrue(self.queue.empty())
        self.g.resume_input()
        await task
        self.assertFalse(self.queue.empty())

    async def test_cancelled_tool_never_delivers_plan(self):
        await self.g.handle_event(tool())
        await self.g.handle_event({'toolCallCancellation': {'ids': ['call-1']}})
        await self.g.handle_event(audio())
        self.assertFalse(any(k == 'expression.plan' for k, _ in self.events))

    async def test_unsupported_manual_cancel_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'new conversation'):
            await self.g.cancel_response(1)
        self.g.ws.send.assert_not_called()

    async def test_shutdown_and_invalid_format_fail_without_replay(self):
        with self.assertRaises(RuntimeError):
            await self.g.handle_event({'goAway': {'timeLeft': '1s'}})
        msg = audio()
        msg['serverContent']['modelTurn']['parts'][0]['inlineData']['mimeType'] = 'audio/pcm;rate=16000'
        with self.assertRaises(ValueError): await self.g.handle_event(msg)
        self.assertTrue(self.queue.empty())

    async def test_no_key_and_provider_selection(self):
        g = GeminiLive(replace(self.g.config, gemini_api_key=None), self.queue, lambda t: True, lambda *a: None)
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'GEMINI_API_KEY'): await g.connect()
        args = (self.queue, lambda t: True, lambda *a: None)
        self.assertIsInstance(create_provider(replace(BridgeConfig(), voice_provider='xai'), *args), VadXai)
        with self.assertRaises(ValueError):
            create_provider(replace(BridgeConfig(), voice_provider='bad'), *args)


class GeminiSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_coordinator_opus_output_and_native_interrupt(self):
        config = replace(BridgeConfig(), voice_provider='gemini', save_input_wav=False,
                         save_output_wav=False, output_send_interval_sec=0,
                         output_end_repeat_interval=0)
        s = ContinuousSession(config)
        s.transport = Transport()
        s.remote_addr = ('127.0.0.1', 4567)
        s.client_nonce = 11
        identity = Identity(11, s.server_nonce, 1, 0)
        s.conversation.open(identity)
        s.events.start()
        s.xai.ws = AsyncMock()
        sender = asyncio.create_task(s._udp_output_sender())
        async def ack_events():
            for _ in range(8):
                await settle()
                for p in list(s.transport.sent):
                    if p.kind == Kind.EVENT:
                        s.datagram_received(encode(Packet(Kind.EVENT_ACK, 0, p.identity,
                            p.sequence, (20).to_bytes(4, 'big'))), s.remote_addr)
            await settle()
        try:
            await s.xai.handle_event(tool())
            await s.xai.handle_event(audio(bytes(1920)))
            first = s.conversation.response
            await ack_events()
            self.assertTrue(any(p.kind == Kind.OPUS and p.identity == first for p in s.transport.sent))
            await s.xai.handle_event({'serverContent': {'interrupted': True}})
            self.assertIsNone(s.conversation.response)
            await ack_events()
            self.assertFalse(s.interruptions.pending)
            self.assertIsNotNone(s.conversation.identity)
            await s.xai.handle_event({'serverContent': {'turnComplete': True}})
            await s.xai.handle_event(audio())
            second = s.conversation.response
            self.assertNotEqual(first, second)
            await ack_events()
            await s.xai.handle_event({'serverContent': {'turnComplete': True}})
            await settle()
            self.assertTrue(any(p.kind == Kind.OPUS and p.identity == second for p in s.transport.sent))
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            s.connection_lost(None)
            await settle()
