"""Offline wire vectors, negotiation, delivery faults and the real Opus coordinator."""
import asyncio
import base64
import struct
import unittest
from dataclasses import replace
import test_v3
from bridge.config import BridgeConfig
from bridge.conversation.session import ContinuousSession
from bridge.expressions.delivery import EyePlanDelivery
from bridge.expressions.plan import Plan, Step
from bridge.transport.protocols.eyes import encode_plan, decode_plan
from bridge.transport.protocols.conversation import Kind, Identity, Packet, encode, decode, CAP_EYES_PLAN

PLAN = Plan((Step('happy', 'soft_emphasis', 2400), Step('curious', 'question', 1800)))
TOKEN = Identity(1, 2, 3, 4)
# Hand-authored golden vector: version/count/reserved, then emotion/gesture/duration/offset.
BODY = bytes.fromhex('01 02 00 00 01 07 09 60 00 00 00 00 02 02 07 08 00 00 09 60')


class WireTests(unittest.TestCase):
    def test_exact_wire_and_enum_ids(self):
        self.assertEqual(encode_plan(PLAN), BODY)
        self.assertEqual(decode_plan(BODY), PLAN)
        packet = Packet(Kind.EYES_PLAN, 0, TOKEN, 1, BODY)
        self.assertEqual(encode(packet), bytes.fromhex(
            '50 49 55 33 03 1a 00 00 00 00 00 00 00 00 00 01 '
            '00 00 00 00 00 00 00 02 00 00 00 03 00 00 00 04 '
            '00 00 00 01 00 14') + BODY)
        self.assertEqual(decode(encode(packet)), packet)
        maximum = Plan((Step('neutral', 'none', 4000),) * 3)
        self.assertEqual(len(encode_plan(maximum)), 28)
        self.assertEqual(decode_plan(encode_plan(maximum)), maximum)

    def test_rejects_bad_version_counts_ids_offsets_and_lengths(self):
        invalid = [b'', BODY[:-1], BODY+b'\0', bytes(29)]
        for index, value in ((0, 2), (1, 0), (1, 4), (3, 1), (4, 7), (5, 9), (6, 0), (11, 1), (19, 0)):
            mutated = bytearray(BODY); mutated[index] = value; invalid.append(bytes(mutated))
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValueError): decode_plan(body)
        for plan in (Plan(()), Plan(PLAN.steps*2), Plan((Step('happy', 'none', True),)),
                     Plan((Step('happy', 'unknown', 1200),)), Plan((Step('happy', 'none', 4001),))):
            with self.assertRaises(ValueError): encode_plan(plan)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.delivery = EyePlanDelivery(self.sent.append)

    def test_loss_retry_ack_and_no_replay(self):
        d = self.delivery
        self.assertTrue(d.start(TOKEN, PLAN, 0))
        self.assertFalse(d.start(TOKEN, PLAN, .01))
        d.poll(.099); self.assertEqual(len(self.sent), 1)
        d.poll(.101); d.poll(.202)
        self.assertEqual(self.sent, [self.sent[0]]*3)
        for ack in (Packet(Kind.EYES_PLAN_ACK, 0, TOKEN, 2),
                    Packet(Kind.EYES_PLAN_ACK, 1, TOKEN, 1),
                    Packet(Kind.EYES_PLAN_ACK, 0, TOKEN, 1, b'x'),
                    Packet(Kind.EYES_PLAN_ACK, 0, replace(TOKEN, response=5), 1)):
            self.assertFalse(d.acknowledge(ack))
        self.assertTrue(d.acknowledge(Packet(Kind.EYES_PLAN_ACK, 0, TOKEN, 1)))
        self.assertFalse(d.start(TOKEN, PLAN, .3))
        d.poll(10); self.assertEqual(len(self.sent), 3)
        self.assertEqual(d.stats['acked'], 1)

    def test_timeout_and_late_poll_never_catch_up_with_burst(self):
        d = self.delivery; d.start(TOKEN, PLAN, 0); d.poll(.5)
        self.assertIsNone(d.pending); self.assertEqual(len(self.sent), 1)
        self.assertEqual(d.stats['unconfirmed'], 1)
        self.assertFalse(d.start(TOKEN, PLAN, 1))
        d.reset(); d.poll(2); self.assertEqual(len(self.sent), 1)

    def test_local_send_errors_are_bounded_and_nonfatal(self):
        def fail(packet): raise OSError('test UDP fault')
        d = EyePlanDelivery(fail); d.start(TOKEN, PLAN, 0)
        for now in (.101, .202, .303, .5, 10): d.poll(now)
        self.assertEqual(d.stats['attempts'], 3)
        self.assertEqual(d.stats['send_errors'], 3)
        self.assertEqual(d.stats['unconfirmed'], 1)


class NegotiationTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_new_disabled_unknown_flags_and_nonce_changes(self):
        for offered, enabled, expected in ((0, True, 3), (4, True, 7), (4, False, 3)):
            with self.subTest(offered=offered, enabled=enabled):
                cfg = replace(BridgeConfig(), expression_udp_enabled=enabled,
                              save_input_wav=False, save_output_wav=False)
                s = ContinuousSession(cfg); s.transport = test_v3.Transport()
                peer = ('127.0.0.1', 12000)
                def hello(nonce, flags, addr=peer):
                    s.datagram_received(encode(Packet(Kind.HELLO, flags, Identity(nonce, 0, 0, 0), 1)), addr)
                try:
                    hello(11, offered)
                    self.assertEqual(s.transport.sent[-1].flags, expected)
                    hello(11, offered); self.assertEqual(len(s.transport.sent), 2)
                    # Neither a changed offer nor an unknown bit renegotiates a live nonce.
                    hello(11, offered ^ 4); hello(11, 8)
                    hello(11, offered, ('127.0.0.2', 12000))
                    self.assertEqual(len(s.transport.sent), 2)
                    hello(12, 0)  # A restarted, idle peer can negotiate a new capability set.
                    self.assertEqual(s.transport.sent[-1].flags, 3)
                    self.assertFalse(s.eyes_capable)
                    hello(11, offered); self.assertEqual(len(s.transport.sent), 3)
                finally:
                    s.connection_lost(None); await test_v3.settle()


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_v3.SessionTests.asyncSetUp
    asyncTearDown = test_v3.SessionTests.asyncTearDown
    ack_events = test_v3.SessionTests.ack_events

    def deliver(self, kind, identity=None, seq=0, body=b'', peer=None):
        flags = CAP_EYES_PLAN if kind == Kind.HELLO else 0
        self.s.datagram_received(encode(Packet(kind, flags, identity or self.i, seq, body)), peer or self.peer)

    async def prepare(self, name='r1', plan=PLAN, audio=True):
        await self.s.xai.handle_event({'type':'response.created', 'response':{'id':name}})
        token = self.s.conversation.response
        if plan is not None: self.s.provider_event('expression.plan', {'identity':token, 'plan':plan})
        if audio:
            await self.s.xai.handle_event({'type':'response.output_audio.delta', 'response_id':name,
                'item_id':'audio-'+name, 'delta':base64.b64encode(bytes(640)).decode()})
            await self.s.xai.handle_event({'type':'response.done', 'response':{'id':name, 'status':'completed'}})
        return token

    def plans(self): return [p for p in self.s.transport.sent if p.kind == Kind.EYES_PLAN]

    async def test_plan_after_response_ack_before_opus_audio_does_not_wait_for_eye_ack(self):
        token = await self.prepare(); await test_v3.settle()
        self.assertFalse(self.plans())
        self.assertFalse(any(p.kind == Kind.OPUS for p in self.s.transport.sent))
        await self.ack_events(); await test_v3.settle()
        self.assertEqual(len(self.plans()), 1)
        packet = self.plans()[0]; self.assertEqual(packet.body, BODY)
        opus = next(p for p in self.s.transport.sent if p.kind == Kind.OPUS)
        self.assertLess(self.s.transport.sent.index(packet), self.s.transport.sent.index(opus))
        self.assertTrue(any(p.kind == Kind.AUDIO_END for p in self.s.transport.sent))
        # Send completion is not playback completion: a lost plan can still be retried.
        self.assertIsNotNone(self.s.eye_plans.pending)
        self.deliver(Kind.EYES_PLAN_ACK, token, 1)
        self.assertIsNone(self.s.eye_plans.pending)
        self.assertEqual(self.s.conversation.response, token)

    async def test_unconfirmed_plan_or_visual_socket_failure_does_not_abort_audio(self):
        original = self.s.eye_plans.send
        def fail(packet): raise OSError('eye-only local fault')
        self.s.eye_plans.send = fail
        token = await self.prepare(); await self.ack_events()
        d = self.s.eye_plans; d.poll(d.started + .5)
        self.assertEqual(d.stats['unconfirmed'], 1)
        self.assertEqual(self.s.conversation.response, token)
        self.assertTrue(any(p.kind == Kind.OPUS for p in self.s.transport.sent))
        self.assertFalse(any(p.kind == Kind.ERROR for p in self.s.transport.sent))
        d.send = original

    async def test_cancel_before_gate_open_sends_neither_plan_nor_audio(self):
        await self.prepare()
        self.deliver(Kind.CANCEL, self.i, 30, struct.pack('!I', 0))
        await test_v3.settle()
        self.assertFalse(self.plans())
        self.assertIsNone(self.s.eye_plans.pending)
        self.assertFalse(any(p.kind == Kind.OPUS for p in self.s.transport.sent))

    async def test_interrupt_drops_retry_and_rejects_old_ack(self):
        old = await self.prepare(); await self.ack_events()
        await self.s.xai.handle_event({'type':'input_audio_buffer.speech_started'})
        await self.ack_events()
        self.assertIsNone(self.s.eye_plans.pending)
        count = len(self.plans()); self.s.eye_plans.poll(1e20)
        self.assertEqual(len(self.plans()), count)
        new = await self.prepare('r2'); await self.ack_events()
        self.assertNotEqual(old, new)
        self.deliver(Kind.EYES_PLAN_ACK, old, 1)
        self.assertIsNotNone(self.s.eye_plans.pending)
        self.deliver(Kind.EYES_PLAN_ACK, new, 1)
        self.assertIsNone(self.s.eye_plans.pending)

    async def test_played_clears_pending_and_short_audio_does_not_emit_neutral_commands(self):
        token = await self.prepare(); await self.ack_events()
        self.deliver(Kind.PLAYED, token, 31, struct.pack('!I', 20)); await test_v3.settle()
        self.assertIsNone(self.s.eye_plans.pending)
        self.assertEqual(len(self.plans()), 1)
        self.assertFalse(any(p.kind == Kind.EYES for p in self.s.transport.sent))

    async def test_no_plan_no_audio_and_old_peer_are_silent(self):
        await self.prepare(audio=False); await self.ack_events()
        self.assertFalse(self.plans())
        self.s.eyes_capable = False  # Legacy negotiation is tested separately.
        await self.prepare('r2'); await self.ack_events()
        self.assertFalse(self.plans())
        self.assertTrue(any(p.kind == Kind.OPUS for p in self.s.transport.sent))

    async def test_duplicate_plan_does_not_replace_first_and_late_plan_is_ignored(self):
        token = await self.prepare()
        other = Plan((Step('sad', 'blink', 1200),))
        self.s.provider_event('expression.plan', {'identity':token, 'plan':other})
        await self.ack_events()
        self.assertEqual(self.plans()[0].body, BODY)
        self.s.provider_event('expression.plan', {'identity':token, 'plan':other})
        self.s.send_expression_plan(token)
        self.assertEqual(len(self.plans()), 1)

    async def test_close_invalidates_visual_retry_before_async_cleanup(self):
        token = await self.prepare(); await self.ack_events()
        blocked = asyncio.Event()
        original = self.s.interruptions.close
        async def cleanup():
            await blocked.wait()
            await original()
        self.s.interruptions.close = cleanup
        task = asyncio.create_task(self.s.close_remote(token, 0))
        try:
            await test_v3.settle()
            self.assertFalse(task.done())
            self.assertIsNone(self.s.eye_plans.pending)
        finally:
            blocked.set(); await task

    async def test_provider_tool_continuation_reaches_negotiated_wire(self):
        from test_expressions import arguments
        await self.s.xai.handle_event({'type':'response.created', 'response':{'id':'tools'}})
        token = self.s.conversation.response
        await self.s.xai.handle_event({'type':'response.function_call_arguments.done',
            'response_id':'tools', 'call_id':'plan-1', 'name':'set_expression_sequence',
            'arguments':arguments()})
        await self.s.xai.handle_event({'type':'response.done', 'response':{'id':'tools', 'status':'completed'}})
        await self.s.xai.handle_event({'type':'response.created', 'response':{'id':'voice'}})
        self.assertEqual(self.s.conversation.response, token)
        await self.s.xai.handle_event({'type':'response.output_audio.delta', 'response_id':'voice',
            'item_id':'audio-voice', 'delta':base64.b64encode(bytes(640)).decode()})
        await self.ack_events()
        self.assertEqual(len(self.plans()), 1)
        self.assertEqual(decode_plan(self.plans()[0].body), Plan((Step('happy', 'soft_emphasis', 1200),)))
        self.assertEqual(self.plans()[0].identity, token)
