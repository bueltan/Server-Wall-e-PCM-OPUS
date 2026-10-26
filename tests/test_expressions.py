"""Offline expression-tool regressions, including the real v3 audio coordinator."""
import asyncio
import base64
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
import test_v3
from bridge.expressions.plan import parse_plan, tool_schema, TOOL_NAME
from bridge.expressions.timeline import ExpressionTimeline
from bridge.transport.protocols.conversation import Kind, Event


def arguments(**overrides):
    return json.dumps({'sequence': [dict(emotion='happy', gesture='soft_emphasis', duration_ms=1200, **overrides)]})

class PlanTests(unittest.TestCase):
    def test_schema_and_parser_share_limits(self):
        schema=tool_schema()['parameters']['properties']['sequence']
        self.assertEqual(schema['maxItems'],3)
        plan=parse_plan(arguments())
        self.assertEqual(plan.at(0)[1].emotion,'happy')
        self.assertIsNotNone(plan.at(1199));self.assertIsNone(plan.at(1200))
        self.assertIsNone(plan.at(-1))
        self.assertEqual(plan.as_dict()['timebase'],'playback_start')

    def test_invalid_payloads_are_rejected_not_clamped(self):
        valid={'emotion':'happy','gesture':'none','duration_ms':1200}
        bad=[None,'', '[]','null','{}','x'*4097, json.dumps({'sequence':[]}),
             json.dumps({'sequence':[valid]*4}), json.dumps({'sequence':'bad'})]
        for duration in [True,False,1200.0,'1200',0,1199,4001,None]:
            bad.append(json.dumps({'sequence':[dict(valid,duration_ms=duration)]}))
        for field,value in [('emotion',[]),('emotion','evil'),('gesture','nod'),('extra',3)]:
            bad.append(json.dumps({'sequence':[dict(valid,**{field:value})]}))
        for payload in bad:
            with self.subTest(payload=payload),self.assertRaises(ValueError):parse_plan(payload)

    def test_media_preview_skips_stale_steps_and_resets(self):
        plan=parse_plan(json.dumps({'sequence':[
            {'emotion':'happy','gesture':'blink','duration_ms':1200},
            {'emotion':'curious','gesture':'question','duration_ms':1200}]}))
        t=ExpressionTimeline()
        with patch('bridge.expressions.timeline.event') as log:
            t.prepare('identity',plan)
            t.advance('old',0);self.assertEqual(log.call_count,1)
            t.advance('identity',1500);self.assertEqual(log.call_args.kwargs['step'],1)
            t.advance('identity',1600);self.assertEqual(log.call_count,2)
            t.advance('identity',2500);self.assertEqual(log.call_args.kwargs['emotion'],'neutral')
            t.reset('interrupt');t.advance('identity',0);self.assertEqual(log.call_count,4)

class ExpressionIntegration(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=test_v3.SessionTests.asyncSetUp
    asyncTearDown=test_v3.SessionTests.asyncTearDown
    deliver=test_v3.SessionTests.deliver
    ack_events=test_v3.SessionTests.ack_events

    async def begin(self,rid='tools'):
        await self.s.xai.handle_event({'type':'response.created','response':{'id':rid}})
        return self.s.conversation.response

    async def call(self,cid='call-1',args=None,rid='tools',name=TOOL_NAME):
        await self.s.xai.handle_event({'type':'response.function_call_arguments.done',
            'response_id':rid,'call_id':cid,'name':name,'arguments':arguments() if args is None else args})

    async def done(self,rid='tools',status='completed',output=None):
        await self.s.xai.handle_event({'type':'response.done',
            'response':{'id':rid,'status':status,'output':output or []}})

    async def audio(self,rid='voice'):
        await self.s.xai.handle_event({'type':'response.output_audio.delta','response_id':rid,
            'item_id':'audio-'+rid,'delta':base64.b64encode(bytes(640)).decode()})

    def outputs(self):return [p for p in self.provider.calls if p['type']=='conversation.item.create']
    def creates(self):return [p for p in self.provider.calls if p['type']=='response.create']

    async def test_plan_is_bound_to_same_token_and_audio_never_ends_on_tool_only_response(self):
        token=await self.begin();await self.call()
        self.assertFalse(self.outputs());self.assertIsNone(self.s.expressions.plan)
        await self.s.xai.handle_event({'type':'response.output_audio.done','response_id':'tools'})
        self.assertTrue(self.s.output_pcm_queue.empty())
        await self.done();self.assertEqual(len(self.creates()),1)
        self.assertTrue(self.s.output_pcm_queue.empty());self.assertNotIn(token,self.s.xai.output_done)
        await self.begin('voice');self.assertEqual(self.s.conversation.response,token)
        await test_v3.settle()
        starts=[p for p in self.s.transport.sent if p.kind==Kind.EVENT and p.body[0]==Event.RESPONSE_STARTED]
        self.assertEqual(len(starts),1)
        await self.audio();self.assertEqual(self.s.expressions.identity,token)
        self.assertEqual(self.s.expressions.plan.steps[0].emotion,'happy')
        await self.done('voice');await self.ack_events();await test_v3.settle()
        self.assertTrue(any(p.kind==Kind.OPUS for p in self.s.transport.sent))
        self.assertEqual(sum(p.kind==Kind.AUDIO_END for p in self.s.transport.sent),1)
        self.assertIsNone(self.s.expressions.plan)

    async def test_parallel_duplicate_and_terminal_calls_return_once_then_one_continuation(self):
        await self.begin();await self.call();await self.call()
        second={'type':'function_call','call_id':'call-2','name':TOOL_NAME,'arguments':arguments()}
        await self.done(output=[second]);await self.done(output=[second])
        self.assertEqual(len(self.outputs()),2);self.assertEqual(len(self.creates()),1)
        results=[json.loads(p['item']['output']) for p in self.outputs()]
        self.assertTrue(results[0]['ok']);self.assertFalse(results[1]['ok'])

    async def test_invalid_arguments_continue_audio_without_plan(self):
        await self.begin();await self.call(args='[]');await self.done()
        self.assertFalse(json.loads(self.outputs()[0]['item']['output'])['ok'])
        self.assertFalse(self.s.xai.expression_control.plans)
        await self.begin('voice');await self.audio();await self.done('voice');await self.ack_events()
        self.assertTrue(any(p.kind==Kind.OPUS for p in self.s.transport.sent))

    async def test_explicit_call_id_correlation_when_event_omits_response_id(self):
        await self.begin()
        await self.s.xai.handle_event({'type':'response.output_item.added','response_id':'tools',
            'item':{'type':'function_call','call_id':'correlated','id':'function-item','name':TOOL_NAME}})
        await self.s.xai.handle_event({'type':'response.function_call_arguments.done',
            'call_id':'correlated','name':TOOL_NAME,'arguments':arguments()})
        await self.done();self.assertEqual(len(self.outputs()),1)

    async def test_unknown_or_stale_call_cannot_execute_or_continue(self):
        await self.begin();await self.call(rid='unknown');self.assertFalse(self.s.xai.expression_control.calls)
        await self.call();self.s.conversation.speech_started();await self.done()
        self.assertFalse(self.outputs());self.assertFalse(self.creates())
        self.assertFalse(self.s.xai.expression_control.plans)

    async def test_cancelled_terminal_never_continues(self):
        await self.begin();await self.call();await self.done(status='cancelled')
        self.assertFalse(self.outputs());self.assertFalse(self.creates())

    async def test_late_tool_after_audio_is_rejected_without_second_response(self):
        await self.begin();await self.audio('tools');await self.call();await self.done()
        result=json.loads(self.outputs()[0]['item']['output'])
        self.assertEqual(result['error'],'audio_already_started');self.assertFalse(self.creates())
        self.assertFalse(self.s.xai.expression_control.plans)

    async def test_old_continuation_cannot_become_a_new_reply_after_interruption(self):
        await self.begin();await self.call();await self.done()
        self.s.conversation.speech_started()
        with self.assertRaisesRegex(RuntimeError,'stale_tool_continuation'):await self.begin('late')
        self.assertIsNone(self.s.conversation.response)

    async def test_repeated_tool_round_is_bounded(self):
        await self.begin();await self.call();await self.done();await self.begin('voice')
        await self.call(cid='again',rid='voice')
        with self.assertRaisesRegex(RuntimeError,'repeated_expression_tool_round'):await self.done('voice')
        self.assertEqual(len(self.creates()),1)

    async def test_connection_close_clears_all_tool_state(self):
        await self.begin();await self.call();await self.done()
        await self.s.xai.close()
        state=self.s.xai.expression_control
        self.assertFalse(state.plans);self.assertFalse(state.calls);self.assertIsNone(state.continuation)

    async def test_session_update_has_schema_and_disable_switch(self):
        await self.s.xai._send_session_update()
        payload=[p for p in self.provider.calls if p['type']=='session.update'][-1]
        self.assertEqual(payload['session']['tools'][1]['name'],TOOL_NAME)
        self.assertIn('1 a 3 pasos',payload['session']['instructions'])
        self.s.xai.config=replace(self.s.xai.config,expression_tools_enabled=False)
        await self.s.xai._send_session_update()
        payload=[p for p in self.provider.calls if p['type']=='session.update'][-1]
        self.assertEqual(payload['session']['tools'],[{'type':'web_search'}])
