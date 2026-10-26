"""Bounded custom-tool lifecycle for the continuous VAD adapter.

Collect calls until response.done, then return every output and request exactly
one continuation. No eye execution in the provider receive loop.
"""
import json
from bridge.expressions.plan import TOOL_NAME, parse_plan
from bridge.diagnostics.logging import event

class ExpressionTools:
    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = {}
        self.owners = {}
        self.completed = {}
        self.used = set()
        self.plans = {}
        self.continuation = None

    def reset(self):
        self.calls.clear(); self.owners.clear(); self.completed.clear(); self.used.clear()
        self.plans.clear(); self.continuation = None

    def discard(self, token):
        self.plans.pop(token, None)
        for rid in list(self.calls):
            if self.adapter.responses.get(rid) == token: self.calls.pop(rid, None)
        # Keep a stale continuation token until its response.created arrives.
        # It must never be mistaken for a fresh user response.

    def observe_item(self, message):
        item = message.get('item', {})
        rid, cid = message.get('response_id'), item.get('call_id')
        if (item.get('type') == 'function_call' and isinstance(cid, str)
                and len(cid) <= 256 and rid in self.adapter.responses):
            self.owners[cid] = rid
            while len(self.owners) > 32: self.owners.pop(next(iter(self.owners)))

    def collect(self, message):
        a = self.adapter
        cid = message.get('call_id')
        if not isinstance(cid, str): return
        rid = message.get('response_id') or self.owners.get(cid)
        if not isinstance(rid, str) or not isinstance(cid, str) or not cid or len(cid)>256:
            return
        token = a.responses.get(rid)
        if token is None or not a.valid(token) or rid in self.completed: return
        if rid not in self.calls:
            if len(self.calls) >= 4: raise RuntimeError('too_many_tool_responses')
            self.calls[rid] = {}
        calls = self.calls[rid]
        if cid in calls: return
        if len(calls) >= 8: raise RuntimeError('too_many_tool_calls')
        # Bound stored payloads even before validation; never retain oversized args.
        args = message.get('arguments')
        if not isinstance(args, str) or len(args.encode('utf-8'))>4096: args = None
        calls[cid] = (message.get('name'), args)

    def terminal_calls(self, message):
        response = message.get('response', {})
        rid = response.get('id', message.get('response_id'))
        for item in response.get('output', []):
            if item.get('type') == 'function_call':
                self.collect(dict(type='response.function_call_arguments.done', response_id=rid,
                                  call_id=item.get('call_id'), name=item.get('name'),
                                  arguments=item.get('arguments')))
        return rid, self.calls.get(rid, {})

    async def finish(self, rid, token, status, had_audio):
        a = self.adapter
        calls = self.calls.pop(rid, {})
        if not calls or rid in self.completed: return
        self.completed[rid] = True
        while len(self.completed)>64: self.completed.pop(next(iter(self.completed)))
        if token is None or not a.valid(token) or status not in (None, 'completed'):
            self.discard(token); return
        repeat = token in self.used
        self.used.add(token)
        if len(self.used)>64: self.used = {token}
        accepted = token in self.plans
        for cid, (name, args) in calls.items():
            result = {'ok': False}
            if name != TOOL_NAME: result['error'] = 'unsupported_tool'
            elif had_audio: result['error'] = 'audio_already_started'
            elif repeat or accepted: result['error'] = 'expression_already_requested'
            else:
                try: plan = parse_plan(args)
                except ValueError as exc: result['error'] = str(exc)
                else:
                    self.plans[token] = plan; accepted = True
                    result = {'ok': True, 'status': 'prepared', 'steps': len(plan.steps)}
                    event('EXPRESSION_PREPARED', token, steps=len(plan.steps))
            if not result['ok']: event('EXPRESSION_REJECTED', token, reason=result['error'])
            if not a.valid(token): self.discard(token); return
            await a._safe_send({'type': 'conversation.item.create', 'item': {
                'type': 'function_call_output', 'call_id': cid, 'output': json.dumps(result)}})
        if not a.valid(token): self.discard(token); return
        if had_audio or not a.history_ready.is_set(): return
        if repeat:
            # Model requested another tool-only round despite the prompt. Bound loops rather than resubmitting indefinitely.
            raise RuntimeError('repeated_expression_tool_round')
        self.continuation = token
        a.generation_done.clear()
        a.done[token].clear()
        await a._safe_send({'type': 'response.create'})

    def take_plan(self, token):
        return self.plans.pop(token, None)
