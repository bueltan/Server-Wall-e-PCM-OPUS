"""Backend-only media-clock preview. CORE playback scheduling is a future adapter."""
from bridge.diagnostics.logging import event

class ExpressionTimeline:
    def __init__(self):
        self.identity = None
        self.plan = None
        self.index = None
        self.started = False

    def prepare(self, identity, plan):
        self.reset('replaced')
        self.identity, self.plan = identity, plan
        event('EXPRESSION_PLAN', identity, **plan.as_dict())

    def advance(self, identity, position_ms):
        if identity != self.identity or self.plan is None: return
        current = self.plan.at(position_ms)
        index = current[0] if current else -1
        if self.started and index == self.index: return
        self.started = True; self.index = index
        if current:
            _, step = current
            event('EXPRESSION_PREVIEW', identity, clock='opus_send', position_ms=position_ms,
                  step=index, emotion=step.emotion, gesture=step.gesture, duration_ms=step.duration_ms)
        else:
            event('EXPRESSION_PREVIEW', identity, clock='opus_send', position_ms=position_ms,
                  step=-1, emotion='neutral', gesture='none')

    def reset(self, reason):
        if self.plan is not None:
            event('EXPRESSION_RESET', self.identity, reason=reason)
        self.identity = self.plan = self.index = None
        self.started = False
