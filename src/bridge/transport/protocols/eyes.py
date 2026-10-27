"""PIUU eye plan v1: atomic steps on CORE's response playback timeline."""
import struct
from bridge.expressions.plan import Plan, Step, MAX_STEPS, MIN_DURATION_MS, MAX_DURATION_MS

# Wire IDs are stable and match PIUU_EYES enums, not the tool-schema list order.
EMOTIONS = ('neutral', 'happy', 'curious', 'calm', 'concerned', 'sad', 'excited')
GESTURES = ('none', 'emphasis', 'question', 'acknowledge', 'glance', 'blink',
            'doubleblink', 'soft_emphasis', 'sway')
VERSION = 1
HEADER = struct.Struct('!BBH')  # version, count, reserved=0
STEP = struct.Struct('!BBHI')  # emotion, gesture, duration_ms, offset_ms
MAX_PLAN_BYTES = HEADER.size + MAX_STEPS * STEP.size


def encode_plan(plan: Plan) -> bytes:
    if not isinstance(plan, Plan) or not 1 <= len(plan.steps) <= MAX_STEPS:
        raise ValueError('invalid_plan_size')
    output = bytearray(HEADER.pack(VERSION, len(plan.steps), 0))
    offset = 0
    for step in plan.steps:
        if (not isinstance(step, Step) or step.emotion not in EMOTIONS or
                step.gesture not in GESTURES or type(step.duration_ms) is not int or
                not MIN_DURATION_MS <= step.duration_ms <= MAX_DURATION_MS):
            raise ValueError('invalid_plan_step')
        output.extend(STEP.pack(EMOTIONS.index(step.emotion), GESTURES.index(step.gesture),
                                step.duration_ms, offset))
        offset += step.duration_ms
    return bytes(output)


def decode_plan(body: bytes) -> Plan:
    if not HEADER.size <= len(body) <= MAX_PLAN_BYTES:
        raise ValueError('invalid_plan_size')
    version, count, reserved = HEADER.unpack_from(body)
    if version != VERSION or reserved or not 1 <= count <= MAX_STEPS or len(body) != HEADER.size + count * STEP.size:
        raise ValueError('invalid_plan_header')
    steps, expected_offset = [], 0
    for index in range(count):
        emotion, gesture, duration, offset = STEP.unpack_from(body, HEADER.size + index * STEP.size)
        if (emotion >= len(EMOTIONS) or gesture >= len(GESTURES) or offset != expected_offset or
                not MIN_DURATION_MS <= duration <= MAX_DURATION_MS):
            raise ValueError('invalid_plan_step')
        steps.append(Step(EMOTIONS[emotion], GESTURES[gesture], duration))
        expected_offset += duration
    return Plan(tuple(steps))
