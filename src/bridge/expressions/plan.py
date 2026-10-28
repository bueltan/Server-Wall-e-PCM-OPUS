"""Bounded, validated plans whose offsets are relative to audio playback start."""
import json
from dataclasses import dataclass

EMOTIONS = ('neutral', 'happy', 'curious', 'calm', 'concerned', 'sad', 'excited')
GESTURES = ('none', 'soft_emphasis', 'emphasis', 'question', 'acknowledge',
            'glance', 'blink', 'doubleblink', 'sway')
MAX_STEPS = 3
MIN_DURATION_MS = 1200
MAX_DURATION_MS = 4000
MAX_ARGUMENT_BYTES = 4096
TOOL_NAME = 'set_expression_sequence'

EXPRESSION_INSTRUCTIONS = '''
Disponés de set_expression_sequence para acompañar visualmente tu respuesta.
Si aporta expresividad, llamala una sola vez antes de hablar, con 1 a 3 pasos.
Para respuestas de una palabra o confirmaciones breves, omitila.
Usá emotion neutral, happy, curious, calm, concerned, sad o excited.
Preferí neutral, happy o curious; sad y excited solo cuando el contexto lo justifica.
Usá gesture none por defecto; question solo si realmente vas a preguntar.
Cada duration_ms debe ser un entero entre 1200 y 4000. Son tiempos visuales
aproximados, no una predicción exacta de la duración de tu audio. No alargues
la respuesta para completar una secuencia ni prometas sincronía con palabras.
No vuelvas a llamar la herramienta al continuar después de recibir su resultado.
Si la herramienta falla, seguí hablando normalmente sin reintentar.
Nunca pronuncies nombres de herramientas, emociones de control ni JSON.
'''.strip()


def tool_schema():
    return {'type': 'function', 'name': TOOL_NAME,
            'description': 'Prepare a short visual expression plan for the upcoming spoken reply; once per reply, before audio.',
            'parameters': {'type': 'object', 'additionalProperties': False,
                'properties': {'sequence': {'type': 'array', 'minItems': 1, 'maxItems': MAX_STEPS,
                    'items': {'type': 'object', 'additionalProperties': False,
                        'properties': {'emotion': {'type': 'string', 'enum': list(EMOTIONS)},
                                       'gesture': {'type': 'string', 'enum': list(GESTURES)},
                                       'duration_ms': {'type': 'integer', 'minimum': MIN_DURATION_MS,
                                                       'maximum': MAX_DURATION_MS}},
                        'required': ['emotion', 'gesture', 'duration_ms']}}},
                'required': ['sequence']}}

@dataclass(frozen=True)
class Step:
    emotion: str
    gesture: str
    duration_ms: int

@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]

    def at(self, position_ms):
        """Return current index/step, or None after the plan. Never replay missed gestures."""
        if position_ms < 0: return None
        offset = 0
        for index, step in enumerate(self.steps):
            if position_ms < offset + step.duration_ms: return index, step
            offset += step.duration_ms
        return None

    def as_dict(self):
        offset = 0
        sequence = []
        for step in self.steps:
            sequence.append(dict(emotion=step.emotion, gesture=step.gesture,
                                 duration_ms=step.duration_ms, offset_ms=offset))
            offset += step.duration_ms
        return {'sequence': sequence, 'total_ms': offset, 'timebase': 'playback_start'}


def parse_plan(arguments):
    if not isinstance(arguments, str) or len(arguments.encode('utf-8')) > MAX_ARGUMENT_BYTES:
        raise ValueError('arguments_too_large_or_not_string')
    try: data = json.loads(arguments)
    except (ValueError, RecursionError): raise ValueError('invalid_json') from None
    if not isinstance(data, dict) or set(data) != {'sequence'}:
        raise ValueError('expected_sequence_object')
    sequence = data['sequence']
    if not isinstance(sequence, list) or not 1 <= len(sequence) <= MAX_STEPS:
        raise ValueError('sequence_must_have_1_to_3_steps')
    steps = []
    for step in sequence:
        if not isinstance(step, dict) or set(step) != {'emotion', 'gesture', 'duration_ms'}:
            raise ValueError('invalid_step_fields')
        if step['emotion'] not in EMOTIONS or step['gesture'] not in GESTURES:
            raise ValueError('unknown_emotion_or_gesture')
        duration = step['duration_ms']
        if type(duration) is not int or not MIN_DURATION_MS <= duration <= MAX_DURATION_MS:
            raise ValueError('duration_ms_must_be_integer_1200_to_4000')
        steps.append(Step(step['emotion'], step['gesture'], duration))
    return Plan(tuple(steps))
