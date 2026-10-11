"""Provider-independent identities/state. No sockets, audio buffers or UI."""
from enum import Enum
from .conversation_protocol import Identity

class Phase(Enum):
    CLOSED='closed'
    LISTENING='listening'
    THINKING='thinking'
    RESPONDING='responding'

class Conversation:
    def __init__(self):
        self.identity = None
        self.response = None
        self.utterance = 0
        self.next_response = 0
        self.phase = Phase.CLOSED

    def open(self, identity):
        self.identity = identity
        self.response = None
        self.utterance = self.next_response = 0
        self.phase = Phase.LISTENING

    def speech_started(self):
        previous = self.response
        self.response = None
        self.utterance += 1
        self.phase = Phase.LISTENING
        return previous

    def speech_stopped(self):
        self.phase = Phase.THINKING

    def response_started(self):
        if self.identity is None: return None
        self.next_response += 1
        if self.next_response > 0xffffffff: raise RuntimeError('Response identity exhausted')
        i = self.identity
        self.response = Identity(i.client, i.server, i.turn, self.next_response)
        self.phase = Phase.RESPONDING
        return self.response

    def played(self, identity):
        if self.response == identity:
            self.response = None
            self.phase = Phase.LISTENING

    def close(self):
        self.identity = self.response = None
        self.phase = Phase.CLOSED
