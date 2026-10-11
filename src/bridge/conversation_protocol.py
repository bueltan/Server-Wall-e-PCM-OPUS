"""Shared UDP v3 envelope; audio-body codec is independent of session transport.

Both endpoints require this envelope; naked legacy datagrams are rejected.
"""
import struct
from dataclasses import dataclass
from enum import IntEnum

VERSION = 3
HEADER = struct.Struct('!4sBBHQQIIIH')
MAX_BODY = 658

class Kind(IntEnum):
    HELLO=1
    HELLO_ACK=2
    START=3
    START_ACK=4
    COMMIT=5
    COMMIT_ACK=6
    CANCEL=7
    CANCEL_ACK=8
    PCM=9
    OPUS=10
    NACK=11
    MISSING=12
    ERROR=13
    PING=14
    PONG=15
    EVENT=16
    EVENT_ACK=17
    INTERRUPT=18
    INTERRUPT_ACK=19
    PLAYED=20
    PLAYED_ACK=21

@dataclass(frozen=True)
class Identity:
    client: int
    server: int
    turn: int
    response: int

@dataclass(frozen=True)
class Packet:
    kind: Kind
    flags: int
    identity: Identity
    sequence: int
    body: bytes = b''

def encode(packet: Packet) -> bytes:
    kind = Kind(packet.kind)
    body = bytes(packet.body)
    if len(body)>MAX_BODY:
        raise ValueError('Body exceeds v3 limit')
    identity=packet.identity
    return HEADER.pack(b'PIU3',VERSION,kind,packet.flags,identity.client,identity.server,
                       identity.turn,identity.response,packet.sequence,len(body))+body

def decode(data: bytes) -> Packet:
    if len(data)<HEADER.size:
        raise ValueError('Truncated v3 header')
    magic,version,kind,flags,client,server,turn,response,seq,size=HEADER.unpack_from(data)
    if magic!=b'PIU3' or version!=VERSION or size>MAX_BODY or len(data)!=HEADER.size+size:
        raise ValueError('Invalid v3 envelope')
    return Packet(Kind(kind),flags,Identity(client,server,turn,response),seq,data[HEADER.size:])

class Event(IntEnum):
    SPEECH_STARTED=1
    SPEECH_STOPPED=2
    RESPONSE_STARTED=3
    CLOSED=4
    READY=5
