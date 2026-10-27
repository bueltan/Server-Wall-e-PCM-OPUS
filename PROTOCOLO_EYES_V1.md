# PIUU eye plans over UDP v3 — extension v1

Status: backend and PIUU_CORE implemented and offline-tested, including a real
Python/C++ plan/ACK round trip. CORE now negotiates support and schedules plans
against estimated consumed I2S samples. Physical timing/heap/audio validation is
pending on CYD. No firmware upload, server restart or deployment was performed.

## Negotiation and compatibility

Use the existing PIU3/version-3 envelope and the CORE endpoint learned from HELLO.
Do not open a second socket or assume the CORE source port is always 12000.

- `HELLO.flags & 0x0004`: CORE offers eye-plan v1 support.
- `HELLO_ACK.flags & 0x0004`: backend accepts the offer when
  `BridgeConfig.expression_udp_enabled=True` (default).
- Existing `HELLO_ACK` bit 0 remains readiness; bit 1 remains audio recovery.
- Legacy `HELLO.flags=0` receives the existing flags 2/3. No visual plans are sent.
- Only HELLO permits the new request bit. Other client messages still require
  flags=0. Unknown HELLO bits are rejected.
- The offer is immutable for a client session nonce. Repeated HELLO must repeat
  the offer. A new nonce negotiates again; foreign endpoints and retired nonces
  cannot enable the extension. Both client and server must retain the result.
- Setting `expression_udp_enabled=False` disables only transport. The existing
  `expression_tools_enabled` separately controls the model tool and preview.

This backend remains compatible with current CORE audio. A future CORE supporting
this offer needs a compatible backend; an older backend rejects HELLO flags=4.
If supporting older backends is required, the client must explicitly implement
an audio-only handshake fallback, rather than assuming the extension succeeded.

## Packet kinds

| Kind | Direction | Body | Meaning |
| --- | --- | --- | --- |
| 25 `EYES` | server → CORE | Existing ASCII command | Reserved for immediate commands; this sender does not emit it |
| 26 `EYES_PLAN` | server → CORE | Binary plan, 12/20/28 bytes | One immutable plan for the identified response |
| 27 `EYES_PLAN_ACK` | CORE → server | Empty | Valid plan accepted/handled; not proof of rendering or audio completion |

All three use flags=0 and the existing 38-byte envelope:

```python
struct.Struct('!4sBBHQQIIIH')
# magic='PIU3', version=3, kind, flags,
# client_session:u64, server_session:u64, turn:u32, response:u32,
# sequence:u32, body_length:u16
```

The full identity must match the current nonzero response; neither a session-only
plan nor a guessed/newer/retired response is admitted. ACK echoes the exact full
identity and plan sequence. The server also validates the source endpoint.

There is **one plan per response**, `sequence=1`. This sequence space is separate
from audio, reliable events and immediate kind-25 commands. Retries contain the
identical plan, identity and sequence. A new response can use sequence=1 again.
No replacement plan is sent halfway through a response.

## Binary plan body

All multi-byte integers are unsigned, network byte order (big endian).

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u8 | `format_version=1` |
| 1 | u8 | `step_count`, 1..3 |
| 2 | u16 | Reserved, must be zero |
| 4 + 8×i | u8 | Emotion ID |
| 5 + 8×i | u8 | Gesture ID |
| 6 + 8×i | u16 | `duration_ms`, 1200..4000 |
| 8 + 8×i | u32 | `offset_ms` from the first played audio sample |

The first offset is zero. Later offsets equal the sum of previous durations;
gaps/overlaps, unknown IDs, invalid durations, nonzero reserved bits, unsupported
versions, extra/truncated bytes and counts outside 1..3 are rejected atomically.
Maximum timeline length: 12000 ms. Maximum datagram length: **66 bytes** including
the envelope. The model's JSON stays on the server.

| Emotion ID | Name | Gesture ID | Name |
| --- | --- | --- | --- |
| 0 | neutral | 0 | none |
| 1 | happy | 1 | emphasis |
| 2 | curious | 2 | question |
| 3 | calm | 3 | acknowledge |
| 4 | concerned | 4 | glance |
| 5 | sad | 5 | blink |
| 6 | excited | 6 | doubleblink |
| — | — | 7 | soft_emphasis |
| — | — | 8 | sway |

These IDs match PIUU_EYES's enums. In particular `soft_emphasis=7`: the order of
the model's gesture schema is different and must never define wire IDs implicitly.

Example: happy + soft_emphasis for 2400 ms, then curious + question for 1800 ms:

```text
01 02 00 00
01 07 09 60 00 00 00 00
02 02 07 08 00 00 09 60
```

With identity `(1,2,3,4)` and sequence 1, the complete reference packet is:

```text
50 49 55 33 03 1a 00 00
00 00 00 00 00 00 00 01
00 00 00 00 00 00 00 02
00 00 00 03 00 00 00 04
00 00 00 01 00 14
01 02 00 00 01 07 09 60 00 00 00 00 02 02 07 08 00 00 09 60
```

The codec and tests are `bridge/transport/protocols/eyes.py` and
`tests/test_eye_transport.py`. The vector is checked byte for byte.

## Sender lifecycle

1. The model uses the existing `set_expression_sequence`. Server validation
   accepts at most one bounded plan for the response; the continuation retains
   its response identity. A duplicate/late plan cannot replace the first plan.
2. Wait for the existing reliable `RESPONSE_STARTED` ACK. This is the ordinary
   audio admission gate, not an additional eye gate.
3. Immediately before the first actual Opus transmission, send the complete plan
   if both peers negotiated support. No audio means no eye-plan transmission.
4. Send audio normally without waiting for `EYES_PLAN_ACK`. Optional visual
   delivery never uses the critical `ReliableEvents` failure path.
5. If no matching ACK arrives, retry the identical packet at most twice. Minimum
   separation is 100 ms. The retry deadline is 500 ms from the first attempt.
   The existing 20 ms recovery loop drives this; no new task or per-step sleeps.
   A delayed recovery tick sends at most one packet, never a catch-up burst.
6. At the deadline, log unconfirmed delivery and discard the pending retry;
   keep the response's once-only marker. Do not abort audio or send a fallback
   stream of kind-25 commands. Local socket errors count as attempts.
7. VAD/manual interruption, cancel, output reset, playback completion, peer
   replacement and connection loss clear retries. Invalidation precedes async
   cleanup. A stale ACK cannot complete a newer response's plan.

Opus END means the server finished sending, not that the DAC finished playing.
It does **not** send a neutral command or clear an outstanding visual retry.
Existing PLAYED/cancellation events terminate the response on both endpoints.
ACK means valid receipt/handling only; logs never assert that the eyes moved.

## CORE receiver behavior

PIUU_CORE implements these receiver requirements; the backend sends the plan
and does not drive the rendering clock:

- Validate the entire body and identity before copying a fixed three-step plan.
  Keep plan storage independent of audio and reliable control queues. Never
  render or wait for the display from the UDP worker.
- ACK a valid admitted plan and exact duplicate, including when the original ACK
  was lost. A duplicate must never restart timing, transition or gesture. Reject
  conflicting content with the same identity/sequence. Retain the dedup marker
  through the response, even after the last step. Invalid/stale plans get no ACK.
- Store early plans until playback starts. Anchor `offset_ms=0` to the first
  played sample, not UDP arrival, response creation or the first Opus enqueue.
  Use consumed audio progress, including consumed concealment/silence; pause
  progress during a real rebuffer pause. Do not run a wall-clock task for steps.
- At a boundary, apply emotion and gesture together. Duration covers the full
  step interval **including its transition**. Do not add the legacy emotion
  transition duration to every step or accumulate timing drift.
- If the first valid packet arrives late, select the current nonexpired step
  and apply its remaining emotion interval; skip historical one-shot gestures.
  Never play missed gestures in a burst. If the whole plan elapsed, retain the
  dedup marker and ACK without replaying it.
- Reserve UDP visual priority while the admitted plan is pending/active. Drop
  conflicting EARS emotions/gestures during that interval; continue accepting
  EARS configuration changes. On final-step expiry or response cancellation/end,
  release the plan's priority and return to neutral. Resume idle blink/glance;
  speaking state still follows actual playback. Newer immediate UDP visual
  commands may explicitly cancel a plan; receipt of its retries must not restore it.
- Explicit plan gestures replace lower-priority automatic gestures at a boundary.
  Do not allow an automatic speaking gesture to discard the planned one as busy.
- Clear the pending/active plan on response invalidation even if no packet arrives
  to say 'neutral'. Audio ending early cancels the remaining visual steps.

The payload needs at most 28 bytes plus fixed state/identity/ACK metadata. It
requires no bitmap frames, audio copies, LVGL or additional rendering task.

## Diagnostics and validation

`EYES_PLAN_SENT`, `EYES_PLAN_ACK`, `EYES_PLAN_UNCONFIRMED`, `EYES_PLAN_SEND_ERROR`
and `EYES_PLAN_REJECTED` describe transport events. `EYES_PLAN_STATS` contains
cumulative plans/attempts/acked/unconfirmed/send_errors for the session object.
`EXPRESSION_PREVIEW clock=opus_send` remains a diagnostic preview only.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests cover exact bytes, malformed packets, legacy/disabled negotiation, bounded
loss recovery, wrong/stale ACKs, tool-to-wire continuation, response admission,
nonblocking audio, short/no-audio replies, interruptions and async close cleanup.
The CORE host suite covers playback scheduling and visual priority; its visual
clock estimates progress from accepted I2S samples and caps it at queued media.
Exact DAC timing still requires hardware tests and is not inferred from host tests.
