# Changelog

## 2026-10-27

- Revert incompatible custom-voice changes and restore the Live handshake and 1–3 expression steps.
- Add PIUU baby-bird delivery instructions over the prebuilt Puck voice.

- Default to Gemini when PIUU_VOICE_PROVIDER is unset.

- Add selectable Gemini 3.8 Live with setup confirmation and streaming 24-to-16 kHz audio.
- Bridge expression tools and native voice interruptions into the existing UDP v3 pipeline.
- Add offline DSP/provider/integration tests and document manual interruption and session limits.

## 2026-10-26

- Add capability-negotiated binary eye plans with atomic gestures and playback offsets.
- Bound visual ACK retries without gating audio; clear plans on interruption and close.
- Document the wire contract and pending CORE scheduler; test negotiation and delivery faults.

- Add validated PIUU expression sequences through xAI function calling.
- Preserve response identity across tool continuation and reject stale/duplicate calls.
- Preview expressions on outgoing audio positions independently of device delivery.
- Add offline expression lifecycle and audio regression tests.

## 2026-10-12

- Group backend modules by transport, conversation, audio, providers, diagnostics and legacy responsibilities.
- Update imports and module documentation while preserving the main entrypoint and runtime behavior.

- Decouple UDP recovery from provider history repair with bounded five-second PCM retention.
- Pace provider catch-up, clear audio on close, and report retention/drop/age metrics.
- Preserve startup backpressure and add offline retention/lifecycle regression tests.

- Add bounded Opus resend caching, per-response downlink NACK and recovery metrics.
- Retry audio END until receipt ACK; fail explicitly on unconfirmed termination.
- Advertise the downlink capability and invalidate recovery on cancellation/reset.

- Correlate xAI operation errors without terminating the event receiver.
- Preserve VAD conversations on explicit already-cancelled or already-deleted outcomes.
- Log provider error details and add interruption recovery regression tests.
git 
## 2026-10-11

- Extend bounded uplink NACK recovery and expose successful retries, expiry and recovery counters.
- Add delayed recovery, saturation and cross-language regression coverage.

- Apply startup backpressure to queued and reordered PCM; add a gap/backlog regression test.
- Log startup transfer completion and its drop counters.

- Recover early VAD interruptions with a bounded response cancellation fallback.
- Preserve confirmed conversations and repair items from terminal response snapshots.
- Add offline regression tests for early cancellation and subsequent conversation turns.
