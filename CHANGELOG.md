# Changelog

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
