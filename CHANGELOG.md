# Changelog

## 2026-10-12

- Correlate xAI operation errors without terminating the event receiver.
- Preserve VAD conversations on explicit already-cancelled or already-deleted outcomes.
- Log provider error details and add interruption recovery regression tests.

## 2026-10-11

- Extend bounded uplink NACK recovery and expose successful retries, expiry and recovery counters.
- Add delayed recovery, saturation and cross-language regression coverage.

- Apply startup backpressure to queued and reordered PCM; add a gap/backlog regression test.
- Log startup transfer completion and its drop counters.

- Recover early VAD interruptions with a bounded response cancellation fallback.
- Preserve confirmed conversations and repair items from terminal response snapshots.
- Add offline regression tests for early cancellation and subsequent conversation turns.
