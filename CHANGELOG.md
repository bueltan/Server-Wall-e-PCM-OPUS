# Changelog

## 2026-10-28

- Bind CORE by its declared WiFi station MAC (`CAP_DEVICE_MAC=0x0008`, six-byte HELLO body) and confirm the extension in HELLO_ACK.
- Admit the same device with a fresh session from a changed IP/UDP port after closing the previous conversation and provider; require a new START without retaining history.
- Rotate the server nonce, clear pending event/downlink state and reject queued controls during peer reset; keep the last 32 retired client nonces from restoring old sessions.
- Reject a different or missing MAC once a device is bound; preserve original-address admission for legacy HELLO clients. The binding lasts for this process and currently trusts the declared MAC without cryptographic authentication.
- Add CORE_ADMITTED/CORE_REBOUND lifecycle logs and offline regressions for IP/port changes, malformed MACs, delayed old packets, queued controls and provider-close ordering.
- Pass all 161 backend tests plus CORE/backend wire interoperability checks; document server-first deployment and pending hardware validation. No backend deployment or CORE flashing was performed in this chat.

- Configure neutral Latin American Spanish (international dubbing style) delivery prompt and system instructions.

## 2026-10-27

- Switch the default Gemini voice to Leda while retaining the PIUU delivery prompt.

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
