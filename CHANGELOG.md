# Changelog

## 2026-10-30

- Support explicitly authorized personal folders as Codex workspaces for coding or file tasks. Add locally configured excluded_paths that deny both reading and writing, prune excluded trees during symlink validation and avoid reading excluded PlatformIO metadata.
- Keep the bridge control directory and external-link virtual environments excluded when authorizing Documents. Flatten secret denials outside excluded trees to avoid the installed Codex/bwrap nested-denial mount conflict; document startup snapshot behavior for those secret patterns.
- Use scandir metadata for large-workspace validation and serialize overlapping parent/child workspace roots. Extend policy, sandbox profile and queue regression coverage; preserve existing project entries and authentication.
- Pass all 296 local backend tests, syntax/dependency checks and real sandbox create/read/delete probes in Documents, Downloads and Music; confirm an alias into the excluded bridge remains unreadable. Personal authorization paths remain private local configuration.

- Bound task-manager shutdown and use cancellation-safe timeout contexts around Codex queue/event waits on Python 3.11. Add a shutdown regression; pass all 290 local backend tests and all 31 Codex tests on the VPS runtime.
- Verify real VPS MCP discovery, task execution, unittest results, saved-thread continuation, cancellation, automatic reverse-tunnel reconnection during an active 40-second task, persistent results and idempotent retries. Confirm completed history survives a gateway restart.
- Record the remaining remote full-suite timeout in the pre-existing ingress coordination test, reproduced independently on baseline 7256ef2 under Python 3.11. Keep audio implementation unchanged; full local Python 3.12 suite passes.

- Validate all 289 backend tests, Python/Bash syntax, dependency consistency, unit syntax and diff whitespace. Verify real Codex file editing/unittest execution through the local systemd gateway, idempotency and an actual filesystem sandbox check using disposable projects. Enable the local user services and linger; authorize a service-specific existing AppArmor runtime profile on Ubuntu while retaining mandatory runtime NoNewPrivs/private-device checks.

- Add optional Gemini Codex MCP tools on a separate ToolManager: status, authorized projects, asynchronous execute, progress/result, cancellation, recent tasks and saved-context continuation. Keep audio and firmware unchanged.
- Add authenticated loopback task gateway, private Codex App Server stdio adapter, bounded local queue, SQLite state/event history, request idempotency and conservative restart/timeout reconciliation without automatic mutation replay.
- Apply a named restricted permission profile, canonical project/symlink validation, private configuration and bearer files, denied permission escalation, isolated command environment and per-project serialization. Expose task operations instead of raw Codex RPC over the reverse tunnel.
- Add user systemd service templates, configurable OpenSSH reverse tunnel with strict host-key checking and paced restart, environment/project examples, bounded command evidence and durable backend event polling without audio delivery.
- Document installation, integration, PlatformIO prerequisites, isolation limits, recovery, diagnostics and explicit real-model smoke validation in CODEX_MCP.md and README. Add mocked lifecycle/security tests plus real local MCP stdio/WebSocket coverage.

## 2026-10-29

- Revalidate the captured Gemini/xAI WebSocket after acquiring the send lock so queued audio cannot move to a replacement connection. Invalidate Gemini's socket at close entry before asynchronous MCP cleanup.
- Discard Gemini audio/tool events whose connection changed while waiting for the response/history barrier; isolate Gemini/xAI receiver processing and readiness cleanup to their captured socket. Extend downlink stream/pacer checks across the output gate, final PCM flush and END pacing/repetition so stale END cannot finish a reset stream.
- Add deterministic offline regressions that reproduce these races against the original code, cover cancellation and verify subsequent fresh audio/END delivery. Document the limited scope and in-flight-delivery limitation; no live provider calls or deployment performed.
- Pass all 259 backend tests, including eight new audio lifecycle tests, Python syntax compilation and diff whitespace checks. Live provider/hardware validation remains pending deployment.

- Add optional Device MCP over stdio with an independent ToolManager and a persistent authenticated loopback WebSocket gateway into the session-owned UDP transport. Keep device controls independent of slow Google tool calls and bind local connections to their opening conversation.
- Add volume query/set/relative adjustment and shared mute/restore, optional explicit persistence, CAP_VOLUME_CONTROL=0x0040 and VOLUME/VOLUME_ACK=30/31. Confirm only CORE acknowledgements, deduplicate UDP retries and preserve unknown outcomes without replaying WebSocket mutations.
- Interpret Spanish lower/higher-volume requests as −20/+20 percentage points when unspecified, including «hablá más despacio» as lower volume; an explicit amount replaces that step and a repeated reference in one request must produce one adjustment. Playback speed is unchanged.
- Add offline real stdio/WebSocket, validation, session binding, authentication, failure and Python/C++ wire coverage. Document activation, server-first upgrade order, temporary versus saved levels, and pending live voice/hardware validation; no deployment or firmware upload performed.
- Catch unexpected device-tool, WebSocket client/gateway and UDP transport exceptions with fixed safe codes and uncertain-write outcomes. Add correlated device lifecycle/retry/result logs, sanitize exception metadata and keep tokens, arbitrary request IDs and raw exception text out of diagnostics. Cover injected failures and connection recovery offline.
- Pass all 251 backend tests, Python syntax compilation and dependency consistency checks; the paired CORE host suite, Python/C++ interoperability and cyd-uart0 build pass. Live deployment, firmware upload and spoken-command validation remain pending.

- Deliver Calendar reminders through the existing Gemini connection when CORE explicitly admits a preserved standby turn. Negotiate CAP_STANDBY_NOTIFICATIONS=0x0020 and NOTICE_ACK=29; keep existing idle START/READY delivery for older CORE firmware.
- Revalidate session/utterance, provider readiness, response/tool/history/input activity and reminder eligibility after CORE admission. Preserve durable claim-before-send deduplication, defer busy sessions and keep the conversation open after a lost admission ACK.
- Add offline coverage for standby reuse, manual activation/busy rejection, stale/malformed ACKs, state changes, expired events, lost ACKs, optional negotiation and uncertain-send deduplication; document server-first deployment and pending live reproduction.
- Pass all 234 backend tests and Python syntax checks; the paired CORE host suite, Python/C++ notification interoperability and cyd-uart0 build passed. No deployment or real Calendar writes were performed for validation.

- Preserve fixed validation/auth/API diagnostic codes across all Gmail, Contacts and Calendar MCP tools instead of discarding domain exceptions as mcp_tool_error. Add safe HTTP metadata and distinguish rejected Gmail sends from uncertain writes without enabling retries.
- Validate discovered MCP input schemas before dispatch; report bounded field/rule diagnostics without argument values. Mark domain failures and unknown write outcomes ok=False while retaining data.
- Correlate tool lifecycle logs by call ID and voice-session identity; record queue/execution timing, startup phases, rejection, cancellation and transport/timeout failures. Keep raw exceptions, remote error text, OAuth paths and private payloads out of these diagnostics.
- Document the diagnostic contract and add offline regression coverage with real local MCP stdio exchanges. Historical logs cannot recover discarded error details; deployment and live Calendar reproduction remain pending.
- Pass all 226 backend tests, Python syntax compilation, dependency consistency and diff whitespace checks; no real Google messages, contacts or events were created by validation.

- Enable Gemini sliding-window context compression for conversations retained while CORE microphone input is in standby; preserve existing error/GoAway closure handling without claiming transparent session resumption.
- Pass all 216 backend tests, including 30 simulated minutes of valid CORE heartbeats without PCM, no manufactured provider input, and same-turn audio resumption without reopening the provider. Document CORE's 20-second microphone standby, 15-minute idle closure and provider/Calendar notification limitations. Deployment and live validation remain pending.

- Add a single desktop OAuth helper for the eight agreed Gmail, Contacts, Calendar, Tasks and drive.file scopes, with private atomic storage in a new shared token file.
- Reject incomplete consent or missing refresh tokens before replacing the shared token; preserve existing service tokens and stored grants during refresh.
- Add offline scope checking and document API activation, PC consent, atomic VPS token transfer and shared Gmail/Calendar service configuration in GOOGLE_AUTH.md, README and integration guides. Tasks/Drive/Docs/Sheets tools remain pending.
- Pass all 215 backend tests, including shared-token compatibility, incomplete consent, absent refresh tokens and sanitized errors. No live authorization, deployment or Google writes were performed for this preparation.

- Add gmail_create_contact for explicit user requests with a name plus email and/or phone; serialize People API mutations, disable automatic retries and protect cancelled tool IDs from duplicate creation.
- Request the contacts write scope on Gmail OAuth authorization while preserving Gmail-only email access and contacts.readonly lookup grants. No contact update/delete tools or email sending are part of contact creation.
- Pass all 209 backend tests, including contact creation payloads, input validation, uncertain outcomes, cancellation deduplication and old read-only token compatibility. Document new consent and VPS token replacement; no real contacts were created.

- Add read-only Google Contacts lookup to Gmail MCP through People API: gmail_search_contacts and gmail_get_contact return saved-contact names, email labels/addresses and phone numbers.
- Extend Gmail OAuth authorization with contacts.readonly while preserving existing Gmail-only tokens for email operations; require new consent only for contact access and keep Calendar tokens independent.
- Warm the People search cache before prefix queries, bound search results and expose fixed auth/API diagnostic codes. Instruct Gemini to resolve named recipients through contacts and ask about ambiguous people or addresses before sending.
- Pass all 203 backend tests, including saved-contact lookup, cache warmup, read-only MCP annotations and compatibility with Gmail-only tokens. Document People API activation, local OAuth renewal and VPS token transfer; real-contact validation remains pending.

- Pass all 197 backend tests, CORE host tests and Python/C++ notification wire interoperability; compile cyd-uart0 successfully without flashing hardware.
- Integrate an optional PIUU Calendar MCP server alongside Gmail: calendar/event discovery, FreeBusy availability and timed/all-day event creation with explicit time zones, pagination and uncertain-write reconciliation IDs.
- Share OAuth helpers while keeping independent Gmail/Calendar tokens and scopes; add fixed diagnostic codes and tool lifecycle logs without argument or credential logging.
- Add opt-in Calendar reminders ten minutes before timed events, with 120-second polling, local scheduling, quota-error backoff, stale-cache suppression and durable attempt deduplication.
- Negotiate optional UDP v3 CAP_NOTIFICATIONS=0x0010 and NOTICE=28; CORE owns the correlated START, stays muted until playback, defers notices while busy and bounds the initial response wait. Voice Calendar tools remain compatible with earlier CORE firmware.
- Document Calendar OAuth, the VPS service's /home/denis/apps/.venv interpreter, deployment and future HTTPS webhooks. User confirmed Gmail working after enabling Gmail API; Calendar live authorization/deployment and hardware notice validation remain pending.

- Copy the locally authorized Gmail token to the VPS over SCP; verify the nonempty token, denis ownership, 600 credential/token permissions and 700 .secrets directory permissions. Gmail was later confirmed working by the user after enabling its API.

- Document the confirmed VPS path /home/denis/apps/Server-Wall-e-PCM-OPUS, local OAuth authorization, credential/token transfer over SSH and systemd environment settings.

- Add a PIUU-owned Gmail MCP server over stdio and a provider-independent Tool Manager with tool discovery, prefixed routing, bounded queues and call timeouts.
- Integrate Gmail search, message reading and plain-text email sending into Gemini Live using explicit NON_BLOCKING declarations and WHEN_IDLE results; keep expression preparation BLOCKING.
- Preserve pending calls across audio turns, handle tool cancellation and deduplicate call IDs; retain cancelled-send markers and disable automatic send retries to avoid duplicate messages after uncertain outcomes.
- Add Google OAuth desktop authorization with gmail.readonly and gmail.send scopes, automatic token refresh, private token files and detection of old read-only tokens requiring new consent.
- Add optional src/requirements-mcp.txt dependencies; verify the existing Python 3.12 .venv, package consistency and project imports.
- Document environment setup, Gmail activation, OAuth reauthorization and manual send testing in README.md and GMAIL_MCP.md. Home MCP remains pending; Calendar is documented separately.
- Pass all 173 backend tests, including real local MCP stdio exchanges, MIME encoding, send validation, OAuth scope migration and cancellation regressions. These offline tests did not send real email.

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
