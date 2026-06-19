# Overlay Sentence Audio Plan

## Goal

Add optional sentence-audio mining to screen OCR without changing existing OCR, lookup, screenshot, word-audio, duplicate, crop, or popup behavior when the feature is unavailable or disabled.

The active Anki field map must contain `{sentence-audio}` before Chimahon requests audio permission, starts playback capture, buffers audio, runs Silero VAD, or runs Whisper.

## Fixed decisions

- The `Mine Overlay Audio` setting defaults off.
- Default candidate window: 15 seconds before and 5 seconds after the OCR-button timestamp.
- Keep approximately three minutes of 16 kHz mono PCM16 playback audio in a bounded ring buffer.
- Use Android playback capture through the screen lookup service's existing `MediaProjection`.
- Run Silero VAD first. Run whisper.cpp only when VAD finds a plausible voiced segment.
- Add no sentence audio when Whisper cannot align the OCR sentence to the transcript.
- Missing models, denied permission, silence, capture failure, VAD failure, Whisper failure, timeout, or alignment failure must not prevent Anki card creation.
- Models are downloaded explicitly from Dictionary settings. Do not bundle large model files in the APK.
- Initial model targets are whisper.cpp `base-q5_1` and its supported `silero-v6.2.0` VAD model. Benchmarking may change the Whisper model before the native stage ships.

## PR 1: Contracts, settings, and Anki media plumbing

Status: implemented in the current worktree.

- Add preferences for the feature toggle and before/after offsets.
- Add an Overlay Sentence Audio settings group for the toggle and offsets.
- Add shared sentence-audio request/result types and an injectable provider boundary.
- Add an optional popup callback that is invoked only for a mapped `{sentence-audio}` field.
- Teach `AnkiCardCreator` to store a supplied sentence-audio result and render `{sentence-audio}` as `[sound:...]`.
- Keep all new parameters optional so manga OCR, novel lookup, process-text lookup, dictionary lookup, and regular Anki mining are unchanged.
- Add pure tests for field-map gating and sentence-audio marker rendering/storage preparation where practical.

Exit criteria:

- Feature defaults off.
- No caller is required to supply sentence audio.
- A supplied successful result can populate `{sentence-audio}`.
- No result leaves the field empty and card creation proceeds.

## PR 2: Model manager and explicit downloads

Status: implemented in the current worktree.

- Add a model manager based on the existing local-OCR downloader pattern.
- Add model status, download, retry, and delete controls to the Overlay Sentence Audio settings group.
- Download Whisper and Silero files only after the user presses the settings button.
- Download to temporary files, validate expected size/checksum, then atomically rename into `filesDir/overlay_audio_models`.
- Expose installed/downloading/error status and model deletion/retry controls.
- Never start a download merely because screen lookup or the feature toggle was enabled.

Exit criteria:

- Download, retry, installed-state, and delete paths are visible in settings.
- Interrupted or invalid downloads never replace valid models.
- Model absence is reported but does not affect normal mining.

## PR 3: Playback capture and bounded ring buffer

Status: implemented in the current worktree.

- Add `RECORD_AUDIO` to the manifest.
- Request it only when the feature is enabled and the active profile maps `{sentence-audio}`; denial still starts ordinary screen lookup.
- Add playback `AudioRecord` capture to `ScreenLookupService` using the existing `MediaProjection`.
- Match `USAGE_GAME`, `USAGE_MEDIA`, and `USAGE_UNKNOWN` on API 29+.
- Add a timestamped, bounded PCM ring buffer and record the OCR timestamp from a monotonic clock when capture begins.
- Carry the timestamp through `ScreenLookupOverlayController` to the optional sentence-audio provider.
- Stop and release the recorder with the service and projection callback.
- Add unit tests for wraparound, time-window extraction, missing history, and waiting for the post-capture interval.

Exit criteria:

- Toggle off or marker absent means no permission request, `AudioRecord`, or ring buffer.
- Memory stays bounded and recorder/native resources are released.
- Existing screen capture and overlay behavior remains unchanged.

## PR 4: whisper.cpp and Silero VAD native pipeline

Status: planned.

- Vendor a pinned whisper.cpp revision and record its license/revision.
- Add a small JNI wrapper to load/free the Whisper and Silero contexts and process in-memory 16 kHz PCM.
- Run Silero first with conservative speech-duration and silence thresholds.
- Skip Whisper when VAD finds no plausible dialogue.
- For voiced candidates, run Whisper in Japanese with timestamps and no previous-text conditioning.
- Normalize OCR/transcript text, score contiguous timestamped spans, bias ties toward the OCR timestamp, and require a minimum match score.
- Return an in-memory WAV only for a successful match; do not fall back to the untrimmed candidate.
- Serialize inference, bound it with a timeout, and run all native/audio work off the main thread.

Exit criteria:

- Unvoiced samples never invoke Whisper.
- Failed alignment never produces sentence audio.
- Successful alignment returns a trimmed WAV and releases all temporary/native resources.

## PR 5: Screen-overlay integration and device validation

Status: planned.

- Bind the screen lookup service's timestamped audio provider to the optional `OcrLookupPopup` callback.
- Preserve existing duplicate, open, crop, screenshot, and word-audio behavior.
- Add structured recoverable logging and user-visible model/permission status where useful.
- Run regression checks for OCR overlay, screenshot fields, word audio, duplicate handling, recursive lookup, and regular reader mining.
- Validate playback capture on API 29 and API 34+ with capturable game/media audio, blocked capture, silence, permission denial, and projection revocation.

Exit criteria:

- Successful overlay mining produces an Anki `[sound:...]` sentence-audio field.
- Every failure mode still creates the card without sentence audio.
- Disabled behavior is indistinguishable from the current implementation.

## Verification policy

- CI uses fake PCM, fake VAD, and fake Whisper results; it never invokes live playback capture or downloads models.
- Native tests cover resource lifetime and alignment inputs with deterministic fixtures.
- Device-only tests cover MediaProjection playback capture and performance.
