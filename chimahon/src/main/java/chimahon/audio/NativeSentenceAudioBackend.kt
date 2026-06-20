package chimahon.audio

import android.util.Log
import java.io.File

internal data class NativeTranscriptSegment(
    val text: String,
    val startMillis: Long,
    val endMillis: Long,
)

data class SileroVadParameters(
    val threshold: Float = 0.60f,
    val minSpeechDurationMillis: Int = 300,
    val minSilenceDurationMillis: Int = 700,
    val maxSpeechDurationSeconds: Float = 20f,
    val speechPaddingMillis: Int = 100,
    val samplesOverlapSeconds: Float = 0.10f,
) {
    init {
        require(threshold in 0f..1f) { "Silero threshold must be between zero and one" }
        require(minSpeechDurationMillis >= 0) { "Silero minimum speech duration must not be negative" }
        require(minSilenceDurationMillis >= 0) { "Silero minimum silence duration must not be negative" }
        require(maxSpeechDurationSeconds > 0f) { "Silero maximum speech duration must be positive" }
        require(speechPaddingMillis >= 0) { "Silero speech padding must not be negative" }
        require(samplesOverlapSeconds >= 0f) { "Silero sample overlap must not be negative" }
    }
}

class NativeSentenceAudioBackend(
    whisperModel: File?,
    vadModel: File,
    threadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
    vadParameters: SileroVadParameters = SileroVadParameters(),
) : SentenceAudioInferenceBackend {
    private var nativeHandle: Long

    init {
        require(whisperModel == null || whisperModel.isFile) { "Whisper model is missing" }
        require(vadModel.isFile) { "Silero VAD model is missing" }
        nativeHandle = nativeCreate(
            whisperModel?.absolutePath,
            vadModel.absolutePath,
            threadCount.coerceAtLeast(1),
            vadParameters.threshold,
            vadParameters.minSpeechDurationMillis,
            vadParameters.minSilenceDurationMillis,
            vadParameters.maxSpeechDurationSeconds,
            vadParameters.speechPaddingMillis,
            vadParameters.samplesOverlapSeconds,
        )
        check(nativeHandle != 0L) { "Unable to load sentence-audio models" }
        runCatching {
            Log.d(
                TAG,
                "Silero parameters: threshold=${vadParameters.threshold}, " +
                    "minSpeechMs=${vadParameters.minSpeechDurationMillis}, " +
                    "minSilenceMs=${vadParameters.minSilenceDurationMillis}, " +
                    "maxSpeechSeconds=${vadParameters.maxSpeechDurationSeconds}, " +
                    "paddingMs=${vadParameters.speechPaddingMillis}, " +
                    "overlapSeconds=${vadParameters.samplesOverlapSeconds}",
            )
        }
    }

    @Synchronized
    override fun detectSpeech(pcm16: ShortArray): List<SpeechSegment> {
        val handle = requireOpenHandle()
        val boundaries = nativeDetectSpeech(handle, pcm16)
        if (boundaries.size % 2 != 0) return emptyList()
        return boundaries.asList().chunked(2).map { (start, end) ->
            SpeechSegment(start.toLong(), end.toLong())
        }
    }

    @Synchronized
    override fun transcribe(pcm16: ShortArray, timeoutMillis: Long): List<TranscriptSegment> {
        val handle = requireOpenHandle()
        return nativeTranscribe(handle, pcm16, timeoutMillis.coerceAtLeast(1L))
            .map { segment ->
                TranscriptSegment(segment.text, segment.startMillis, segment.endMillis)
            }
    }

    @Synchronized
    override fun close() {
        if (nativeHandle == 0L) return
        nativeDestroy(nativeHandle)
        nativeHandle = 0L
    }

    private fun requireOpenHandle(): Long = checkNotNull(nativeHandle.takeIf { it != 0L }) {
        "Sentence-audio backend is closed"
    }

    private external fun nativeCreate(
        whisperModelPath: String?,
        vadModelPath: String,
        threadCount: Int,
        vadThreshold: Float,
        vadMinSpeechDurationMillis: Int,
        vadMinSilenceDurationMillis: Int,
        vadMaxSpeechDurationSeconds: Float,
        vadSpeechPaddingMillis: Int,
        vadSamplesOverlapSeconds: Float,
    ): Long

    private external fun nativeDestroy(handle: Long)

    private external fun nativeDetectSpeech(handle: Long, pcm16: ShortArray): IntArray

    private external fun nativeTranscribe(
        handle: Long,
        pcm16: ShortArray,
        timeoutMillis: Long,
    ): Array<NativeTranscriptSegment>

    companion object {
        private const val TAG = "SentenceAudioNative"

        init {
            System.loadLibrary("sentence_audio_jni")
        }
    }
}
