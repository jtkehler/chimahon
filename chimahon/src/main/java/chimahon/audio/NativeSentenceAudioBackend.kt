package chimahon.audio

import java.io.File

internal data class NativeTranscriptSegment(
    val text: String,
    val startMillis: Long,
    val endMillis: Long,
)

class NativeSentenceAudioBackend(
    whisperModel: File?,
    vadModel: File,
    threadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
) : SentenceAudioInferenceBackend {
    private var nativeHandle: Long

    init {
        require(whisperModel == null || whisperModel.isFile) { "Whisper model is missing" }
        require(vadModel.isFile) { "Silero VAD model is missing" }
        nativeHandle = nativeCreate(
            whisperModel?.absolutePath,
            vadModel.absolutePath,
            threadCount.coerceAtLeast(1),
        )
        check(nativeHandle != 0L) { "Unable to load sentence-audio models" }
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
    ): Long

    private external fun nativeDestroy(handle: Long)

    private external fun nativeDetectSpeech(handle: Long, pcm16: ShortArray): IntArray

    private external fun nativeTranscribe(
        handle: Long,
        pcm16: ShortArray,
        timeoutMillis: Long,
    ): Array<NativeTranscriptSegment>

    companion object {
        init {
            System.loadLibrary("sentence_audio_jni")
        }
    }
}
