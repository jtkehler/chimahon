package chimahon.audio

data class SentenceAudioRequest(
    val captureTimestampNanos: Long,
    val ankiButtonTimestampNanos: Long,
    val sentence: String,
    val beforeSeconds: Int,
    val afterSeconds: Int,
)

data class SentenceAudioResult(
    val bytes: ByteArray,
    val extension: String = "wav",
)

fun interface SentenceAudioProvider {
    suspend fun create(request: SentenceAudioRequest): SentenceAudioResult?
}
