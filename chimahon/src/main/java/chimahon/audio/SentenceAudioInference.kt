package chimahon.audio

import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import kotlin.math.abs
import kotlin.math.max

data class SentenceAudioInferenceRequest(
    val pcm16: ShortArray,
    val sentence: String,
    val ocrOffsetMillis: Long,
)

data class SpeechSegment(
    val startMillis: Long,
    val endMillis: Long,
)

data class TranscriptSegment(
    val text: String,
    val startMillis: Long,
    val endMillis: Long,
)

internal data class AlignedSentenceSpan(
    val startMillis: Long,
    val endMillis: Long,
    val score: Double,
)

interface SentenceAudioInferenceBackend : Closeable {
    fun detectSpeech(pcm16: ShortArray): List<SpeechSegment>

    fun transcribe(pcm16: ShortArray, timeoutMillis: Long): List<TranscriptSegment>
}

class SentenceAudioInferencePipeline(
    private val backend: SentenceAudioInferenceBackend,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val timeoutMillis: Long = DEFAULT_TIMEOUT_MILLIS,
    private val minimumMatchScore: Double = DEFAULT_MINIMUM_MATCH_SCORE,
    private val vadOnly: Boolean = false,
    private val logDebug: (String) -> Unit = DEFAULT_LOG_DEBUG,
    private val logWarning: (String) -> Unit = DEFAULT_LOG_WARNING,
) : Closeable {
    private val inferenceMutex = Mutex()

    init {
        require(timeoutMillis > 0) { "Inference timeout must be positive" }
        require(minimumMatchScore in 0.0..1.0) { "Match score must be between zero and one" }
    }

    suspend fun create(request: SentenceAudioInferenceRequest): SentenceAudioResult? = withContext(dispatcher) {
        try {
            withTimeout(timeoutMillis) {
                inferenceMutex.withLock {
                    createSerialized(request)
                }
            }
        } catch (_: TimeoutCancellationException) {
            logWarning("Inference timed out after ${timeoutMillis}ms")
            null
        }
    }

    private fun createSerialized(request: SentenceAudioInferenceRequest): SentenceAudioResult? {
        if (request.pcm16.isEmpty() || request.sentence.isBlank()) {
            logDebug("Inference skipped: empty PCM or sentence")
            return null
        }

        return try {
            val startedAtNanos = System.nanoTime()
            logDebug(
                "VAD starting: durationMs=${request.pcm16.size * 1_000L / SAMPLE_RATE_HZ}, " +
                    "ocrOffsetMs=${request.ocrOffsetMillis}, vadOnly=$vadOnly",
            )
            val speechSegments = backend.detectSpeech(request.pcm16)
                .filter { it.endMillis > it.startMillis }
            val vadElapsedMillis = (System.nanoTime() - startedAtNanos) / 1_000_000L
            logDebug(
                "VAD finished: durationMs=$vadElapsedMillis, segmentCount=${speechSegments.size}, " +
                    "segments=${speechSegments.summary()}",
            )
            if (speechSegments.isEmpty()) {
                logDebug("Inference produced no audio: VAD found no speech segments")
                return null
            }

            if (vadOnly) {
                val segment = speechSegments
                    .filter { it.startMillis <= request.ocrOffsetMillis }
                    .maxWithOrNull(compareBy<SpeechSegment> { it.startMillis }.thenBy { it.endMillis })
                if (segment == null) {
                    logDebug(
                        "Inference produced no audio: no VAD segment starts at or before " +
                            "ocrOffsetMs=${request.ocrOffsetMillis}",
                    )
                    return null
                }
                val trimmedPcm = trimPcm(request.pcm16, segment.startMillis, segment.endMillis)
                if (trimmedPcm == null) {
                    logDebug("Inference produced no audio: selected VAD segment could not be trimmed")
                    return null
                }
                val result = SentenceAudioResult(bytes = Pcm16Wav.encode(trimmedPcm))
                logDebug(
                    "VAD-only audio created: segment=${segment.startMillis}-${segment.endMillis}ms, " +
                        "samples=${trimmedPcm.size}, bytes=${result.bytes.size}",
                )
                return result
            }

            val elapsedMillis = (System.nanoTime() - startedAtNanos) / 1_000_000L
            val remainingMillis = (timeoutMillis - elapsedMillis).coerceAtLeast(1L)
            val transcript = backend.transcribe(request.pcm16, remainingMillis)
            val match = SentenceAudioAligner.findBestMatch(
                sentence = request.sentence,
                transcript = transcript,
                ocrOffsetMillis = request.ocrOffsetMillis,
                minimumScore = minimumMatchScore,
            )
            if (match == null) {
                logDebug("Inference produced no audio: transcript did not match the OCR sentence")
                return null
            }

            if (speechSegments.none { it.overlaps(match.startMillis, match.endMillis) }) {
                logDebug("Inference produced no audio: transcript match did not overlap VAD speech")
                return null
            }

            val trimmedPcm = trimPcm(request.pcm16, match.startMillis, match.endMillis)
            if (trimmedPcm == null) {
                logDebug("Inference produced no audio: aligned segment could not be trimmed")
                return null
            }
            SentenceAudioResult(bytes = Pcm16Wav.encode(trimmedPcm)).also { result ->
                logDebug(
                    "Aligned sentence audio created: segment=${match.startMillis}-${match.endMillis}ms, " +
                        "score=${match.score}, samples=${trimmedPcm.size}, bytes=${result.bytes.size}",
                )
            }
        } catch (error: CancellationException) {
            throw error
        } catch (error: Throwable) {
            Log.w(TAG, "Sentence-audio inference failed", error)
            null
        }
    }

    override fun close() {
        backend.close()
    }

    private fun trimPcm(pcm16: ShortArray, startMillis: Long, endMillis: Long): ShortArray? {
        val startSample = millisToSample(startMillis).coerceIn(0, pcm16.size)
        val endSample = millisToSample(endMillis).coerceIn(0, pcm16.size)
        if (endSample <= startSample) return null
        return pcm16.copyOfRange(startSample, endSample)
    }

    private fun millisToSample(millis: Long): Int =
        ((millis.coerceAtLeast(0) * SAMPLE_RATE_HZ) / 1_000L)
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()

    private fun SpeechSegment.overlaps(startMillis: Long, endMillis: Long): Boolean =
        this.startMillis < endMillis && this.endMillis > startMillis

    private fun List<SpeechSegment>.summary(): String {
        if (isEmpty()) return "[]"
        val shown = take(MAX_LOGGED_SEGMENTS).joinToString(prefix = "[", postfix = "]") {
            "${it.startMillis}-${it.endMillis}"
        }
        return if (size > MAX_LOGGED_SEGMENTS) "$shown+${size - MAX_LOGGED_SEGMENTS}more" else shown
    }

    companion object {
        const val SAMPLE_RATE_HZ = 16_000
        const val DEFAULT_TIMEOUT_MILLIS = 30_000L
        const val DEFAULT_MINIMUM_MATCH_SCORE = 0.72
        private const val MAX_LOGGED_SEGMENTS = 8
        private const val TAG = "SentenceAudioInference"
        private val DEFAULT_LOG_DEBUG: (String) -> Unit = { message ->
            runCatching { Log.d(TAG, message) }
        }
        private val DEFAULT_LOG_WARNING: (String) -> Unit = { message ->
            runCatching { Log.w(TAG, message) }
        }
    }
}

internal object SentenceAudioAligner {
    fun findBestMatch(
        sentence: String,
        transcript: List<TranscriptSegment>,
        ocrOffsetMillis: Long,
        minimumScore: Double,
    ): AlignedSentenceSpan? {
        val target = normalize(sentence)
        if (target.isEmpty()) return null

        var best: AlignedSentenceSpan? = null
        transcript.forEachIndexed { startIndex, startSegment ->
            if (startSegment.endMillis <= startSegment.startMillis) return@forEachIndexed

            val candidate = StringBuilder()
            for (endIndex in startIndex until transcript.size) {
                val endSegment = transcript[endIndex]
                if (endSegment.endMillis <= endSegment.startMillis) continue
                candidate.append(normalize(endSegment.text))
                if (candidate.isEmpty()) continue

                val score = similarity(target, candidate.toString())
                val span = AlignedSentenceSpan(
                    startMillis = startSegment.startMillis,
                    endMillis = endSegment.endMillis,
                    score = score,
                )
                if (isBetter(span, best, ocrOffsetMillis)) best = span

                if (candidate.length > target.length * MAX_CANDIDATE_LENGTH_MULTIPLIER) break
            }
        }

        return best?.takeIf { it.score >= minimumScore }
    }

    internal fun normalize(text: String): String = buildString(text.length) {
        Normalizer.normalize(text, Normalizer.Form.NFKC).forEach { character ->
            if (character.isLetterOrDigit()) append(character.lowercaseChar())
        }
    }

    private fun isBetter(
        candidate: AlignedSentenceSpan,
        current: AlignedSentenceSpan?,
        ocrOffsetMillis: Long,
    ): Boolean {
        if (current == null || candidate.score > current.score + SCORE_EPSILON) return true
        if (abs(candidate.score - current.score) > SCORE_EPSILON) return false
        return candidate.distanceFrom(ocrOffsetMillis) < current.distanceFrom(ocrOffsetMillis)
    }

    private fun AlignedSentenceSpan.distanceFrom(timestampMillis: Long): Long =
        abs(((startMillis + endMillis) / 2L) - timestampMillis)

    private fun similarity(expected: String, actual: String): Double {
        val longest = max(expected.length, actual.length)
        if (longest == 0) return 1.0
        return 1.0 - levenshteinDistance(expected, actual).toDouble() / longest
    }

    private fun levenshteinDistance(left: String, right: String): Int {
        if (left.isEmpty()) return right.length
        if (right.isEmpty()) return left.length

        var previous = IntArray(right.length + 1) { it }
        var current = IntArray(right.length + 1)
        left.forEachIndexed { leftIndex, leftCharacter ->
            current[0] = leftIndex + 1
            right.forEachIndexed { rightIndex, rightCharacter ->
                val substitution = previous[rightIndex] +
                    if (leftCharacter == rightCharacter) 0 else 1
                current[rightIndex + 1] = minOf(
                    current[rightIndex] + 1,
                    previous[rightIndex + 1] + 1,
                    substitution,
                )
            }
            val swap = previous
            previous = current
            current = swap
        }
        return previous[right.length]
    }

    private const val MAX_CANDIDATE_LENGTH_MULTIPLIER = 2
    private const val SCORE_EPSILON = 0.000_001
}

internal object Pcm16Wav {
    fun encode(samples: ShortArray): ByteArray {
        val pcmBytes = samples.size * Short.SIZE_BYTES
        return ByteBuffer.allocate(WAV_HEADER_BYTES + pcmBytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply {
                put("RIFF".toByteArray(Charsets.US_ASCII))
                putInt(36 + pcmBytes)
                put("WAVE".toByteArray(Charsets.US_ASCII))
                put("fmt ".toByteArray(Charsets.US_ASCII))
                putInt(16)
                putShort(1)
                putShort(1)
                putInt(SentenceAudioInferencePipeline.SAMPLE_RATE_HZ)
                putInt(SentenceAudioInferencePipeline.SAMPLE_RATE_HZ * Short.SIZE_BYTES)
                putShort(Short.SIZE_BYTES.toShort())
                putShort(Short.SIZE_BITS.toShort())
                put("data".toByteArray(Charsets.US_ASCII))
                putInt(pcmBytes)
                samples.forEach { putShort(it) }
            }
            .array()
    }

    private const val WAV_HEADER_BYTES = 44
}
