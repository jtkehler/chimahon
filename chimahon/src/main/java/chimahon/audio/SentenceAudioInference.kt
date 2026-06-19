package chimahon.audio

import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
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
) : Closeable {
    private val inferenceMutex = Mutex()

    init {
        require(timeoutMillis > 0) { "Inference timeout must be positive" }
        require(minimumMatchScore in 0.0..1.0) { "Match score must be between zero and one" }
    }

    suspend fun create(request: SentenceAudioInferenceRequest): SentenceAudioResult? =
        withContext(dispatcher) {
            withTimeoutOrNull(timeoutMillis) {
                inferenceMutex.withLock {
                    createSerialized(request)
                }
            }
        }

    private fun createSerialized(request: SentenceAudioInferenceRequest): SentenceAudioResult? {
        if (request.pcm16.isEmpty() || request.sentence.isBlank()) return null

        return try {
            val startedAtNanos = System.nanoTime()
            val speechSegments = backend.detectSpeech(request.pcm16)
                .filter { it.endMillis > it.startMillis }
            if (speechSegments.isEmpty()) return null

            val elapsedMillis = (System.nanoTime() - startedAtNanos) / 1_000_000L
            val remainingMillis = (timeoutMillis - elapsedMillis).coerceAtLeast(1L)
            val transcript = backend.transcribe(request.pcm16, remainingMillis)
            val match = SentenceAudioAligner.findBestMatch(
                sentence = request.sentence,
                transcript = transcript,
                ocrOffsetMillis = request.ocrOffsetMillis,
                minimumScore = minimumMatchScore,
            ) ?: return null

            if (speechSegments.none { it.overlaps(match.startMillis, match.endMillis) }) return null

            val trimmedPcm = trimPcm(request.pcm16, match.startMillis, match.endMillis)
                ?: return null
            SentenceAudioResult(bytes = Pcm16Wav.encode(trimmedPcm))
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

    companion object {
        const val SAMPLE_RATE_HZ = 16_000
        const val DEFAULT_TIMEOUT_MILLIS = 30_000L
        const val DEFAULT_MINIMUM_MATCH_SCORE = 0.72
        private const val TAG = "SentenceAudioInference"
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
