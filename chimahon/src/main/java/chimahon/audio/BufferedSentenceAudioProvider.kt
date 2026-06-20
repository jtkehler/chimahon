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
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Builds sentence audio from a bounded playback-capture buffer.
 *
 * The native pipeline is created only after a complete audio window is available
 * and is retained for subsequent cards until this provider is closed.
 */
class BufferedSentenceAudioProvider(
    private val ringBuffer: TimestampedPcmRingBuffer,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val windowWaitGraceMillis: Long = DEFAULT_WINDOW_WAIT_GRACE_MILLIS,
    private val logWarning: (String, Throwable?) -> Unit = DEFAULT_LOG_WARNING,
    private val logDebug: (String) -> Unit = DEFAULT_LOG_DEBUG,
    private val pipelineFactory: () -> SentenceAudioInferencePipeline?,
) : SentenceAudioProvider, Closeable {
    private val stateLock = Any()
    private val pipelineInitMutex = Mutex()
    private var pipeline: SentenceAudioInferencePipeline? = null
    private var activeCalls = 0
    private var closeRequested = false

    init {
        require(windowWaitGraceMillis > 0) { "Window wait grace period must be positive" }
    }

    override suspend fun create(request: SentenceAudioRequest): SentenceAudioResult? =
        withContext(dispatcher) {
            createOffMainThread(request)
        }

    private suspend fun createOffMainThread(request: SentenceAudioRequest): SentenceAudioResult? {
        if (request.sentence.isBlank() || request.beforeSeconds < 0 || request.afterSeconds < 0) return null

        val beforeNanos = request.beforeSeconds.toDurationNanos() ?: return null
        val afterNanos = request.afterSeconds.toDurationNanos() ?: return null
        val startTimestampNanos = request.captureTimestampNanos - beforeNanos
        val endTimestampNanos = minOf(
            request.captureTimestampNanos + afterNanos,
            request.ankiButtonTimestampNanos,
        )
        if (startTimestampNanos >= endTimestampNanos) return null

        val window = withTimeoutOrNull(windowWaitGraceMillis) {
            ringBuffer.awaitClampedWindow(startTimestampNanos, endTimestampNanos)
        }
        if (window == null) {
            logWarning("Sentence audio unavailable: requested playback window is incomplete", null)
            return null
        }
        if (window.sampleRateHz < SentenceAudioInferencePipeline.SAMPLE_RATE_HZ) {
            logWarning("Sentence audio unavailable: playback sample rate is below the inference rate", null)
            return null
        }

        val stats = window.samples.signalStats()
        logDebug(
            "Audio window: durationMs=${window.durationMillis}, samples=${window.samples.size}, " +
                "beforeOcrMs=${(request.captureTimestampNanos - window.startTimestampNanos) / NANOS_PER_MILLISECOND}, " +
                "afterOcrMs=${(window.endTimestampNanos - request.captureTimestampNanos) / NANOS_PER_MILLISECOND}, " +
                "peak=${stats.peak}, rms=${stats.rms}, aboveNoise=${stats.aboveNoiseSamples}/${window.samples.size}",
        )

        val currentPipeline = acquirePipeline() ?: return null
        return try {
            val inferencePcm = window.samples.downsampleMono(
                sourceSampleRateHz = window.sampleRateHz,
                targetSampleRateHz = SentenceAudioInferencePipeline.SAMPLE_RATE_HZ,
            )
            val segment = currentPipeline.findSegment(
                SentenceAudioInferenceRequest(
                    pcm16 = inferencePcm,
                    sentence = request.sentence,
                    ocrOffsetMillis = (request.captureTimestampNanos - window.startTimestampNanos) /
                        NANOS_PER_MILLISECOND,
                ),
            ) ?: return null
            val outputPcm = window.samples.sliceMillis(
                startMillis = segment.startMillis,
                endMillis = segment.endMillis,
                sampleRateHz = window.sampleRateHz,
            ) ?: return null
            SentenceAudioResult(
                bytes = Pcm16Wav.encode(outputPcm, sampleRateHz = window.sampleRateHz),
            ).also { result ->
                logDebug(
                    "High-quality sentence audio created: segment=${segment.startMillis}-${segment.endMillis}ms, " +
                        "sampleRateHz=${window.sampleRateHz}, samples=${outputPcm.size}, bytes=${result.bytes.size}",
                )
            }
        } catch (error: CancellationException) {
            throw error
        } catch (error: Throwable) {
            logWarning("Sentence audio preparation failed", error)
            null
        } finally {
            releasePipelineCall()
        }
    }

    private suspend fun acquirePipeline(): SentenceAudioInferencePipeline? {
        synchronized(stateLock) {
            if (closeRequested) return null
            pipeline?.let {
                activeCalls++
                return it
            }
        }

        return pipelineInitMutex.withLock {
            synchronized(stateLock) {
                if (closeRequested) return@withLock null
                pipeline?.let {
                    activeCalls++
                    return@withLock it
                }
            }

            val startedAtNanos = System.nanoTime()
            logDebug("Initializing sentence-audio inference pipeline")
            val created = runCatching(pipelineFactory)
                .onFailure { logWarning("Sentence audio models could not be loaded", it) }
                .getOrNull()
            logDebug(
                "Sentence-audio pipeline initialization finished: " +
                    "durationMs=${(System.nanoTime() - startedAtNanos) / NANOS_PER_MILLISECOND}, " +
                    "success=${created != null}",
            )
            if (created == null) {
                logWarning("Sentence audio unavailable: models are not installed", null)
                return@withLock null
            }

            val accepted = synchronized(stateLock) {
                if (closeRequested) {
                    false
                } else {
                    pipeline = created
                    activeCalls++
                    true
                }
            }
            if (accepted) {
                created
            } else {
                created.closeSafely()
                null
            }
        }
    }

    private fun releasePipelineCall() {
        val pipelineToClose = synchronized(stateLock) {
            activeCalls--
            if (closeRequested && activeCalls == 0) takePipeline() else null
        }
        pipelineToClose?.closeSafely()
    }

    override fun close() {
        val pipelineToClose = synchronized(stateLock) {
            closeRequested = true
            if (activeCalls == 0) takePipeline() else null
        }
        pipelineToClose?.closeSafely()
    }

    private fun takePipeline(): SentenceAudioInferencePipeline? = pipeline.also { pipeline = null }

    private fun SentenceAudioInferencePipeline.closeSafely() {
        runCatching { close() }
            .onFailure { logWarning("Unable to release sentence audio models", it) }
    }

    private fun Int.toDurationNanos(): Long? = runCatching {
        Math.multiplyExact(toLong(), NANOS_PER_SECOND)
    }.getOrNull()

    private fun ShortArray.downsampleMono(
        sourceSampleRateHz: Int,
        targetSampleRateHz: Int,
    ): ShortArray {
        require(sourceSampleRateHz >= targetSampleRateHz)
        if (sourceSampleRateHz == targetSampleRateHz) return copyOf()
        val outputSize = (size.toLong() * targetSampleRateHz / sourceSampleRateHz).toInt()
        return ShortArray(outputSize) { outputIndex ->
            val sourceStart = (outputIndex.toLong() * sourceSampleRateHz / targetSampleRateHz).toInt()
            val sourceEnd = (((outputIndex + 1L) * sourceSampleRateHz) / targetSampleRateHz)
                .toInt()
                .coerceAtMost(size)
            var sum = 0L
            for (sourceIndex in sourceStart until sourceEnd) sum += this[sourceIndex]
            (sum / (sourceEnd - sourceStart)).toShort()
        }
    }

    private fun ShortArray.sliceMillis(
        startMillis: Long,
        endMillis: Long,
        sampleRateHz: Int,
    ): ShortArray? {
        val startSample = (startMillis.coerceAtLeast(0) * sampleRateHz / MILLIS_PER_SECOND)
            .coerceAtMost(size.toLong())
            .toInt()
        val endSample = (
            (endMillis.coerceAtLeast(0) * sampleRateHz + MILLIS_PER_SECOND - 1) /
                MILLIS_PER_SECOND
            )
            .coerceAtMost(size.toLong())
            .toInt()
        if (endSample <= startSample) return null
        return copyOfRange(startSample, endSample)
    }

    private val PcmAudioWindow.durationMillis: Long
        get() = (endTimestampNanos - startTimestampNanos) / NANOS_PER_MILLISECOND

    private fun ShortArray.signalStats(): PcmSignalStats {
        if (isEmpty()) return PcmSignalStats(peak = 0, rms = 0, aboveNoiseSamples = 0)
        var peak = 0
        var sumSquares = 0L
        var aboveNoiseSamples = 0
        for (sample in this) {
            val amplitude = if (sample == Short.MIN_VALUE) 32_768 else kotlin.math.abs(sample.toInt())
            peak = maxOf(peak, amplitude)
            sumSquares += amplitude.toLong() * amplitude
            if (amplitude > PCM_NOISE_FLOOR) aboveNoiseSamples++
        }
        return PcmSignalStats(
            peak = peak,
            rms = sqrt(sumSquares.toDouble() / size).roundToInt(),
            aboveNoiseSamples = aboveNoiseSamples,
        )
    }

    private data class PcmSignalStats(
        val peak: Int,
        val rms: Int,
        val aboveNoiseSamples: Int,
    )

    private companion object {
        const val TAG = "BufferedSentenceAudio"
        const val NANOS_PER_MILLISECOND = 1_000_000L
        const val NANOS_PER_SECOND = 1_000_000_000L
        const val MILLIS_PER_SECOND = 1_000L
        const val PCM_NOISE_FLOOR = 128
        const val DEFAULT_WINDOW_WAIT_GRACE_MILLIS = 1_000L
        val DEFAULT_LOG_DEBUG: (String) -> Unit = { message ->
            runCatching { Log.d(TAG, message) }
        }
        val DEFAULT_LOG_WARNING: (String, Throwable?) -> Unit = { message, error ->
            if (error == null) Log.w(TAG, message) else Log.w(TAG, message, error)
        }
    }
}
