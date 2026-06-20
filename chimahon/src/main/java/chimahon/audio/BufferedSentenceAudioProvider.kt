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
        if (window.sampleRateHz != SentenceAudioInferencePipeline.SAMPLE_RATE_HZ) {
            logWarning("Sentence audio unavailable: playback sample rate is unsupported", null)
            return null
        }

        val currentPipeline = acquirePipeline() ?: return null
        return try {
            currentPipeline.create(
                SentenceAudioInferenceRequest(
                    pcm16 = window.samples,
                    sentence = request.sentence,
                    ocrOffsetMillis = (request.captureTimestampNanos - window.startTimestampNanos) /
                        NANOS_PER_MILLISECOND,
                ),
            )
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

            val created = runCatching(pipelineFactory)
                .onFailure { logWarning("Sentence audio models could not be loaded", it) }
                .getOrNull()
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

    private companion object {
        const val TAG = "BufferedSentenceAudio"
        const val NANOS_PER_MILLISECOND = 1_000_000L
        const val NANOS_PER_SECOND = 1_000_000_000L
        const val DEFAULT_WINDOW_WAIT_GRACE_MILLIS = 1_000L
        val DEFAULT_LOG_WARNING: (String, Throwable?) -> Unit = { message, error ->
            if (error == null) Log.w(TAG, message) else Log.w(TAG, message, error)
        }
    }
}
