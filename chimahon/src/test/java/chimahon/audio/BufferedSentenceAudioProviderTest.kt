package chimahon.audio

import io.kotest.matchers.shouldBe
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger

class BufferedSentenceAudioProviderTest {

    @Test
    fun `complete buffered window produces aligned sentence audio`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val ringBuffer = completeWindow(captureTimestampNanos)
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(0, 20_000)),
            transcript = listOf(
                TranscriptSegment("同じ台詞", 1_000, 1_500),
                TranscriptSegment("同じ台詞", 15_000, 16_000),
            ),
        )
        val provider = BufferedSentenceAudioProvider(ringBuffer, logWarning = NO_OP_LOGGER) {
            SentenceAudioInferencePipeline(backend)
        }

        val result = provider.create(request(captureTimestampNanos, sentence = "同じ台詞"))

        result?.bytes?.size shouldBe WAV_HEADER_BYTES + SAMPLE_RATE_HZ * Short.SIZE_BYTES
        backend.transcribeCalls shouldBe 1
        provider.close()
        backend.closed shouldBe true
        Unit
    }

    @Test
    fun `missing capture history does not create the native pipeline`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val ringBuffer = TimestampedPcmRingBuffer(SAMPLE_RATE_HZ, capacitySeconds = 30).apply {
            append(
                source = ShortArray(SAMPLE_RATE_HZ),
                endTimestampNanos = captureTimestampNanos + AFTER_SECONDS * NANOS_PER_SECOND,
            )
        }
        val factoryCalls = AtomicInteger()
        val provider = BufferedSentenceAudioProvider(ringBuffer, logWarning = NO_OP_LOGGER) {
            factoryCalls.incrementAndGet()
            null
        }

        provider.create(request(captureTimestampNanos)) shouldBe null
        factoryCalls.get() shouldBe 0
        provider.close()
        Unit
    }

    @Test
    fun `missing models leave sentence audio empty`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val provider = BufferedSentenceAudioProvider(
            completeWindow(captureTimestampNanos),
            logWarning = NO_OP_LOGGER,
        ) { null }

        provider.create(request(captureTimestampNanos)) shouldBe null
        provider.close()
        Unit
    }

    @Test
    fun `closing during model load releases the created pipeline`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val factoryStarted = CountDownLatch(1)
        val allowFactoryToFinish = CountDownLatch(1)
        val backend = FakeBackend(emptyList(), emptyList())
        val provider = BufferedSentenceAudioProvider(
            completeWindow(captureTimestampNanos),
            logWarning = NO_OP_LOGGER,
        ) {
            factoryStarted.countDown()
            allowFactoryToFinish.await()
            SentenceAudioInferencePipeline(backend)
        }

        val result = async(Dispatchers.Default) { provider.create(request(captureTimestampNanos)) }
        factoryStarted.await()
        provider.close()
        allowFactoryToFinish.countDown()

        result.await() shouldBe null
        backend.closed shouldBe true
        Unit
    }

    private fun completeWindow(captureTimestampNanos: Long): TimestampedPcmRingBuffer {
        return TimestampedPcmRingBuffer(SAMPLE_RATE_HZ, capacitySeconds = 30).apply {
            append(
                source = ShortArray((BEFORE_SECONDS + AFTER_SECONDS) * SAMPLE_RATE_HZ),
                endTimestampNanos = captureTimestampNanos + AFTER_SECONDS * NANOS_PER_SECOND,
            )
        }
    }

    private fun request(captureTimestampNanos: Long, sentence: String = "これはテストです") =
        SentenceAudioRequest(
            captureTimestampNanos = captureTimestampNanos,
            sentence = sentence,
            beforeSeconds = BEFORE_SECONDS,
            afterSeconds = AFTER_SECONDS,
        )

    private class FakeBackend(
        private val speech: List<SpeechSegment>,
        private val transcript: List<TranscriptSegment>,
    ) : SentenceAudioInferenceBackend {
        var transcribeCalls = 0
        var closed = false

        override fun detectSpeech(pcm16: ShortArray): List<SpeechSegment> = speech

        override fun transcribe(pcm16: ShortArray, timeoutMillis: Long): List<TranscriptSegment> {
            transcribeCalls++
            return transcript
        }

        override fun close() {
            closed = true
        }
    }

    private companion object {
        const val SAMPLE_RATE_HZ = 16_000
        const val BEFORE_SECONDS = 15
        const val AFTER_SECONDS = 5
        const val NANOS_PER_SECOND = 1_000_000_000L
        const val WAV_HEADER_BYTES = 44
        val NO_OP_LOGGER: (String, Throwable?) -> Unit = { _, _ -> }
    }
}
