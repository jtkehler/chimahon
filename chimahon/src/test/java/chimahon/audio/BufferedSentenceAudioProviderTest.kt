package chimahon.audio

import io.kotest.matchers.shouldBe
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger

class BufferedSentenceAudioProviderTest {

    @Test
    fun `complete buffered window produces aligned sentence audio`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val ringBuffer = completeWindow(captureTimestampNanos)
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(15_000, 16_000)),
            transcript = listOf(
                TranscriptSegment("同じ台詞", 1_000, 1_500),
                TranscriptSegment("同じ台詞", 15_000, 16_000),
            ),
        )
        val provider = BufferedSentenceAudioProvider(ringBuffer, logWarning = NO_OP_LOGGER) {
            SentenceAudioInferencePipeline(backend)
        }

        val result = provider.create(request(captureTimestampNanos, sentence = "同じ台詞"))

        result?.bytes?.size shouldBe WAV_HEADER_BYTES + CAPTURE_SAMPLE_RATE_HZ * Short.SIZE_BYTES
        ByteBuffer.wrap(result!!.bytes, 24, 4).order(ByteOrder.LITTLE_ENDIAN).int shouldBe
            CAPTURE_SAMPLE_RATE_HZ
        ByteBuffer.wrap(result.bytes, WAV_HEADER_BYTES, Short.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .short shouldBe 16_000
        backend.detectSpeechCalls shouldBe 0
        backend.transcribedSampleCount shouldBe (BEFORE_SECONDS + AFTER_SECONDS) * INFERENCE_SAMPLE_RATE_HZ
        backend.transcribeCalls shouldBe 1
        provider.close()
        backend.closed shouldBe true
        Unit
    }

    @Test
    fun `pre-roll is clamped to available capture history`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val ringBuffer = TimestampedPcmRingBuffer(CAPTURE_SAMPLE_RATE_HZ, capacitySeconds = 30).apply {
            append(
                source = ShortArray(10 * CAPTURE_SAMPLE_RATE_HZ),
                endTimestampNanos = captureTimestampNanos + AFTER_SECONDS * NANOS_PER_SECOND,
            )
        }
        val factoryCalls = AtomicInteger()
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(0, 10_000)),
            transcript = listOf(TranscriptSegment("これはテストです", 5_000, 6_000)),
        )
        val provider = BufferedSentenceAudioProvider(ringBuffer, logWarning = NO_OP_LOGGER) {
            factoryCalls.incrementAndGet()
            SentenceAudioInferencePipeline(backend)
        }

        provider.create(request(captureTimestampNanos))?.bytes?.isNotEmpty() shouldBe true
        factoryCalls.get() shouldBe 1
        backend.detectSpeechCalls shouldBe 0
        backend.transcribedSampleCount shouldBe 10 * INFERENCE_SAMPLE_RATE_HZ
        provider.close()
        Unit
    }

    @Test
    fun `post-roll is clamped to the Anki button timestamp`() = runBlocking {
        val captureTimestampNanos = 20 * NANOS_PER_SECOND
        val ankiButtonTimestampNanos = captureTimestampNanos + 2 * NANOS_PER_SECOND
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(0, 17_000)),
            transcript = listOf(TranscriptSegment("これはテストです", 15_000, 16_000)),
        )
        val provider = BufferedSentenceAudioProvider(
            completeWindow(captureTimestampNanos),
            logWarning = NO_OP_LOGGER,
        ) { SentenceAudioInferencePipeline(backend) }

        provider.create(
            request(
                captureTimestampNanos = captureTimestampNanos,
                ankiButtonTimestampNanos = ankiButtonTimestampNanos,
            ),
        )?.bytes?.isNotEmpty() shouldBe true
        backend.detectSpeechCalls shouldBe 0
        backend.transcribedSampleCount shouldBe 17 * INFERENCE_SAMPLE_RATE_HZ
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
        return TimestampedPcmRingBuffer(CAPTURE_SAMPLE_RATE_HZ, capacitySeconds = 30).apply {
            append(
                source = ShortArray((BEFORE_SECONDS + AFTER_SECONDS) * CAPTURE_SAMPLE_RATE_HZ) {
                    (it % 32_000).toShort()
                },
                endTimestampNanos = captureTimestampNanos + AFTER_SECONDS * NANOS_PER_SECOND,
            )
        }
    }

    private fun request(
        captureTimestampNanos: Long,
        ankiButtonTimestampNanos: Long = captureTimestampNanos + AFTER_SECONDS * NANOS_PER_SECOND,
        sentence: String = "これはテストです",
    ) =
        SentenceAudioRequest(
            captureTimestampNanos = captureTimestampNanos,
            ankiButtonTimestampNanos = ankiButtonTimestampNanos,
            sentence = sentence,
            beforeSeconds = BEFORE_SECONDS,
            afterSeconds = AFTER_SECONDS,
        )

    private class FakeBackend(
        private val speech: List<SpeechSegment>,
        private val transcript: List<TranscriptSegment>,
    ) : SentenceAudioInferenceBackend {
        var detectSpeechCalls = 0
        var transcribeCalls = 0
        var closed = false
        var transcribedSampleCount = 0

        override fun detectSpeech(pcm16: ShortArray): List<SpeechSegment> {
            detectSpeechCalls++
            return speech
        }

        override fun transcribe(pcm16: ShortArray, timeoutMillis: Long): List<TranscriptSegment> {
            transcribeCalls++
            transcribedSampleCount = pcm16.size
            return transcript
        }

        override fun close() {
            closed = true
        }
    }

    private companion object {
        const val CAPTURE_SAMPLE_RATE_HZ = 48_000
        const val INFERENCE_SAMPLE_RATE_HZ = 16_000
        const val BEFORE_SECONDS = 15
        const val AFTER_SECONDS = 5
        const val NANOS_PER_SECOND = 1_000_000_000L
        const val WAV_HEADER_BYTES = 44
        val NO_OP_LOGGER: (String, Throwable?) -> Unit = { _, _ -> }
    }
}
