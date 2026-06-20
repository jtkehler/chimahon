package chimahon.audio

import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.shouldBe
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicInteger

class SentenceAudioInferenceTest {

    @Test
    fun `unvoiced samples never invoke Whisper`() = runBlocking {
        val backend = FakeBackend(speech = emptyList())
        val pipeline = SentenceAudioInferencePipeline(backend)

        pipeline.create(request()) shouldBe null

        backend.transcribeCalls shouldBe 0
        pipeline.close()
        backend.closed shouldBe true
        Unit
    }

    @Test
    fun `failed alignment never returns sentence audio`() = runBlocking {
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(0, 1_000)),
            transcript = listOf(TranscriptSegment("まったく違う台詞", 0, 1_000)),
        )
        val pipeline = SentenceAudioInferencePipeline(backend)

        pipeline.create(request(sentence = "これはテストです")) shouldBe null
        backend.transcribeCalls shouldBe 1
        pipeline.close()
    }

    @Test
    fun `successful alignment returns only the matched PCM as WAV`() = runBlocking {
        val pcm = ShortArray(32_000) { it.toShort() }
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(400, 1_100)),
            transcript = listOf(TranscriptSegment("これは、テストです。", 500, 1_000)),
        )
        val pipeline = SentenceAudioInferencePipeline(backend)

        val result = pipeline.create(
            request(
                pcm16 = pcm,
                sentence = "これはテストです",
                ocrOffsetMillis = 750,
            ),
        )

        result?.extension shouldBe "wav"
        result?.bytes?.copyOfRange(0, 4)?.toList() shouldContainExactly
            "RIFF".toByteArray().toList()
        result?.bytes?.size shouldBe 44 + 8_000 * Short.SIZE_BYTES
        ByteBuffer.wrap(result!!.bytes, 40, 4)
            .order(ByteOrder.LITTLE_ENDIAN)
            .int shouldBe 8_000 * Short.SIZE_BYTES
        pipeline.close()
    }

    @Test
    fun `VAD only returns the most recent speech segment before OCR without Whisper`() = runBlocking {
        val backend = FakeBackend(
            speech = listOf(
                SpeechSegment(100, 500),
                SpeechSegment(1_200, 1_700),
                SpeechSegment(2_100, 2_600),
            ),
        )
        val pipeline = SentenceAudioInferencePipeline(backend, vadOnly = true)

        val result = pipeline.create(
            request(
                pcm16 = ShortArray(48_000),
                ocrOffsetMillis = 2_000,
            ),
        )

        result?.bytes?.size shouldBe 44 + 8_000 * Short.SIZE_BYTES
        backend.transcribeCalls shouldBe 0
        pipeline.close()
    }

    @Test
    fun `VAD only ignores speech that starts after OCR`() = runBlocking {
        val backend = FakeBackend(speech = listOf(SpeechSegment(600, 1_000)))
        val pipeline = SentenceAudioInferencePipeline(backend, vadOnly = true)

        pipeline.create(request(ocrOffsetMillis = 500)) shouldBe null

        backend.transcribeCalls shouldBe 0
        pipeline.close()
    }

    @Test
    fun `equal alignment scores prefer the span nearest the OCR timestamp`() {
        val match = SentenceAudioAligner.findBestMatch(
            sentence = "同じ台詞",
            transcript = listOf(
                TranscriptSegment("同じ台詞", 0, 500),
                TranscriptSegment("同じ台詞", 1_500, 2_000),
            ),
            ocrOffsetMillis = 1_800,
            minimumScore = 0.72,
        )

        match?.startMillis shouldBe 1_500
        match?.endMillis shouldBe 2_000
    }

    @Test
    fun `inference is serialized across concurrent requests`() = runBlocking {
        val backend = FakeBackend(
            speech = listOf(SpeechSegment(0, 1_000)),
            transcript = listOf(TranscriptSegment("これはテストです", 0, 1_000)),
            inferenceDelayMillis = 25,
        )
        val pipeline = SentenceAudioInferencePipeline(backend)

        listOf(
            async { pipeline.create(request()) },
            async { pipeline.create(request()) },
        ).awaitAll()

        backend.maximumConcurrentTranscriptions.get() shouldBe 1
        pipeline.close()
    }

    @Test
    fun `normalization removes Japanese punctuation and normalizes width`() {
        SentenceAudioAligner.normalize(" Ｔｅｓｔ、です！ ") shouldBe "testです"
    }

    private fun request(
        pcm16: ShortArray = ShortArray(16_000),
        sentence: String = "これはテストです",
        ocrOffsetMillis: Long = 500,
    ) = SentenceAudioInferenceRequest(
        pcm16 = pcm16,
        sentence = sentence,
        ocrOffsetMillis = ocrOffsetMillis,
    )

    private class FakeBackend(
        private val speech: List<SpeechSegment>,
        private val transcript: List<TranscriptSegment> = emptyList(),
        private val inferenceDelayMillis: Long = 0,
    ) : SentenceAudioInferenceBackend {
        var transcribeCalls = 0
        var closed = false
        val maximumConcurrentTranscriptions = AtomicInteger()
        private val activeTranscriptions = AtomicInteger()

        override fun detectSpeech(pcm16: ShortArray): List<SpeechSegment> = speech

        override fun transcribe(pcm16: ShortArray, timeoutMillis: Long): List<TranscriptSegment> {
            transcribeCalls++
            val active = activeTranscriptions.incrementAndGet()
            maximumConcurrentTranscriptions.updateAndGet { maxOf(it, active) }
            try {
                if (inferenceDelayMillis > 0) Thread.sleep(inferenceDelayMillis)
                return transcript
            } finally {
                activeTranscriptions.decrementAndGet()
            }
        }

        override fun close() {
            closed = true
        }
    }
}
