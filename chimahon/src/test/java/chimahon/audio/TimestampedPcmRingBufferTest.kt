package chimahon.audio

import io.kotest.matchers.booleans.shouldBeFalse
import io.kotest.matchers.shouldBe
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.jupiter.api.Test

class TimestampedPcmRingBufferTest {

    @Test
    fun `wraparound retains newest samples in time order`() {
        val buffer = TimestampedPcmRingBuffer(sampleRateHz = 4, capacitySeconds = 2)
        buffer.append(shortArrayOf(1, 2, 3, 4), endTimestampNanos = seconds(1.0))
        buffer.append(shortArrayOf(5, 6, 7, 8), endTimestampNanos = seconds(2.0))
        buffer.append(shortArrayOf(9, 10, 11, 12), endTimestampNanos = seconds(3.0))

        buffer.extract(seconds(1.5), seconds(2.5))?.samples?.toList() shouldBe
            listOf<Short>(7, 8, 9, 10)
        buffer.capacitySamples shouldBe 8
    }

    @Test
    fun `extract returns the requested retained time window`() {
        val buffer = TimestampedPcmRingBuffer(sampleRateHz = 4, capacitySeconds = 4)
        buffer.append(
            shortArrayOf(1, 2, 3, 4, 5, 6, 7, 8),
            endTimestampNanos = seconds(2.0),
        )

        val window = buffer.extract(seconds(0.5), seconds(1.5))

        window?.samples?.toList() shouldBe listOf<Short>(3, 4, 5, 6)
        window?.startTimestampNanos shouldBe seconds(0.5)
        window?.endTimestampNanos shouldBe seconds(1.5)
    }

    @Test
    fun `extract reports missing pre-capture history`() {
        val buffer = TimestampedPcmRingBuffer(sampleRateHz = 4, capacitySeconds = 1)
        buffer.append(shortArrayOf(5, 6, 7, 8), endTimestampNanos = seconds(2.0))

        buffer.extract(seconds(0.5), seconds(1.5)) shouldBe null
    }

    @Test
    fun `awaitWindow waits for the post-capture interval`() {
        runBlocking {
            val buffer = TimestampedPcmRingBuffer(sampleRateHz = 4, capacitySeconds = 4)
            buffer.append(shortArrayOf(1, 2, 3, 4), endTimestampNanos = seconds(1.0))

            val pending = async { buffer.awaitWindow(seconds(0.5), seconds(1.5)) }
            yield()
            pending.isCompleted.shouldBeFalse()

            buffer.append(shortArrayOf(5, 6), endTimestampNanos = seconds(1.5))

            pending.await()?.samples?.toList() shouldBe listOf<Short>(3, 4, 5, 6)
        }
    }

    private fun seconds(value: Double): Long = (value * 1_000_000_000L).toLong()
}
