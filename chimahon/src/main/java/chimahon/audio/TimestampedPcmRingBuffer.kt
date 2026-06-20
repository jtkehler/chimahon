package chimahon.audio

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first

data class PcmAudioWindow(
    val samples: ShortArray,
    val sampleRateHz: Int,
    val startTimestampNanos: Long,
    val endTimestampNanos: Long,
)

/**
 * Thread-safe PCM16 mono ring buffer anchored to a monotonic clock.
 *
 * [append] timestamps the exclusive end of each audio chunk. AudioRecord reads
 * are continuous, so retained samples are treated as one contiguous timeline.
 */
class TimestampedPcmRingBuffer(
    val sampleRateHz: Int,
    capacitySeconds: Int,
) {
    private val samples = ShortArray(Math.multiplyExact(sampleRateHz, capacitySeconds))
    private val updates = MutableStateFlow(0L)

    private var writeIndex = 0
    private var retainedSamples = 0
    private var newestSampleEndNanos: Long? = null

    init {
        require(sampleRateHz > 0) { "sampleRateHz must be positive" }
        require(capacitySeconds > 0) { "capacitySeconds must be positive" }
    }

    val capacitySamples: Int
        get() = samples.size

    @Synchronized
    fun append(
        source: ShortArray,
        count: Int = source.size,
        endTimestampNanos: Long,
    ) {
        require(count in 0..source.size) { "count must be within source bounds" }
        if (count == 0) return
        require(newestSampleEndNanos?.let { endTimestampNanos >= it } != false) {
            "endTimestampNanos must be monotonic"
        }

        if (count >= samples.size) {
            source.copyInto(
                destination = samples,
                destinationOffset = 0,
                startIndex = count - samples.size,
                endIndex = count,
            )
            writeIndex = 0
            retainedSamples = samples.size
        } else {
            val firstCopyCount = minOf(count, samples.size - writeIndex)
            source.copyInto(
                destination = samples,
                destinationOffset = writeIndex,
                startIndex = 0,
                endIndex = firstCopyCount,
            )
            val remaining = count - firstCopyCount
            if (remaining > 0) {
                source.copyInto(
                    destination = samples,
                    destinationOffset = 0,
                    startIndex = firstCopyCount,
                    endIndex = count,
                )
            }
            writeIndex = (writeIndex + count) % samples.size
            retainedSamples = minOf(samples.size, retainedSamples + count)
        }

        newestSampleEndNanos = endTimestampNanos
        updates.value++
    }

    /** Returns null until the complete interval is retained. */
    @Synchronized
    fun extract(
        startTimestampNanos: Long,
        endTimestampNanos: Long,
    ): PcmAudioWindow? {
        require(startTimestampNanos < endTimestampNanos) { "Audio window must have positive duration" }
        val newestEnd = newestSampleEndNanos ?: return null
        val oldestStart = newestEnd - samplesToNanos(retainedSamples)
        if (startTimestampNanos < oldestStart || endTimestampNanos > newestEnd) return null

        val startOffset = nanosToSamplesFloor(startTimestampNanos - oldestStart)
        val endOffset = nanosToSamplesCeil(endTimestampNanos - oldestStart)
            .coerceAtMost(retainedSamples)
        if (startOffset >= endOffset) return null

        val result = ShortArray(endOffset - startOffset)
        val oldestIndex = floorMod(writeIndex - retainedSamples, samples.size)
        val sourceStart = (oldestIndex + startOffset) % samples.size
        val firstCopyCount = minOf(result.size, samples.size - sourceStart)
        samples.copyInto(result, 0, sourceStart, sourceStart + firstCopyCount)
        if (firstCopyCount < result.size) {
            samples.copyInto(result, firstCopyCount, 0, result.size - firstCopyCount)
        }

        return PcmAudioWindow(
            samples = result,
            sampleRateHz = sampleRateHz,
            startTimestampNanos = startTimestampNanos,
            endTimestampNanos = endTimestampNanos,
        )
    }

    /**
     * Suspends while post-capture audio is still arriving. Returns null when
     * the requested pre-capture history has already fallen out of the buffer.
     */
    suspend fun awaitWindow(
        startTimestampNanos: Long,
        endTimestampNanos: Long,
    ): PcmAudioWindow? = awaitWindow(
        startTimestampNanos = startTimestampNanos,
        endTimestampNanos = endTimestampNanos,
        clampStartToRetainedAudio = false,
    )

    /**
     * Suspends until [endTimestampNanos] is retained and clamps the requested
     * start to the oldest audio still retained from this capture session.
     */
    suspend fun awaitClampedWindow(
        startTimestampNanos: Long,
        endTimestampNanos: Long,
    ): PcmAudioWindow? = awaitWindow(
        startTimestampNanos = startTimestampNanos,
        endTimestampNanos = endTimestampNanos,
        clampStartToRetainedAudio = true,
    )

    private suspend fun awaitWindow(
        startTimestampNanos: Long,
        endTimestampNanos: Long,
        clampStartToRetainedAudio: Boolean,
    ): PcmAudioWindow? {
        require(startTimestampNanos < endTimestampNanos) { "Audio window must have positive duration" }
        while (true) {
            val observedUpdate = updates.value
            when (
                val availability = availability(
                    startTimestampNanos,
                    endTimestampNanos,
                    clampStartToRetainedAudio,
                )
            ) {
                is Availability.Ready -> return availability.window
                Availability.MissingHistory -> return null
                Availability.Waiting -> updates.first { it != observedUpdate }
            }
        }
    }

    @Synchronized
    private fun availability(
        startTimestampNanos: Long,
        endTimestampNanos: Long,
        clampStartToRetainedAudio: Boolean,
    ): Availability {
        val newestEnd = newestSampleEndNanos ?: return Availability.Waiting
        val oldestStart = newestEnd - samplesToNanos(retainedSamples)
        val effectiveStart = if (clampStartToRetainedAudio) {
            maxOf(startTimestampNanos, oldestStart)
        } else {
            if (startTimestampNanos < oldestStart) return Availability.MissingHistory
            startTimestampNanos
        }
        if (effectiveStart >= endTimestampNanos) return Availability.MissingHistory
        if (endTimestampNanos > newestEnd) return Availability.Waiting
        return extract(effectiveStart, endTimestampNanos)
            ?.let(Availability::Ready)
            ?: Availability.MissingHistory
    }

    private fun samplesToNanos(sampleCount: Int): Long =
        sampleCount.toLong() * NANOS_PER_SECOND / sampleRateHz

    private fun nanosToSamplesFloor(durationNanos: Long): Int =
        (durationNanos * sampleRateHz / NANOS_PER_SECOND).toInt()

    private fun nanosToSamplesCeil(durationNanos: Long): Int =
        ((durationNanos * sampleRateHz + NANOS_PER_SECOND - 1) / NANOS_PER_SECOND).toInt()

    private sealed interface Availability {
        data object Waiting : Availability
        data object MissingHistory : Availability
        data class Ready(val window: PcmAudioWindow) : Availability
    }

    private companion object {
        const val NANOS_PER_SECOND = 1_000_000_000L
    }
}

private fun floorMod(value: Int, divisor: Int): Int = ((value % divisor) + divisor) % divisor
