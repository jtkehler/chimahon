package eu.kanade.tachiyomi.ui.dictionary

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import chimahon.audio.TimestampedPcmRingBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

@RequiresApi(Build.VERSION_CODES.Q)
internal class PlaybackAudioCapture(
    private val projection: MediaProjection,
    private val scope: CoroutineScope,
    private val clockNanos: () -> Long = SystemClock::elapsedRealtimeNanos,
) {
    val ringBuffer = TimestampedPcmRingBuffer(
        sampleRateHz = SAMPLE_RATE_HZ,
        capacitySeconds = BUFFER_DURATION_SECONDS,
    )

    private val lock = Any()
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    fun start(): Boolean {
        synchronized(lock) {
            if (audioRecord != null) return true
        }

        val minimumBufferBytes = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minimumBufferBytes <= 0) return false

        val captureConfiguration = android.media.AudioPlaybackCaptureConfiguration.Builder(projection)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .build()
        val bufferSizeBytes = maxOf(minimumBufferBytes, READ_BUFFER_SAMPLES * Short.SIZE_BYTES)
        val recorder = runCatching {
            AudioRecord.Builder()
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE_HZ)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build(),
                )
                .setBufferSizeInBytes(bufferSizeBytes)
                .setAudioPlaybackCaptureConfig(captureConfiguration)
                .build()
        }.onFailure { logcat(LogPriority.WARN, it) { "Unable to create playback AudioRecord" } }
            .getOrNull()
            ?: return false

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return false
        }

        val started = runCatching { recorder.startRecording() }
            .onFailure { logcat(LogPriority.WARN, it) { "Unable to start playback audio capture" } }
            .isSuccess
        if (!started) {
            recorder.release()
            return false
        }

        synchronized(lock) {
            audioRecord = recorder
            captureJob = scope.launch(Dispatchers.IO) { captureLoop(recorder) }
        }
        return true
    }

    fun stop() {
        val (recorder, job) = synchronized(lock) {
            val current = audioRecord to captureJob
            audioRecord = null
            captureJob = null
            current
        }
        job?.cancel()
        recorder?.let(::stopAndRelease)
    }

    private suspend fun captureLoop(recorder: AudioRecord) {
        val readBuffer = ShortArray(READ_BUFFER_SAMPLES)
        try {
            while (kotlin.coroutines.coroutineContext.isActive) {
                val count = recorder.read(
                    readBuffer,
                    0,
                    readBuffer.size,
                    AudioRecord.READ_BLOCKING,
                )
                if (count > 0) {
                    ringBuffer.append(
                        source = readBuffer,
                        count = count,
                        endTimestampNanos = clockNanos(),
                    )
                } else if (count < 0) {
                    logcat(LogPriority.WARN) { "Playback AudioRecord read failed: $count" }
                    break
                }
            }
        } finally {
            val releaseRecorder = synchronized(lock) {
                if (audioRecord === recorder) {
                    audioRecord = null
                    captureJob = null
                    true
                } else {
                    false
                }
            }
            if (releaseRecorder) stopAndRelease(recorder)
        }
    }

    private fun stopAndRelease(recorder: AudioRecord) {
        if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            runCatching { recorder.stop() }
        }
        runCatching { recorder.release() }
    }

    private companion object {
        const val SAMPLE_RATE_HZ = 48_000
        const val BUFFER_DURATION_SECONDS = 180
        const val READ_BUFFER_SAMPLES = 4_096
    }
}
