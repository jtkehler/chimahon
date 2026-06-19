package eu.kanade.tachiyomi.data.audio

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

class OverlayAudioModelManager internal constructor(
    private val modelDirectory: File,
    private val downloadClient: ModelDownloadClient,
    internal val models: List<OverlayAudioModel>,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + ioDispatcher),
) {

    constructor(
        modelDirectory: File,
        httpClient: OkHttpClient,
    ) : this(
        modelDirectory = modelDirectory,
        downloadClient = OkHttpModelDownloadClient(httpClient),
        models = DEFAULT_MODELS,
    )

    private val operationMutex = Mutex()
    private val _status = MutableStateFlow<OverlayAudioModelStatus>(OverlayAudioModelStatus.Checking)
    val status: StateFlow<OverlayAudioModelStatus> = _status.asStateFlow()

    init {
        refreshStatus()
    }

    fun downloadModels() {
        if (_status.value is OverlayAudioModelStatus.Downloading) return

        scope.launch {
            operationMutex.withLock {
                _status.value = OverlayAudioModelStatus.Downloading
                if (areModelsInstalled()) {
                    _status.value = OverlayAudioModelStatus.Installed
                    return@withLock
                }

                val result = installModels()
                _status.value = result.fold(
                    onSuccess = { OverlayAudioModelStatus.Installed },
                    onFailure = { OverlayAudioModelStatus.Error(it.message ?: "Unknown download error") },
                )
            }
        }
    }

    fun deleteModels() {
        if (_status.value is OverlayAudioModelStatus.Downloading) return

        scope.launch {
            operationMutex.withLock {
                val result = deleteModelsNow()
                _status.value = result.fold(
                    onSuccess = { OverlayAudioModelStatus.Missing },
                    onFailure = { OverlayAudioModelStatus.Error(it.message ?: "Unable to delete models") },
                )
            }
        }
    }

    fun refreshStatus() {
        scope.launch {
            operationMutex.withLock {
                _status.value = OverlayAudioModelStatus.Checking
                _status.value = if (areModelsInstalled()) {
                    OverlayAudioModelStatus.Installed
                } else {
                    OverlayAudioModelStatus.Missing
                }
            }
        }
    }

    fun modelFile(model: OverlayAudioModel): File = File(modelDirectory, model.fileName)

    fun modelFiles(): OverlayAudioModelFiles = OverlayAudioModelFiles(
        whisper = modelFile(models.single { it.fileName == WHISPER_MODEL_FILE_NAME }),
        vad = modelFile(models.single { it.fileName == VAD_MODEL_FILE_NAME }),
    )

    internal suspend fun installModels(): Result<Unit> = withContext(ioDispatcher) {
        runCatching {
            ensureModelDirectory()
            models.forEach { model ->
                val target = modelFile(model)
                if (isValidModel(target, model)) return@forEach

                val temporary = temporaryFile(model)
                deleteIfPresent(temporary)
                try {
                    downloadAndValidate(model, temporary)
                    java.nio.file.Files.move(
                        temporary.toPath(),
                        target.toPath(),
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING,
                    )
                } finally {
                    temporary.delete()
                }
            }
            check(areModelsInstalled()) { "Model validation failed after installation" }
        }
    }

    internal suspend fun deleteModelsNow(): Result<Unit> = withContext(ioDispatcher) {
        runCatching {
            models.forEach { model ->
                deleteIfPresent(modelFile(model))
                deleteIfPresent(temporaryFile(model))
            }
        }
    }

    internal suspend fun areModelsInstalled(): Boolean = withContext(ioDispatcher) {
        models.all { model -> isValidModel(modelFile(model), model) }
    }

    private fun ensureModelDirectory() {
        check(modelDirectory.isDirectory || modelDirectory.mkdirs()) {
            "Unable to create model directory"
        }
    }

    private fun downloadAndValidate(model: OverlayAudioModel, temporary: File) {
        val digest = MessageDigest.getInstance("SHA-256")
        var downloadedBytes = 0L

        FileOutputStream(temporary).use { output ->
            downloadClient.download(model) { input ->
                input.use {
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val count = it.read(buffer)
                        if (count < 0) break
                        downloadedBytes += count
                        check(downloadedBytes <= model.expectedSize) {
                            "${model.displayName} is larger than expected"
                        }
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                }
            }
            output.fd.sync()
        }

        check(downloadedBytes == model.expectedSize) {
            "${model.displayName} has an invalid size"
        }
        check(digest.digest().toHexString() == model.sha256) {
            "${model.displayName} failed checksum validation"
        }
    }

    private fun isValidModel(file: File, model: OverlayAudioModel): Boolean {
        if (!file.isFile || file.length() != model.expectedSize) return false
        return file.inputStream().use { input -> input.sha256() } == model.sha256
    }

    private fun temporaryFile(model: OverlayAudioModel): File = File(modelDirectory, ".${model.fileName}.download")

    private fun deleteIfPresent(file: File) {
        check(!file.exists() || file.delete()) { "Unable to delete ${file.name}" }
    }

    companion object {
        const val MODEL_DIRECTORY = "overlay_audio_models"
        const val WHISPER_MODEL_FILE_NAME = "ggml-base-q5_1.bin"
        const val VAD_MODEL_FILE_NAME = "ggml-silero-v6.2.0.bin"

        internal val DEFAULT_MODELS = listOf(
            OverlayAudioModel(
                displayName = "Whisper base-q5_1",
                fileName = WHISPER_MODEL_FILE_NAME,
                url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/87cd18b47b941d2f65d09981dad23bb7d0481c77/ggml-base-q5_1.bin",
                expectedSize = 59_707_625L,
                sha256 = "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
            ),
            OverlayAudioModel(
                displayName = "Silero VAD v6.2.0",
                fileName = VAD_MODEL_FILE_NAME,
                url = "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin",
                expectedSize = 885_098L,
                sha256 = "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
            ),
        )
    }
}

data class OverlayAudioModelFiles(
    val whisper: File,
    val vad: File,
)

data class OverlayAudioModel(
    val displayName: String,
    val fileName: String,
    val url: String,
    val expectedSize: Long,
    val sha256: String,
)

sealed interface OverlayAudioModelStatus {
    data object Checking : OverlayAudioModelStatus
    data object Missing : OverlayAudioModelStatus
    data object Downloading : OverlayAudioModelStatus
    data object Installed : OverlayAudioModelStatus
    data class Error(val message: String) : OverlayAudioModelStatus
}

internal fun interface ModelDownloadClient {
    fun download(model: OverlayAudioModel, consume: (InputStream) -> Unit)
}

private class OkHttpModelDownloadClient(
    private val httpClient: OkHttpClient,
) : ModelDownloadClient {
    override fun download(model: OverlayAudioModel, consume: (InputStream) -> Unit) {
        val request = Request.Builder().url(model.url).build()
        httpClient.newCall(request).execute().use { response ->
            check(response.isSuccessful) {
                "${model.displayName} download failed: HTTP ${response.code}"
            }
            val body = checkNotNull(response.body) { "${model.displayName} returned an empty response" }
            consume(body.byteStream())
        }
    }
}

private fun InputStream.sha256(): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    while (true) {
        val count = read(buffer)
        if (count < 0) break
        digest.update(buffer, 0, count)
    }
    return digest.digest().toHexString()
}

private fun ByteArray.toHexString(): String = joinToString(separator = "") { byte -> "%02x".format(byte) }
