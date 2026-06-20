package eu.kanade.tachiyomi.data.audio

import io.kotest.matchers.booleans.shouldBeFalse
import io.kotest.matchers.booleans.shouldBeTrue
import io.kotest.matchers.shouldBe
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.ByteArrayInputStream
import java.io.IOException
import java.nio.file.Path
import java.security.MessageDigest

class OverlayAudioModelManagerTest {

    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `valid downloads are installed and temporary files are removed`() {
        runBlocking {
            val whisperBytes = "whisper-model".toByteArray()
            val vadBytes = "vad-model".toByteArray()
            val models = listOf(model("whisper.bin", whisperBytes), model("vad.bin", vadBytes))
            val manager = manager(models, mapOf(models[0] to whisperBytes, models[1] to vadBytes))

            manager.installModels().isSuccess.shouldBeTrue()

            manager.modelFile(models[0]).readBytes() shouldBe whisperBytes
            manager.modelFile(models[1]).readBytes() shouldBe vadBytes
            manager.areModelsInstalled().shouldBeTrue()
            temporaryDirectory.toFile().listFiles().orEmpty().none { it.name.endsWith(".download") }.shouldBeTrue()
        }
    }

    @Test
    fun `invalid download does not replace an existing valid model`() {
        runBlocking {
            val validBytes = "valid-model".toByteArray()
            val model = model("whisper.bin", validBytes)
            val target = temporaryDirectory.resolve(model.fileName).toFile().apply { writeBytes(validBytes) }
            var downloadAttempted = false
            val manager = OverlayAudioModelManager(
                modelDirectory = temporaryDirectory.toFile(),
                downloadClient = ModelDownloadClient { _, _ -> downloadAttempted = true },
                models = listOf(model),
            )

            manager.installModels().isSuccess.shouldBeTrue()

            downloadAttempted.shouldBeFalse()
            target.readBytes() shouldBe validBytes
        }
    }

    @Test
    fun `failed validation leaves the prior file untouched and supports retry`() {
        runBlocking {
            val validBytes = "expected-model".toByteArray()
            val invalidBytes = "invalid-model!".toByteArray()
            val model = model("whisper.bin", validBytes)
            val target = temporaryDirectory.resolve(model.fileName).toFile().apply { writeText("old") }
            var responseBytes = invalidBytes
            val manager = OverlayAudioModelManager(
                modelDirectory = temporaryDirectory.toFile(),
                downloadClient = ModelDownloadClient { _, consume -> consume(ByteArrayInputStream(responseBytes)) },
                models = listOf(model),
            )

            manager.installModels().isFailure.shouldBeTrue()
            target.readText() shouldBe "old"

            responseBytes = validBytes
            manager.installModels().isSuccess.shouldBeTrue()
            target.readBytes() shouldBe validBytes
        }
    }

    @Test
    fun `interrupted download leaves installed and prior files untouched`() {
        runBlocking {
            val installedBytes = "installed-model".toByteArray()
            val pendingBytes = "pending-model".toByteArray()
            val installedModel = model("whisper.bin", installedBytes)
            val pendingModel = model("vad.bin", pendingBytes)
            val installedTarget = temporaryDirectory.resolve(installedModel.fileName).toFile().apply {
                writeBytes(installedBytes)
            }
            val priorTarget = temporaryDirectory.resolve(pendingModel.fileName).toFile().apply { writeText("prior") }
            val manager = OverlayAudioModelManager(
                modelDirectory = temporaryDirectory.toFile(),
                downloadClient = ModelDownloadClient { _, _ -> throw IOException("interrupted") },
                models = listOf(installedModel, pendingModel),
            )

            manager.installModels().isFailure.shouldBeTrue()

            installedTarget.readBytes() shouldBe installedBytes
            priorTarget.readText() shouldBe "prior"
            temporaryDirectory.toFile().listFiles().orEmpty().none { it.name.endsWith(".download") }.shouldBeTrue()
        }
    }

    @Test
    fun `delete removes installed models`() {
        runBlocking {
            val bytes = "model".toByteArray()
            val model = model("whisper.bin", bytes)
            val manager = manager(listOf(model), mapOf(model to bytes))
            manager.installModels().isSuccess.shouldBeTrue()

            manager.deleteModelsNow().isSuccess.shouldBeTrue()

            manager.modelFile(model).exists().shouldBeFalse()
            manager.areModelsInstalled().shouldBeFalse()
        }
    }

    private fun manager(
        models: List<OverlayAudioModel>,
        responses: Map<OverlayAudioModel, ByteArray>,
    ) = OverlayAudioModelManager(
        modelDirectory = temporaryDirectory.toFile(),
        downloadClient = ModelDownloadClient { model, consume ->
            consume(ByteArrayInputStream(responses.getValue(model)))
        },
        models = models,
    )

    private fun model(fileName: String, bytes: ByteArray) = OverlayAudioModel(
        displayName = fileName,
        fileName = fileName,
        url = "https://example.invalid/$fileName",
        expectedSize = bytes.size.toLong(),
        sha256 = MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) },
    )
}
