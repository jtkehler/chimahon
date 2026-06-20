package chimahon.anki

import io.kotest.matchers.shouldBe
import org.junit.jupiter.api.Test

class AnkiCardCreatorTest {

    @Test
    fun `sentence audio marker is detected in an explicit field map`() {
        val fieldMap = mapOf("SentenceAudio" to "{sentence-audio}")

        AnkiCardCreator.fieldMapContainsMarker(
            fieldMap = fieldMap,
            marker = Marker.SENTENCE_AUDIO,
        ) shouldBe true
    }

    @Test
    fun `similar marker text does not enable sentence audio`() {
        val fieldMap = mapOf("Notes" to "sentence-audio")

        AnkiCardCreator.fieldMapContainsMarker(
            fieldMap = fieldMap,
            marker = Marker.SENTENCE_AUDIO,
        ) shouldBe false
    }

    @Test
    fun `bundled Lapis defaults map sentence audio`() {
        AnkiCardCreator.fieldMapContainsMarker(
            fieldMap = LapisPreset.defaultFieldMap,
            marker = Marker.SENTENCE_AUDIO,
        ) shouldBe true
    }

    @Test
    fun `empty map does not enable sentence audio`() {
        AnkiCardCreator.fieldMapContainsMarker(
            fieldMap = emptyMap(),
            marker = Marker.SENTENCE_AUDIO,
        ) shouldBe false
    }
}
