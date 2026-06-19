package eu.kanade.tachiyomi.ui.dictionary

import android.os.Build
import chimahon.anki.AnkiCardCreator
import chimahon.anki.Marker

internal fun isOverlayAudioCaptureEnabled(preferences: DictionaryPreferences): Boolean {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
    if (!preferences.mineOverlayAudio().get()) return false
    val profile = preferences.profileStore.getActiveProfile()
    return AnkiCardCreator.fieldMapContainsMarker(
        fieldMapJson = profile.ankiFieldMap,
        marker = Marker.SENTENCE_AUDIO,
        modelName = profile.ankiModel,
    )
}
