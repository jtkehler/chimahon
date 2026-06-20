#include <jni.h>

#include <android/log.h>
#include <whisper.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr char kLogTag[] = "SentenceAudioNative";

constexpr jint vad_centiseconds_to_milliseconds(float centiseconds) {
    return static_cast<jint>(centiseconds * 10.0F);
}

static_assert(vad_centiseconds_to_milliseconds(704.0F) == 7040);

struct SentenceAudioEngine {
    whisper_context * whisper = nullptr;
    whisper_vad_context * vad = nullptr;
    std::string vad_model_path;
    int thread_count = 1;
    float vad_threshold = 0.60F;
    int vad_min_speech_duration_ms = 300;
    int vad_min_silence_duration_ms = 700;
    float vad_max_speech_duration_seconds = 20.0F;
    int vad_speech_padding_ms = 100;
    float vad_samples_overlap_seconds = 0.10F;
    std::mutex mutex;

    ~SentenceAudioEngine() {
        if (whisper != nullptr) {
            whisper_free(whisper);
        }
        if (vad != nullptr) {
            whisper_vad_free(vad);
        }
    }
};

struct AbortDeadline {
    std::chrono::steady_clock::time_point value;
};

bool abort_after_deadline(void * user_data) {
    const auto * deadline = static_cast<AbortDeadline *>(user_data);
    return std::chrono::steady_clock::now() >= deadline->value;
}

void log_error(const char * message) {
    __android_log_write(ANDROID_LOG_WARN, kLogTag, message);
}

std::string to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char * characters = env->GetStringUTFChars(value, nullptr);
    if (characters == nullptr) {
        return {};
    }
    std::string result(characters);
    env->ReleaseStringUTFChars(value, characters);
    return result;
}

std::vector<float> to_float_pcm(JNIEnv * env, jshortArray pcm16) {
    if (pcm16 == nullptr) {
        return {};
    }
    const jsize size = env->GetArrayLength(pcm16);
    if (size <= 0 || size > std::numeric_limits<int>::max()) {
        return {};
    }

    std::vector<jshort> input(static_cast<size_t>(size));
    env->GetShortArrayRegion(pcm16, 0, size, input.data());
    if (env->ExceptionCheck()) {
        return {};
    }

    std::vector<float> output(static_cast<size_t>(size));
    std::transform(input.begin(), input.end(), output.begin(), [](jshort sample) {
        return static_cast<float>(sample) / 32768.0F;
    });
    return output;
}

SentenceAudioEngine * from_handle(jlong handle) {
    return reinterpret_cast<SentenceAudioEngine *>(handle);
}

jintArray empty_int_array(JNIEnv * env) {
    return env->NewIntArray(0);
}

jobjectArray empty_transcript_array(JNIEnv * env) {
    jclass segment_class = env->FindClass("chimahon/audio/NativeTranscriptSegment");
    if (segment_class == nullptr) {
        return nullptr;
    }
    return env->NewObjectArray(0, segment_class, nullptr);
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_chimahon_audio_NativeSentenceAudioBackend_nativeCreate(
        JNIEnv * env,
        jobject,
        jstring whisper_model_path,
        jstring vad_model_path,
        jint thread_count,
        jfloat vad_threshold,
        jint vad_min_speech_duration_ms,
        jint vad_min_silence_duration_ms,
        jfloat vad_max_speech_duration_seconds,
        jint vad_speech_padding_ms,
        jfloat vad_samples_overlap_seconds) {
    const std::string whisper_path = to_string(env, whisper_model_path);
    const std::string vad_path = to_string(env, vad_model_path);
    if (vad_path.empty()) {
        return 0;
    }

    auto engine = std::make_unique<SentenceAudioEngine>();
    engine->vad_model_path = vad_path;
    engine->thread_count = std::max(1, static_cast<int>(thread_count));
    engine->vad_threshold = std::clamp(static_cast<float>(vad_threshold), 0.0F, 1.0F);
    engine->vad_min_speech_duration_ms = std::max(0, static_cast<int>(vad_min_speech_duration_ms));
    engine->vad_min_silence_duration_ms = std::max(0, static_cast<int>(vad_min_silence_duration_ms));
    engine->vad_max_speech_duration_seconds = std::max(0.001F, static_cast<float>(vad_max_speech_duration_seconds));
    engine->vad_speech_padding_ms = std::max(0, static_cast<int>(vad_speech_padding_ms));
    engine->vad_samples_overlap_seconds = std::max(0.0F, static_cast<float>(vad_samples_overlap_seconds));

    if (!whisper_path.empty()) {
        whisper_context_params whisper_params = whisper_context_default_params();
        whisper_params.use_gpu = false;
        whisper_params.flash_attn = false;
        engine->whisper = whisper_init_from_file_with_params(whisper_path.c_str(), whisper_params);
        if (engine->whisper == nullptr) {
            log_error("Unable to load Whisper model");
            return 0;
        }
    }

    if (engine->whisper == nullptr) {
        whisper_vad_context_params vad_context_params = whisper_vad_default_context_params();
        vad_context_params.n_threads = engine->thread_count;
        vad_context_params.use_gpu = false;
        engine->vad = whisper_vad_init_from_file_with_params(vad_path.c_str(), vad_context_params);
        if (engine->vad == nullptr) {
            log_error("Unable to load Silero VAD model");
            return 0;
        }
    }

    return reinterpret_cast<jlong>(engine.release());
}

extern "C" JNIEXPORT void JNICALL
Java_chimahon_audio_NativeSentenceAudioBackend_nativeDestroy(
        JNIEnv *,
        jobject,
        jlong handle) {
    delete from_handle(handle);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_chimahon_audio_NativeSentenceAudioBackend_nativeDetectSpeech(
        JNIEnv * env,
        jobject,
        jlong handle,
        jshortArray pcm16) {
    SentenceAudioEngine * engine = from_handle(handle);
    if (engine == nullptr || engine->vad == nullptr) {
        return empty_int_array(env);
    }

    const std::vector<float> samples = to_float_pcm(env, pcm16);
    if (samples.empty()) {
        return empty_int_array(env);
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    if (!whisper_vad_detect_speech(
            engine->vad,
            samples.data(),
            static_cast<int>(samples.size()))) {
        log_error("Silero VAD inference failed");
        return empty_int_array(env);
    }

    whisper_vad_params vad_params = whisper_vad_default_params();
    vad_params.threshold = engine->vad_threshold;
    vad_params.min_speech_duration_ms = engine->vad_min_speech_duration_ms;
    vad_params.min_silence_duration_ms = engine->vad_min_silence_duration_ms;
    vad_params.max_speech_duration_s = engine->vad_max_speech_duration_seconds;
    vad_params.speech_pad_ms = engine->vad_speech_padding_ms;
    vad_params.samples_overlap = engine->vad_samples_overlap_seconds;

    whisper_vad_segments * segments = whisper_vad_segments_from_probs(engine->vad, vad_params);
    if (segments == nullptr) {
        return empty_int_array(env);
    }

    const int segment_count = whisper_vad_segments_n_segments(segments);
    std::vector<jint> boundaries;
    boundaries.reserve(static_cast<size_t>(std::max(0, segment_count)) * 2U);
    for (int index = 0; index < segment_count; ++index) {
        const float start_centiseconds = whisper_vad_segments_get_segment_t0(segments, index);
        const float end_centiseconds = whisper_vad_segments_get_segment_t1(segments, index);
        if (end_centiseconds <= start_centiseconds) {
            continue;
        }
        boundaries.push_back(vad_centiseconds_to_milliseconds(start_centiseconds));
        boundaries.push_back(vad_centiseconds_to_milliseconds(end_centiseconds));
    }
    whisper_vad_free_segments(segments);

    jintArray result = env->NewIntArray(static_cast<jsize>(boundaries.size()));
    if (result != nullptr && !boundaries.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(boundaries.size()), boundaries.data());
    }
    return result;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_chimahon_audio_NativeSentenceAudioBackend_nativeTranscribe(
        JNIEnv * env,
        jobject,
        jlong handle,
        jshortArray pcm16,
        jlong timeout_millis) {
    SentenceAudioEngine * engine = from_handle(handle);
    if (engine == nullptr || engine->whisper == nullptr) {
        return empty_transcript_array(env);
    }

    const std::vector<float> samples = to_float_pcm(env, pcm16);
    if (samples.empty()) {
        return empty_transcript_array(env);
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    AbortDeadline deadline{
        std::chrono::steady_clock::now() +
            std::chrono::milliseconds(std::max<jlong>(1, timeout_millis)),
    };

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = engine->thread_count;
    params.translate = false;
    params.no_context = true;
    params.no_timestamps = false;
    params.single_segment = false;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.token_timestamps = false;
    params.language = "ja";
    params.detect_language = false;
    params.abort_callback = abort_after_deadline;
    params.abort_callback_user_data = &deadline;
    params.vad = true;
    params.vad_model_path = engine->vad_model_path.c_str();
    params.vad_params.threshold = engine->vad_threshold;
    params.vad_params.min_speech_duration_ms = engine->vad_min_speech_duration_ms;
    params.vad_params.min_silence_duration_ms = engine->vad_min_silence_duration_ms;
    params.vad_params.max_speech_duration_s = engine->vad_max_speech_duration_seconds;
    params.vad_params.speech_pad_ms = engine->vad_speech_padding_ms;
    params.vad_params.samples_overlap = engine->vad_samples_overlap_seconds;

    if (whisper_full(
            engine->whisper,
            params,
            samples.data(),
            static_cast<int>(samples.size())) != 0) {
        log_error("Whisper inference failed or timed out");
        return empty_transcript_array(env);
    }

    jclass segment_class = env->FindClass("chimahon/audio/NativeTranscriptSegment");
    if (segment_class == nullptr) {
        return nullptr;
    }
    jmethodID constructor = env->GetMethodID(segment_class, "<init>", "(Ljava/lang/String;JJ)V");
    if (constructor == nullptr) {
        return nullptr;
    }

    const int segment_count = whisper_full_n_segments(engine->whisper);
    jobjectArray result = env->NewObjectArray(segment_count, segment_class, nullptr);
    if (result == nullptr) {
        return nullptr;
    }

    for (int index = 0; index < segment_count; ++index) {
        const char * text = whisper_full_get_segment_text(engine->whisper, index);
        jstring java_text = env->NewStringUTF(text == nullptr ? "" : text);
        // Built-in VAD maps filtered-audio timestamps back to the original input timeline.
        const jlong start_millis = whisper_full_get_segment_t0(engine->whisper, index) * 10L;
        const jlong end_millis = whisper_full_get_segment_t1(engine->whisper, index) * 10L;
        jobject segment = env->NewObject(
            segment_class,
            constructor,
            java_text,
            start_millis,
            end_millis);
        env->SetObjectArrayElement(result, index, segment);
        env->DeleteLocalRef(segment);
        env->DeleteLocalRef(java_text);
    }
    return result;
}
