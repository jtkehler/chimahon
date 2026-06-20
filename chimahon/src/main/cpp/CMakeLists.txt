cmake_minimum_required(VERSION 3.22.1)

project(hoshidicts_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(hoshidicts)

add_library(hoshidicts_jni SHARED
    hoshidicts_jni.cpp
)

target_link_libraries(hoshidicts_jni PRIVATE
    hoshidicts
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static third-party libraries" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "Build whisper.cpp tests" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "Build whisper.cpp examples" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "Build whisper.cpp server" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Use OpenMP" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Use host-specific instructions" FORCE)

add_subdirectory(whisper.cpp EXCLUDE_FROM_ALL)

# Whisper inference is too slow to be usable when the Android debug build leaves
# the native targets unoptimized. Keep debug symbols, but optimize the inference
# libraries in every build type.
foreach(target whisper ggml ggml-base ggml-cpu)
    target_compile_options(${target} PRIVATE -O3)
endforeach()

add_library(sentence_audio_jni SHARED
    sentence_audio_jni.cpp
)

find_library(log-lib log)

target_link_libraries(sentence_audio_jni PRIVATE
    whisper
    ${log-lib}
)
