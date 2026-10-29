cmake_minimum_required(VERSION 3.22.1)

project(ramzes_native LANGUAGES C CXX)

include(FetchContent)

# ------------------------------------------------------------
# Настройки llama.cpp для Android
# ------------------------------------------------------------

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

# Собираем переносимую CPU-версию для Android NDK.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

# Дополнительные необязательные компоненты отключены.
set(GGML_RPC OFF CACHE BOOL "" FORCE)

# ------------------------------------------------------------
# Фиксированная версия llama.cpp
# ------------------------------------------------------------

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b11906
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(llama_cpp)

# ------------------------------------------------------------
# Нативная библиотека приложения
# ------------------------------------------------------------

add_library(
    ramzes_native
    SHARED
    native-lib.cpp
)

find_library(
    ANDROID_LOG_LIBRARY
    log
)

if(NOT ANDROID_LOG_LIBRARY)
    message(FATAL_ERROR "Android log library not found")
endif()

target_compile_features(
    ramzes_native
    PRIVATE
    cxx_std_17
)

target_compile_options(
    ramzes_native
    PRIVATE
    -Wall
    -Wextra
)

target_link_libraries(
    ramzes_native
    PRIVATE
    llama
    ${ANDROID_LOG_LIBRARY}
)
