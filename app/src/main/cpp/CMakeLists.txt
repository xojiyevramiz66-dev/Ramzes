
cmake_minimum_required(VERSION 3.22.1)

project(ramzes_native LANGUAGES C CXX)

include(FetchContent)

# Настройки llama.cpp для Android
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

# Настройки GGML
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Загружаем llama.cpp
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(llama_cpp)

# Нативная библиотека RAMZES AI
add_library(
    ramzes_native
    SHARED
    native-lib.cpp
)

# Системная библиотека Android Log
find_library(
    android-log
    log
)

# Подключаем llama.cpp и Android Log
target_link_libraries(
    ramzes_native
    PRIVATE
    llama
    ${android-log}
)
