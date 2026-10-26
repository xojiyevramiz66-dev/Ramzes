
cmake_minimum_required(VERSION 3.22.1)

project(ramzes_native LANGUAGES C CXX)

include(FetchContent)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(llama_cpp)

add_library(
    ramzes_native
    SHARED
    native-lib.cpp
)

find_library(android-log log)

target_link_libraries(
    ramzes_native
    PRIVATE
    llama
    ${android-log}
)

target_compile_features(ramzes_native PRIVATE cxx_std_17)
