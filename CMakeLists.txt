cmake_minimum_required(VERSION 3.16)

project(inference_engine
    VERSION 0.1.0
    DESCRIPTION "A from-scratch LLM inference engine in C++ - no ML frameworks."
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(IE_BUILD_TESTS        "Build unit tests"                          ON)
option(IE_ENABLE_NATIVE      "Build with -march=native (AVX2/FMA)"       ON)
option(IE_ENABLE_OPENMP      "Link OpenMP for multithreading if found"   ON)
option(IE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"         OFF)
option(IE_SANITIZE           "Enable Address+UB sanitizers (Debug)"      OFF)

# ---------------------------------------------------------------------------
# Language standard / defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd / editor tooling

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type set; defaulting to Release.")
endif()

# ---------------------------------------------------------------------------
# Core library (shared by the CLI and the tests)
# ---------------------------------------------------------------------------
add_library(ie_core
    src/model.cpp
    src/tokenizer.cpp
    src/forward.cpp
    src/gemm.cpp
    src/quant.cpp)
target_include_directories(ie_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Warnings
if(MSVC)
    target_compile_options(ie_core PRIVATE /W4)
else()
    target_compile_options(ie_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(IE_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(ie_core PRIVATE /WX)
    else()
        target_compile_options(ie_core PRIVATE -Werror)
    endif()
endif()

# -march=native (guarded by a feature check so non-x86 / odd toolchains are OK)
if(IE_ENABLE_NATIVE AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" IE_HAS_MARCH_NATIVE)
    if(IE_HAS_MARCH_NATIVE)
        target_compile_options(ie_core PUBLIC -march=native)
    endif()
endif()

# OpenMP (optional; absence is not an error)
if(IE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ie_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(ie_core PUBLIC IE_HAVE_OPENMP=1)
        message(STATUS "OpenMP found; multithreading enabled.")
    else()
        message(STATUS "OpenMP not found; building single-threaded.")
    endif()
endif()

# Sanitizers (handy for the correctness-heavy forward-pass work)
if(IE_SANITIZE AND NOT MSVC)
    target_compile_options(ie_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(ie_core PUBLIC -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(inference-engine src/main.cpp)
target_link_libraries(inference-engine PRIVATE ie_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(IE_BUILD_TESTS)
    enable_testing()
    add_executable(test_gemm tests/test_gemm.cpp)
    target_link_libraries(test_gemm PRIVATE ie_core)
    add_test(NAME gemm_correctness COMMAND test_gemm)

    # Loader + tokenizer-decode round trip. Runs self-contained (hermetic): it
    # writes tiny fixtures itself, so no Python/numpy is needed at test time.
    # For the cross-language check, run tests/make_fixtures.py and then
    # `test_loader --fixture <dir>` (see README).
    add_executable(test_loader tests/test_loader.cpp)
    target_link_libraries(test_loader PRIVATE ie_core)
    add_test(NAME loader_roundtrip COMMAND test_loader)
endif()
