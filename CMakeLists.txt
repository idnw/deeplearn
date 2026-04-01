cmake_minimum_required(VERSION 3.10)
project(deeplearn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# SIMD detection
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_c_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
    message(STATUS "AVX2 support enabled")
elseif(COMPILER_SUPPORTS_SSE2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2")
    message(STATUS "SSE2 support enabled")
else()
    message(STATUS "No SIMD support detected, using scalar fallback")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(DL_SOURCES
    src/dl_tensor.c
    src/dl_ops.c
    src/dl_autograd.c
    src/dl_nn.c
    src/dl_transformer.c
    src/dl_optimizer.c
    src/dl_tokenizer.c
    src/dl_dataloader.c
    src/dl_serialize.c
)

# Static library
add_library(deeplearn STATIC ${DL_SOURCES})
target_link_libraries(deeplearn m)

# Test executable
add_executable(test_tensor tests/test_tensor.c)
target_link_libraries(test_tensor deeplearn m)

# Training example
add_executable(train_gpt2 examples/train_gpt2.c)
target_link_libraries(train_gpt2 deeplearn m)

# Inference example
add_executable(inference examples/inference.c)
target_link_libraries(inference deeplearn m)

# CTest
enable_testing()
add_test(NAME tensor_tests COMMAND test_tensor)
