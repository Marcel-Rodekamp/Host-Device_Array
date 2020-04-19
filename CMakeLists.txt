# Set CMake minimum version 
cmake_minimum_required(VERSION 3.17)

# Create Project
project(HDA
        VERSION 0.1
        DESCRIPTION "A STL like cuda enabled array class"
        LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Enable CUDA 10.2
find_package(CUDA 10.2 REQUIRED)
enable_language(CUDA)

set(CMAKE_CXX_COMPILER nvcc)

# Library ===============================================================================

add_library(hda STATIC
                ${PROJECT_SOURCE_DIR}/src/array.h
                ${PROJECT_SOURCE_DIR}/src/util/gpu.h
)

set_target_properties(hda PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(hda PROPERTIES LINKER_LANGUAGE CXX)                    
set_target_properties(hda PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Tests =================================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/tests)

# create executables  
cuda_add_executable(test_elementAccess ${PROJECT_SOURCE_DIR}/tests/test_elementAccess.cu) 
cuda_add_executable(test_initialization ${PROJECT_SOURCE_DIR}/tests/test_initialization.cu)
cuda_add_executable(test_sync ${PROJECT_SOURCE_DIR}/tests/test_sync.cu)
