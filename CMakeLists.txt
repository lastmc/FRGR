cmake_minimum_required(VERSION 3.18...3.26)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(FRGR LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

get_cmake_property(IS_MULTI_CONFIG GENERATOR_IS_MULTI_CONFIG)
if (IS_MULTI_CONFIG)
    foreach (config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${config}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${config}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib/${config}")
        set(CMAKE_PDB_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib/${config}")
    endforeach ()
endif ()

find_package(OpenCV CONFIG REQUIRED)

add_subdirectory(ext/cxxopts)

set(LUISA_COMPUTE_ENABLE_DX OFF)
set(LUISA_COMPUTE_ENABLE_METAL OFF)
set(LUISA_COMPUTE_ENABLE_CPU OFF)
set(LUISA_COMPUTE_ENABLE_REMOTE OFF)
set(LUISA_COMPUTE_ENABLE_RUST OFF)
set(LUISA_COMPUTE_ENABLE_VULKAN OFF)
add_subdirectory(ext/compute)

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif ()

add_library(common INTERFACE)
target_link_libraries(common INTERFACE
        ${OpenCV_LIBS}
        cxxopts
        luisa::compute
)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(common INTERFACE "/utf-8")
endif ()

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE common)
