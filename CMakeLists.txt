cmake_minimum_required(VERSION 3.17)

project(ElectronsAndProtons CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenGL REQUIRED)

add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    ./ElectronsAndProtons/main.cpp
    ./ElectronsAndProtons/kernel.cu
    ./ElectronsAndProtons/ElectricField.cpp
    ./ElectronsAndProtons/glad_functions.cpp
    ./ElectronsAndProtons/glfw_functions.cpp
    ./ElectronsAndProtons/interactions.cpp
    ./ElectronsAndProtons/shader_loader.cpp
    ./ElectronsAndProtons/glad.c
)


target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronsAndProtons/
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes/
    ${CMAKE_CURRENT_SOURCE_DIR}/Libraries/)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: >
)

add_subdirectory( external/glfw )

target_link_libraries( ${CMAKE_PROJECT_NAME} glfw )

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()