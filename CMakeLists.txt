cmake_minimum_required(VERSION 3.0.0)
project(ComputerVisionProj VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(
    HandSeg
    headers/handSegmentation.hpp
    src/handSegmentation.cpp
)
target_include_directories(HandSeg PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")

add_library(
    HandLoc
    headers/handLocalization.hpp
    src/handLocalization.cpp
)
target_include_directories(HandLoc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")

add_executable(ComputerVisionProj src/main.cpp)
target_link_libraries(ComputerVisionProj ${OpenCV_LIBS} HandSeg HandLoc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
