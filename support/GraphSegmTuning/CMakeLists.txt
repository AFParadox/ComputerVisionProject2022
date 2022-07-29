cmake_minimum_required(VERSION 3.0.0)
project(GraphSegmTuning VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(
    tuneGraphSegm
    headers/tunableGraphSegm.hpp
    src/tunableGraphSegm.cpp
)
target_include_directories(tuneGraphSegm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")

add_executable(projSegm src/main.cpp)
target_link_libraries(projSegm ${OpenCV_LIBS} tuneGraphSegm)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
