cmake_minimum_required(VERSION 3.0.0)
project(projectSegmentation VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(
    tuneBil
    headers/tunableBilateral.hpp
    src/tunableBilateral.cpp
)
target_include_directories(tuneBil PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")

add_executable(projSegm src/main.cpp)
target_link_libraries(projSegm ${OpenCV_LIBS} tuneBil)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
