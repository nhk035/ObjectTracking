cmake_minimum_required(VERSION 3.10)
project(Tracking)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

AUX_SOURCE_DIRECTORY(./maxflow SOURCE_FILES)

add_executable(Tracking main.cpp ${SOURCE_FILES} common.h prepare.cpp prepare.h)
target_link_libraries(Tracking ${OpenCV_LIBS} pthread)