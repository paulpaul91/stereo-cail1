cmake_minimum_required(VERSION 2.4.8)
project(opencv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(OpenCV_DIR "/usr/local/opencv3.1.0")
find_package( OpenCV  REQUIRED )

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(opencv ${SOURCE_FILES})
target_link_libraries(opencv ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS})
add_executable(test_point test_point.cpp)
target_link_libraries(test_point ${OpenCV_LIBS})
add_executable(people detect.cpp)
target_link_libraries(people ${OpenCV_LIBS})
