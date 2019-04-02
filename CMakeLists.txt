cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx2 -mfma -fopenmp")

find_package(OpenCV 3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest)

include_directories(include/feature_tracker
  ${YAML_CPP_INCLUDE_DIR}
  )

add_library(feature_tracker
    src/feature_tracker.cpp
)
target_link_libraries(feature_tracker
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  )

if (${GTEST_FOUND})
  add_definitions(-DFEATURE_TRACKER_DIR="${CMAKE_CURRENT_LIST_DIR}")
  include_directories(include ${GTEST_INCLUDE_DIRS})
  add_executable(visual_test
    src/test/visual_test.cpp
    )
  target_link_libraries(visual_test
    feature_tracker
    gtest_main
    gtest
    ${GTEST_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    )
endif()
