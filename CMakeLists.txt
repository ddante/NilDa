cmake_minimum_required(VERSION 3.10)

project(NilDa VERSION 0.1)

######################################
option(NILDA_DEBUG_BUILD  "Enable debug build" OFF)

set(NILDA_PRECISION_OPTION "SP")

option(NILDA_RANDOM_SEED "Random seed for random number generator" ON)
######################################

if(${NILDA_DEBUG_BUILD})
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DND_DEBUG_CHECKS)
else()
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${NILDA_PRECISION_OPTION} MATCHES "SP")
    add_definitions(-DND_SP)
elseif(${NILDA_PRECISION_OPTION} MATCHES "DP")
    add_definitions(-DND_DP)
elseif(${NILDA_PRECISION_OPTION} MATCHES "LP")
    add_definitions(-DND_LP)
else()
    message(FATAL_ERROR "${NILDA_PRECISION_OPTION} not supported")
endif()

if(${NILDA_RANDOM_SEED})
  add_definitions(-DND_RAND)
endif()

message("-- Compiling mode:   ${CMAKE_BUILD_TYPE}")
message("-- Precision option: ${NILDA_PRECISION_OPTION}")
message("-- Random seed:      ${NILDA_RANDOM_SEED}")

set(CMAKE_CXX_FLAGS "-Wall -Wall")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS "-Wall -Wpedantic")

#set(CMAKE_CXX_FLAGS "-fp-trap=common")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("${EIGEN3_INCLUDE_DIR}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("${OpenCV_INCLUDE_DIRS}")

#if(NOT TARGET spdlog)
#    find_package(spdlog REQUIRED)
#endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message("${OpenMP_CXX}")
endif()

add_subdirectory(sources)

add_subdirectory(tests)

add_subdirectory(unitTests)

add_subdirectory(examples)
