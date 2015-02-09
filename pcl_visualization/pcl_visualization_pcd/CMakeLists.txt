cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cloud_viewer)

find_package(PCL 1.2 REQUIRED)
message("PCL_LIBRARIES - ${PCL_LIBRARIES}")


include_directories(${PCL_INCLUDE_DIRS})
message(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (cloud_viewer cloud_viewer.cpp)

target_link_libraries (cloud_viewer ${PCL_LIBRARIES})

