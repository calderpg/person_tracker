cmake_minimum_required(VERSION 2.8.3)
project(person_tracker)

## Find dependencies
find_package(catkin REQUIRED COMPONENTS roscpp resource_retriever cv_bridge image_geometry image_transport geometry_msgs message_generation sensor_msgs std_msgs tf)
find_package(OpenCV REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages
add_message_files(DIRECTORY msg FILES SkeletonTrack.msg TrackedPerson.msg TrackerState.msg)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)

###################################
## catkin specific configuration ##
###################################

catkin_package(CATKIN_DEPENDS roscpp resource_retriever cv_bridge image_geometry image_transport geometry_msgs message_runtime sensor_msgs std_msgs tf DEPENDS system_lib opencv rt)

###########
## Build ##
###########

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-g")

## Tracker executable
add_executable(opencv_tracker src/people_tracker.cpp)
add_dependencies(opencv_tracker ${catkin_EXPORTED_TARGETS})
target_link_libraries(opencv_tracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} rt)
