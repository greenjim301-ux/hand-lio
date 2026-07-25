cmake_minimum_required(VERSION 3.10)
project(hand_lio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        nav_msgs
        sensor_msgs
        std_msgs
        geometry_msgs
        tf2_ros
        pcl_ros
        pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp nav_msgs sensor_msgs std_msgs geometry_msgs tf2_ros
        DEPENDS EIGEN3 PCL
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
)

add_executable(hand_lio_node
        src/main.cpp
        src/HandLioNode.cpp
        include/hand_lio/HandLioNode.h
)

add_dependencies(hand_lio_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(hand_lio_node
        ${catkin_LIBRARIES}
        Eigen3::Eigen
        ${PCL_LIBRARIES}
)

install(TARGETS hand_lio_node
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        OPTIONAL
)
