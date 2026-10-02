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
        message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

add_message_files(
        DIRECTORY msg
        FILES
        CustomMsg.msg   # 与 /livox/lidar (livox_ros_driver::CustomMsg) 逐字段一致
        CustomPoint.msg
        PoseFusionDiag.msg   # pose_fusion_shadow_node 的诊断输出
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp nav_msgs sensor_msgs std_msgs geometry_msgs tf2_ros message_runtime
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

# 旁路验证：/lidar_pose 锚点 + 机体速度前推，只发独立话题，不接入控制通路
add_executable(pose_fusion_shadow_node
        src/PoseFusionShadowNode.cpp
        include/hand_lio/PoseFusion.h
)

add_dependencies(pose_fusion_shadow_node
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(pose_fusion_shadow_node
        ${catkin_LIBRARIES}
        Eigen3::Eigen
)

install(TARGETS hand_lio_node pose_fusion_shadow_node
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        OPTIONAL
)
