/*
 * hand_lio_node
 *
 * 订阅 HandBot-S1 的 /livox/lidar (原始点云) 和 /latest_imu_odom (200Hz 高频位姿，
 * map->imu，见 hand-topic.csv)，仿照 Elevator-LIO 的去畸变思路（LidarPipeline::undistortFrame /
 * IMUProcess::get_imu_pose_from_t1_to_t2）：对每个点按其真实采集时刻在位姿缓冲区中插值出
 * world 系下的位姿，直接把该点变换到 world 系——去畸变和转 world 系一步完成。
 *
 * 与 Elevator-LIO 的差异：Elevator-LIO 还需要把去畸变结果保留在 lidar 系（供 IESKF
 * 点面匹配、ikd-Tree 建图等下游使用），所以它是"转 world -> 转回帧末 lidar 系"两步。
 * 这里只需要 world 系输出，且 /latest_imu_odom 已经是 HandBot-S1 自己融合好的位姿
 * （不是原始 IMU 加速度/角速度), 没有后续再匹配修正的步骤，所以没有必要做那次回退，
 * 直接用每个点自身时刻的插值位姿一步转到 world 系，数学上是等价且更直接的。
 */
#pragma once

#include <deque>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>

#include "hand_lio/CustomMsg.h"

namespace hand_lio {

class HandLioNode {
public:
    HandLioNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);

private:
    struct PoseSample {
        double t = 0.0;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        double cov0 = 0.0;  // covariance[0]：定位方差，见 hand-topic.csv（<0.1 较高精度，0.99 定位失败）
    };

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void lidarCallback(const CustomMsgConstPtr& msg);

    // 在 [odom_buf_.front().t, odom_buf_.back().t] 范围内对位姿做线性/球面插值。
    // 调用者需持有 odom_mutex_。
    bool interpolatePose(double t, PoseSample& out) const;

    // 参照 Elevator-LIO::publish_body_odometry：world_T_body = world_T_imu * imu_T_lidar * lidar_T_body。
    // 这里 world_T_imu 直接取自 /latest_imu_odom 这一条消息本身（不需要插值），
    // 跟去畸变点云那条路径（需要缓冲区插值）是相互独立的两条支路。
    void publishVehicleOdom(const nav_msgs::Odometry::ConstPtr& msg);

    ros::Subscriber odom_sub_;
    ros::Subscriber lidar_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher vehicle_odom_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    mutable std::mutex odom_mutex_;
    std::deque<PoseSample> odom_buf_;

    // ---- 外参：lidar -> imu ----
    // Mid-360 内置 IMU 相对雷达原点的出厂固定偏移，取自 Elevator-LIO yaml/sensors/livox.yaml，
    // 与 hku-mars/FAST_LIO 官方 config/mid360.yaml 一致。仅在 /latest_imu_odom 用的是
    // Mid-360 内置 IMU 时成立，见 config/hand_lio.yaml 注释。
    Eigen::Matrix3d imu_R_lidar_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d imu_t_lidar_ = Eigen::Vector3d(-0.011, -0.02329, 0.04412);

    // ---- 外参：lidar -> body（小车/机体中心）----
    // 跟 imu_R_lidar_ 不同，这是 HandBot-S1 底盘特有的机械安装关系，官方资料/Elevator-LIO
    // 都给不出，占位单位阵，务必自己标定后再用于下游导航！键名跟 Elevator-LIO
    // yaml/sensors/livox.yaml 的 lidar_R_body/lidar_t_body 保持一致。
    Eigen::Matrix3d lidar_R_body_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lidar_t_body_ = Eigen::Vector3d::Zero();

    // ---- 参数 ----
    double blind_ = 0.8;  // 跟随 Elevator-LIO 同款 Mid-360 实测值，见 config/hand_lio.yaml 注释
    int point_filter_num_ = 1;
    double buffer_horizon_sec_ = 2.0;
    double pose_cov_reject_thresh_ = 0.99;
    std::string world_frame_id_ = "world";
    std::string vehicle_frame_id_ = "body";

    // Livox tag/line 质量过滤：跟 Elevator-LIO 一样无条件开启，不留开关。
    int n_scans_ = 4;  // Mid-360 专属: livox_ros_driver2 src/comm/comm.h kLineNumberMid360 = 4（line 取值 0~3）
    int tag_mask_ = 0x30;
};

}  // namespace hand_lio
