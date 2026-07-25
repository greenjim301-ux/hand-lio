/*
 * hand_lio_node
 *
 * Elevator-LIO 点云的全局系桥接（map <- elio-world）。
 *
 * 背景：SCAN-Planner 需要全局重定位系（/latest_imu_odom 所在的 map 系）下的
 * 去畸变点云，但 /latest_imu_odom 天然滞后于 /livox/lidar，直接用它给原始点
 * 逐点插值会持续丢掉帧尾的点（插值区间覆盖不到）。而 Elevator-LIO 内部用
 * 自己的 IMU 前向传播做去畸变，不受该滞后影响，点不丢、质量更好——只是
 * 它输出在自己的局部 world 系（启动位置为原点，随时间漂移）。
 *
 * 桥接思路（标准 map->odom->base_link 分层）：
 *   1. 缓存 /latest_imu_odom（map_T_imu，滞后但全局无漂移）和 Elevator-LIO
 *      的 odom_imu（elio_T_imu，及时但漂移）。两者是同一颗 Mid-360 内置 IMU
 *      的位姿，可直接按时间戳配对，无需额外外参。
 *   2. 在两条流共同覆盖的最新时刻 t 上插值配对，得系间修正
 *      T_map_elio = map_T_imu(t) * elio_T_imu(t)^-1。
 *      它只随 Elevator-LIO 的漂移变化，是慢变量——所以 /latest_imu_odom 的
 *      滞后在这里无害：用"稍旧"的修正量转换刚到的点云，误差只是滞后时段内
 *      的漂移增量。对慢变量做指数平滑压噪；检测到大幅跳变（全局重定位修正）
 *      时直接跟上不平滑。
 *   3. 收到 Elevator-LIO 的 world 系点云时，整体乘 T_map_elio 转到 map 系发布。
 *
 * /hand_lio/odom_vehicle 支路与桥接无关：仍直接由每条 /latest_imu_odom 合成
 * （参照 Elevator-LIO::publish_body_odometry 的外参链）。
 *
 * 前提：Elevator-LIO 与 /latest_imu_odom 的时间戳来自同一时钟源（同一台主机
 * 的 ROS time / 同一雷达的硬件时间戳），否则"同一时刻"配对本身不成立。
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

    void mapOdomCallback(const nav_msgs::Odometry::ConstPtr& msg);   // /latest_imu_odom
    void elioOdomCallback(const nav_msgs::Odometry::ConstPtr& msg);  // Elevator-LIO odom_imu
    void elioCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    // 追加样本并按 horizon 裁剪；时间戳回退（播包重播/节点重启，坐标系可能已变）
    // 时清空缓冲并返回 true，调用者据此作废当前修正量。调用者需持有 buf_mutex_。
    static bool pushSample(std::deque<PoseSample>& buf, const nav_msgs::Odometry::ConstPtr& msg, double horizon,
                           const char* name);

    // 在 [buf.front().t, buf.back().t] 范围内对位姿做线性/球面插值。调用者需持有 buf_mutex_。
    static bool interpolatePose(const std::deque<PoseSample>& buf, double t, PoseSample& out);

    // 在两条位姿流共同覆盖的最新时刻配对，更新 T_map_elio。调用者需持有 buf_mutex_。
    void tryUpdateCorrection();

    // 参照 Elevator-LIO::publish_body_odometry：world_T_body = world_T_imu * imu_T_lidar * lidar_T_body。
    // 这里 world_T_imu 直接取自 /latest_imu_odom 这一条消息本身（不需要插值），
    // 跟点云桥接那条路径是相互独立的两条支路。
    void publishVehicleOdom(const nav_msgs::Odometry::ConstPtr& msg);

    ros::Subscriber map_odom_sub_;
    ros::Subscriber elio_odom_sub_;
    ros::Subscriber elio_cloud_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher vehicle_odom_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    mutable std::mutex buf_mutex_;
    std::deque<PoseSample> map_odom_buf_;   // /latest_imu_odom，map 系
    std::deque<PoseSample> elio_odom_buf_;  // Elevator-LIO odom_imu，elio-world 系

    // ---- 系间修正 T_map_elio（由 buf_mutex_ 保护）----
    bool correction_valid_ = false;
    double correction_pair_t_ = 0.0;  // 最近一次成功配对的时刻（消息时间）
    Eigen::Quaterniond corr_q_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d corr_p_ = Eigen::Vector3d::Zero();

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
    double buffer_horizon_sec_ = 2.0;
    double pose_cov_reject_thresh_ = 0.99;
    double correction_smooth_tau_ = 0.5;      // 修正量指数平滑时间常数 [s]
    double correction_jump_pos_ = 0.5;        // 平移跳变阈值 [m]，超过视为全局重定位修正，直接跟上
    double correction_jump_ang_deg_ = 10.0;   // 旋转跳变阈值 [deg]，同上
    double correction_timeout_sec_ = 1.0;     // 修正量过期时长 [s]，过期则丢弃点云
    std::string world_frame_id_ = "world";
    std::string vehicle_frame_id_ = "body";
};

}  // namespace hand_lio
