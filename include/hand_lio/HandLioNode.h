/*
 * hand_lio_node
 *
 * 订阅 HandBot-S1 的 /livox/lidar (原始点云) 和 /latest_imu_odom (200Hz 高频位姿,
 * map->imu，见 hand-topic.csv)，对每个点按其真实采集时刻在位姿缓冲区中插值出
 * map 系下的位姿，直接把该点变换到 map 系——去畸变和转全局系一步完成。
 * 200Hz 的位姿流样本间隔仅 5ms，插值去畸变的精度接近 IMU 前向传播。
 *
 * 丢点问题的解法——延迟处理（deferred processing）：/latest_imu_odom 天然比
 * /livox/lidar 晚到（实测滞后 2~20ms），若在 lidar 回调里立即处理，帧尾最后
 * 几毫秒的点会因插值区间覆盖不到而被丢弃。所以 lidar 帧先进待处理队列，
 * 每收到新 odom 就检查队首帧是否已被完整覆盖（odom_buf_.back().t >= 帧尾时刻），
 * 覆盖了才处理发布。代价只是滞后量级（~25ms）的额外延迟，换来一个点都不丢、
 * 且所有点都用真插值（无外推）。odom 停更时由 pending_timeout_sec 兜底：
 * 超时的帧按当时覆盖范围尽力处理，未覆盖的尾部点丢弃，不无限积压。
 *
 * 曾试过的桥接方案（用 Elevator-LIO 的去畸变点云 + 系间修正搬到 map 系，
 * 见 git 历史）已放弃：它引入第二套状态估计的全部失效面和算力开销，点云还
 * 叠了 elio 漂移 + 配对 + 平滑三层误差；在滞后仅 2~20ms 的前提下，延迟处理
 * 更简单、点云-里程计一致性也更好（同一个源，天然零系差）。
 */
#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

    struct PendingFrame {
        CustomMsgConstPtr msg;
        double end_time = 0.0;        // 帧尾最后一个点的采集时刻
        ros::Time arrival_wall_time;  // 到达本节点的墙钟时刻，用于超时兜底
    };

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void lidarCallback(const CustomMsgConstPtr& msg);

    // 缓存 navibot 发来的虚拟障碍点（世界系，latched，只在用户改禁行区时更新）。
    void virtualObstacleCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    // 把缓存里"从 sensor_pos 看得见、且在量程内"的虚拟障碍点追加到本帧点云。
    // 见 .cpp 里的实现说明——为什么要可见性剔除、为什么要按距离加密。
    void appendVirtualObstacles(const Eigen::Vector3d& sensor_pos, pcl::PointCloud<pcl::PointXYZI>& cloud) const;

    // 处理待处理队列：队首帧已被 odom 覆盖到帧尾、或等待超时，则出队处理。
    // 调用者需持有 odom_mutex_。
    void drainPendingFrames();

    // 去畸变 + 转 map 系 + 发布。调用者需持有 odom_mutex_。
    void processFrame(const PendingFrame& frame);

    // 在 [odom_buf_.front().t, odom_buf_.back().t] 范围内对位姿做线性/球面插值。
    // 调用者需持有 odom_mutex_。
    bool interpolatePose(double t, PoseSample& out) const;

    // 参照 Elevator-LIO::publish_body_odometry：world_T_body = world_T_imu * imu_T_lidar * lidar_T_body。
    // 这里 world_T_imu 直接取自 /latest_imu_odom 这一条消息本身（不需要插值），
    // 跟去畸变点云那条路径（需要缓冲区插值）是相互独立的两条支路。
    void publishVehicleOdom(const nav_msgs::Odometry::ConstPtr& msg);

    ros::Subscriber odom_sub_;
    ros::Subscriber lidar_sub_;
    ros::Subscriber virtual_obstacle_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher vehicle_odom_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    mutable std::mutex odom_mutex_;
    std::deque<PoseSample> odom_buf_;
    std::deque<PendingFrame> pending_frames_;

    // 虚拟障碍缓存单独一把锁：它的写入在 navibot 那条回调上（很少发生），读取在
    // 点云处理路径上，跟 odom_mutex_ 保护的东西没有任何交集，共用一把锁只会让
    // 10Hz 的点云路径白白跟 200Hz 的 odom 回调抢锁。
    // 位置 + 朝外法向, 法向用来做背面剔除(见 .cpp 的 appendVirtualObstacles)。
    struct VirtualObstaclePoint {
        Eigen::Vector3f p = Eigen::Vector3f::Zero();
        Eigen::Vector3f n = Eigen::Vector3f::Zero();  // 零向量 = 没给法向, 不剔除
    };
    mutable std::mutex virtual_obstacle_mutex_;
    std::vector<VirtualObstaclePoint> virtual_obstacle_pts_;

    // ---- 外参：lidar -> imu ----
    // Mid-360 内置 IMU 相对雷达原点的出厂固定偏移，取自 Elevator-LIO yaml/sensors/livox.yaml，
    // 与 hku-mars/FAST_LIO 官方 config/mid360.yaml 一致。仅在 /latest_imu_odom 用的是
    // Mid-360 内置 IMU 时成立，见 config/hand_lio.yaml 注释。
    Eigen::Matrix3d imu_R_lidar_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d imu_t_lidar_ = Eigen::Vector3d(-0.011, -0.02329, 0.04412);

    // ---- 外参：lidar -> body（Go2 机体中心）----
    // 跟 imu_R_lidar_ 不同，这是设备绑在 Go2 背上的机械安装关系，装好后必须自己
    // 标定，占位单位阵仅供联调！键名跟 Elevator-LIO yaml/sensors/livox.yaml 的
    // lidar_R_body/lidar_t_body 保持一致。
    Eigen::Matrix3d lidar_R_body_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lidar_t_body_ = Eigen::Vector3d::Zero();

    // ---- 参数 ----
    double blind_ = 0.8;  // 跟随 Elevator-LIO 同款 Mid-360 实测值，见 config/hand_lio.yaml 注释
    int point_filter_num_ = 1;
    double buffer_horizon_sec_ = 2.0;
    double pose_cov_reject_thresh_ = 0.99;
    double pending_timeout_sec_ = 0.5;  // 队首帧等待 odom 覆盖的最长墙钟时长，超时按当前覆盖尽力处理
    std::string world_frame_id_ = "world";
    std::string vehicle_frame_id_ = "body";

    // ---- 虚拟障碍注入 ----
    // navibot 把用户在 2D 图上圈的禁行区采样成世界系点云发过来（见它的
    // backend/app/virtual_obstacles.py），这里混进输出点云，让 SCAN-Planner 的
    // 局部避障也看得见——本节点不认识"禁行区"这个概念，只负责把别人给的障碍点
    // 混进来。为什么非得混在同一条消息里（而不是另发一路）见 .cpp 的说明。
    bool enable_virtual_obstacles_ = true;
    std::string virtual_obstacle_topic_ = "/navibot/virtual_obstacles";
    double virtual_obstacle_max_range_ = 5.0;   // 跟 SCAN-Planner grid_map/max_ray_length 对齐
    double virtual_obstacle_density_gain_ = 12; // 1m 处每个可见点复制几份，见 .cpp
    int virtual_obstacle_max_copies_ = 32;
    int virtual_obstacle_max_points_ = 40000;   // 每帧注入点数上限，兜底

    // Livox tag/line 质量过滤：跟 Elevator-LIO 一样无条件开启，不留开关。
    int n_scans_ = 4;  // Mid-360 专属: livox_ros_driver2 src/comm/comm.h kLineNumberMid360 = 4（line 取值 0~3）
    int tag_mask_ = 0x30;
};

}  // namespace hand_lio
