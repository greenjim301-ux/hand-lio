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
 * 等队首帧被完整覆盖（odom_buf_.back().t >= 帧尾时刻）了才处理发布。代价只是
 * 滞后量级（~25ms）的额外延迟，换来一个点都不丢、且所有点都用真插值（无外推）。
 * odom 停更时由 pending_timeout_sec 兜底：超时的帧按当时覆盖范围尽力处理，
 * 未覆盖的尾部点丢弃，不无限积压。
 *
 * 线程：/latest_imu_odom 走自己的回调队列 + 专用 spinner 线程，回调里只做
 * "发 /hand_lio/odom_vehicle + 追加缓冲区"，不碰点云。点云处理留在主线程
 * （lidar 回调 + 5ms 定时器检查队首帧），在锁里只拷出这一帧用得到的那段位姿，
 * 锁外做去畸变。以前两者共用一个单线程 spin、点云还是在 odom 回调里处理的，
 * 处理一帧期间 200Hz 的位姿全堆在队列里，出来时成批发出（实测 LubanCat-4 上
 * odom_vehicle 比 /latest_imu_odom 晚 23ms 中位、53ms 最大，4~10 条一批），
 * 下游 100Hz 的控制器实际只拿到 ~35Hz 的新位姿。
 *
 * 位姿来源（pose_source）：
 *   stream  /hand_lio/odom_vehicle 直接由 /latest_imu_odom 合成（原来的做法）。
 *   fused   /lidar_pose 做锚点 + 机体速度前推（PoseFusion，输入侧见 FusionFrontend）。
 *           /latest_imu_odom 的“突变”其实就是 /lidar_pose 到达时的修正，坏的是两次
 *           修正之间的前推（楼梯上 z 向能偏到 -1~-1.5 m/s，/lidar_pose 延迟超过约 1 s
 *           时干脆不前推）。融合位姿和高频流之差记成一个世界系修正量
 *               T_corr(t) = T_fused_body(t) * T_stream_body(t)^-1
 *           同一个修正量施加到 odom_vehicle、TF、/hand_lio/odom_sensor（栅格地图的射线
 *           原点）和去畸变后的点云上——只换 odom_vehicle 的话，楼梯上机器人和障碍物的
 *           高度会差出 0.5~0.9 m（规划器“起点在障碍物里”就是这么来的）。扫描内部的
 *           相对运动仍用高频流插值（100ms 内它是准的），整帧再按修正量对齐。
 *           还没有锚点、或 /lidar_pose 断流超过 fusion/lidar_timeout 时，修正量为单位
 *           变换，行为跟 stream 完全一样。
 * 两种模式下都另发 /hand_lio/odom_vehicle_stream（高频流合成的机体位姿），录包对比用。
 *
 * 曾试过的桥接方案（用 Elevator-LIO 的去畸变点云 + 系间修正搬到 map 系，
 * 见 git 历史）已放弃：它引入第二套状态估计的全部失效面和算力开销，点云还
 * 叠了 elio 漂移 + 配对 + 平滑三层误差；在滞后仅 2~20ms 的前提下，延迟处理
 * 更简单、点云-里程计一致性也更好（同一个源，天然零系差）。
 */
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <ros/callback_queue.h>
#include <ros/spinner.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>

#include "hand_lio/CustomMsg.h"
#include "hand_lio/FusionFrontend.h"

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
        // 这一时刻的世界系修正量 T_corr（见文件头 pose_source=fused），stream 模式下恒为单位变换。
        // 点云的世界系坐标 = corr_q * (q * p_imu + p) + corr_p
        Eigen::Quaterniond corr_q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d corr_p = Eigen::Vector3d::Zero();
    };

    struct PendingFrame {
        CustomMsgConstPtr msg;
        double end_time = 0.0;        // 帧尾最后一个点的采集时刻
        ros::Time arrival_wall_time;  // 到达本节点的墙钟时刻，用于超时兜底
        // 已经夹到 msg->points.size() 的点数。**遍历 msg->points 必须用这个**，
        // 不要再去读 msg->point_num —— 那是上游声称的值，比实际带的点多就越界。
        int point_num = 0;
    };

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void lidarCallback(const CustomMsgConstPtr& msg);

    // 缓存 navibot 发来的虚拟障碍点（世界系，latched，只在用户改禁行区时更新）。
    void virtualObstacleCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    // 把缓存里"从 sensor_pos 看得见、且在量程内"的虚拟障碍点追加到本帧点云。
    // 见 .cpp 里的实现说明——为什么要可见性剔除、为什么要按距离加密。
    void appendVirtualObstacles(const Eigen::Vector3d& sensor_pos, pcl::PointCloud<pcl::PointXYZI>& cloud) const;

    // 处理待处理队列：队首帧已被 odom 覆盖到帧尾、或等待超时，则出队处理。
    // 只在主线程调用（lidar 回调、drain_timer_）。自己拿 odom_mutex_，只在锁里
    // 出队 + 拷出这一帧用得到的那段位姿，去畸变在锁外做，不挡 odom 线程。
    void drainPendingFrames();

    // 把点云打包成紧凑的 16 字节 PointCloud2(x/y/z/intensity)。不用 pcl::toROSMsg——
    // 那个照搬 pcl::PointXYZI 的 32 字节布局, 一半是没有字段声明、也没人写过的
    // 填充字节(实测是未初始化堆内存)。见 .cpp 里的说明。
    static void packXYZI(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                         sensor_msgs::PointCloud2& out);

    // 去畸变 + 转 map 系 + 发布。poses 是 drainPendingFrames 在锁里拷出来的那段
    // 位姿（覆盖这一帧），latest_cov0 是当时缓冲区最新一条的定位方差。不需要锁。
    void processFrame(const PendingFrame& frame, const std::deque<PoseSample>& poses, double latest_cov0);

    // 在 [buf.front().t, buf.back().t] 范围内对位姿做线性/球面插值。
    static bool interpolatePose(const std::deque<PoseSample>& buf, double t, PoseSample& out);

    // 发一条 world -> body 的 Odometry（twist 全零）；with_tf 时同时发 TF。
    void publishBodyOdom(ros::Publisher& pub, const ros::Time& stamp, const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& p, double cov0, bool with_tf);

    // odom 专用回调队列，必须声明在 odom_sub_ 之前（订阅先析构，队列后析构）。
    ros::CallbackQueue odom_queue_;
    ros::Subscriber odom_sub_;
    ros::Subscriber lidar_sub_;
    ros::Subscriber virtual_obstacle_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher vehicle_odom_pub_;
    ros::Publisher stream_vehicle_odom_pub_;  // 高频流合成的机体位姿，两种模式都发，录包对比用
    ros::Publisher sensor_odom_pub_;          // 修正后的 IMU 位姿（口径同 /latest_imu_odom），给栅格地图当射线原点
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    ros::WallTimer drain_timer_;

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
    // shared_ptr 而不是直接放 vector: 每帧要读它, 直接拷一份的话 11.6 万点就是
    // 每帧 2.8MB 的 memcpy(10Hz = 28MB/s), 纯浪费。改成在锁里只拷一个指针, 扫描
    // 在锁外做; 回调整个换一个新的 shared_ptr, 老的等最后一个读者用完自己释放。
    mutable std::mutex virtual_obstacle_mutex_;
    std::shared_ptr<const std::vector<VirtualObstaclePoint>> virtual_obstacle_pts_;

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
    // 跟 config/hand_lio.yaml 保持一致。**两处必须同步**：launch 才会加载那个
    // yaml，直接 rosrun 时用的是这里的默认值——两边不一致的话，同一个节点用
    // 两种方式启动会跑在不同的盲区半径上，而且不会有任何提示。
    // 这个值本身该实测，理由见 yaml 里的注释。
    double blind_ = 0.35;
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

    // ---- 位姿来源（见文件头）----
    // 融合相关的东西只在 odom 线程里访问（FusionFrontend 的回调也挂在 odom_queue_ 上），不用锁。
    bool fused_ = false;
    std::unique_ptr<FusionFrontend> frontend_;
    bool fused_active_ = false;  // 上一条位姿用的是不是融合结果，只用来打切换日志
    bool have_stream_ = false;
    Eigen::Vector3d last_stream_p_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond last_stream_q_ = Eigen::Quaterniond::Identity();

    // 最后声明 = 最先析构：先停掉 odom 线程，它用到的成员才能放心析构。
    std::unique_ptr<ros::AsyncSpinner> odom_spinner_;
};

}  // namespace hand_lio
