/*
 * FusionFrontend —— PoseFusion 的 ROS 输入侧，hand_lio_node（pose_source=fused）和
 * pose_fusion_shadow_node 共用：
 *
 *   - 订阅 /lidar_pose，换算到机体后交给 PoseFusion 当锚点，并发一条 PoseFusionDiag；
 *   - 订阅速度源，按来源乘各自的比例后交给 PoseFusion；
 *   - 判断融合结果现在能不能用（有过锚点、/lidar_pose 没断流）。
 *
 * 速度源（velocity_source）：
 *   motion_status  deep_bridge 转发的本体实测机体速度（10Hz）。默认。10-06 两段 M20S
 *                  楼梯实录上，LinearX 就是机身前进速度（比例 0.97~0.98），前推比
 *                  cmd_vel 略准；更要紧的是闭环后它不依赖指令——机器狗被挡住时实测
 *                  速度是零，cmd_vel 不是。
 *   cmd_vel        控制器下发的指令。
 * velocity_fallback=true 时，motion_status 超过 velocity_fallback_timeout 没来就临时
 * 改用 cmd_vel，恢复后自动切回。
 *
 * 所有回调都挂在构造时给的回调队列上。hand_lio_node 把它们放进 /latest_imu_odom 那条
 * 专用线程，PoseFusion 就只在一个线程里被访问，不用加锁。
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include "hand_lio/PoseFusion.h"
#include "hand_lio/PoseFusionDiag.h"

namespace hand_lio {

class FusionFrontend {
public:
    struct Scale {
        double vx = 1.0;
        double vy = 1.0;
        double wz = 1.0;
    };

    struct Config {
        PoseFusion::Params params;  // 其中的 scale_* 不用，比例按来源在这里乘
        std::string velocity_source = "motion_status";
        bool velocity_fallback = true;
        double velocity_fallback_timeout = 0.5;  // [s]
        Scale cmd_vel_scale;
        Scale motion_status_scale;
        double lidar_timeout = 3.0;  // [s] /lidar_pose 断流超过它，融合结果不再可用
        std::string lidar_pose_topic = "/lidar_pose";
        std::string cmd_vel_topic = "/cmd_vel";
        std::string motion_status_topic = "/deep_bridge_node/motion_status";
        std::string diag_topic = "/hand_lio/pose_fusion_diag";
    };

    // 从 nh 所在命名空间读参数（键名见 config/hand_lio.yaml 的 fusion 段）
    static Config loadConfig(const ros::NodeHandle& nh)
    {
        Config c;
        PoseFusion::Params& p = c.params;
        nh.param("tau", p.tau, p.tau);
        nh.param("velocity_timeout", p.velocity_timeout, p.velocity_timeout);
        nh.param("slope_window", p.slope_window, p.slope_window);
        nh.param("slope_min_ds", p.slope_min_ds, p.slope_min_ds);
        nh.param("slope_max", p.slope_max, p.slope_max);
        nh.param("jump_threshold", p.jump_threshold, p.jump_threshold);
        nh.param("max_vz", p.max_vz, p.max_vz);
        nh.param("buffer_horizon", p.buffer_horizon, p.buffer_horizon);
        nh.param("yaw_from_stream", p.yaw_from_stream, true);
        nh.param("stream_gap_max", p.stream_gap_max, p.stream_gap_max);
        nh.param("stream_yaw_rate_max", p.stream_yaw_rate_max, p.stream_yaw_rate_max);

        nh.param("velocity_source", c.velocity_source, c.velocity_source);
        if (c.velocity_source != "cmd_vel" && c.velocity_source != "motion_status")
            throw std::runtime_error("velocity_source must be cmd_vel or motion_status, got '" + c.velocity_source + "'");
        nh.param("velocity_fallback", c.velocity_fallback, c.velocity_fallback);
        nh.param("velocity_fallback_timeout", c.velocity_fallback_timeout, c.velocity_fallback_timeout);
        nh.param("cmd_vel_scale/vx", c.cmd_vel_scale.vx, c.cmd_vel_scale.vx);
        nh.param("cmd_vel_scale/vy", c.cmd_vel_scale.vy, c.cmd_vel_scale.vy);
        nh.param("cmd_vel_scale/wz", c.cmd_vel_scale.wz, c.cmd_vel_scale.wz);
        nh.param("motion_status_scale/vx", c.motion_status_scale.vx, c.motion_status_scale.vx);
        nh.param("motion_status_scale/vy", c.motion_status_scale.vy, c.motion_status_scale.vy);
        nh.param("motion_status_scale/wz", c.motion_status_scale.wz, c.motion_status_scale.wz);
        nh.param("lidar_timeout", c.lidar_timeout, c.lidar_timeout);
        nh.param("lidar_pose_topic", c.lidar_pose_topic, c.lidar_pose_topic);
        nh.param("cmd_vel_topic", c.cmd_vel_topic, c.cmd_vel_topic);
        nh.param("motion_status_topic", c.motion_status_topic, c.motion_status_topic);
        nh.param("diag_topic", c.diag_topic, c.diag_topic);
        return c;
    }

    // queue 为空时用 nh 的默认队列。lidar_R_body/lidar_t_body 同 hand_lio.yaml：
    // p_lidar = lidar_R_body * p_body + lidar_t_body。
    FusionFrontend(ros::NodeHandle& nh, ros::CallbackQueue* queue, const Config& config,
                   const Eigen::Matrix3d& lidar_R_body, const Eigen::Vector3d& lidar_t_body,
                   const std::string& world_frame_id)
        : cfg_(config), lidar_R_body_(lidar_R_body), lidar_t_body_(lidar_t_body), world_frame_id_(world_frame_id)
    {
        PoseFusion::Params p = cfg_.params;
        p.scale_vx = p.scale_vy = p.scale_wz = 1.0;
        fusion_.reset(new PoseFusion(p));

        diag_pub_ = nh.advertise<PoseFusionDiag>(cfg_.diag_topic, 50);
        const ros::TransportHints no_delay = ros::TransportHints().tcpNoDelay();
        lidar_sub_ = subscribe<geometry_msgs::PoseWithCovarianceStamped>(
            nh, queue, cfg_.lidar_pose_topic, 20, &FusionFrontend::lidarPoseCallback, no_delay);
        cmd_sub_ = subscribe<geometry_msgs::Twist>(nh, queue, cfg_.cmd_vel_topic, 200, &FusionFrontend::cmdVelCallback,
                                                   no_delay);
        if (cfg_.velocity_source == "motion_status")
            motion_sub_ = subscribe<geometry_msgs::TwistStamped>(nh, queue, cfg_.motion_status_topic, 50,
                                                                 &FusionFrontend::motionStatusCallback, no_delay);
    }

    PoseFusion& fusion() { return *fusion_; }
    const Config& config() const { return cfg_; }

    // 融合结果现在能不能用：有过锚点，且 /lidar_pose 没有断流
    bool active(const ros::Time& now) const
    {
        return fusion_->initialized() && (now - last_lidar_arrival_).toSec() <= cfg_.lidar_timeout;
    }
    double lidarSilence(const ros::Time& now) const { return (now - last_lidar_arrival_).toSec(); }

    // 最新一条 /lidar_pose 换算到机体后的姿态、航向和定位方差
    const Eigen::Matrix3d& lastLidarBodyR() const { return last_R_world_body_; }
    double lastLidarBodyYaw() const { return last_yaw_; }
    double lastLidarCov0() const { return last_cov0_; }

    // 仿真时间回绕（rosbag play -l）或时钟被改：从头来
    bool resetIfTimeWentBack(double now)
    {
        if (fusion_->initialized() && now < fusion_->lastTime() - 1.0)
        {
            ROS_WARN("[pose_fusion] time went backwards (%.3f -> %.3f), resetting", fusion_->lastTime(), now);
            fusion_->reset();
            return true;
        }
        return false;
    }

    static double yawOf(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

private:
    template <class M>
    ros::Subscriber subscribe(ros::NodeHandle& nh, ros::CallbackQueue* queue, const std::string& topic, int depth,
                              void (FusionFrontend::*cb)(const boost::shared_ptr<const M>&),
                              const ros::TransportHints& hints)
    {
        ros::SubscribeOptions ops = ros::SubscribeOptions::create<M>(topic, depth, boost::bind(cb, this, _1),
                                                                     ros::VoidPtr(), queue);
        ops.transport_hints = hints;
        return nh.subscribe(ops);
    }

    void addVelocity(const Scale& s, double vx, double vy, double wz)
    {
        fusion_->addVelocity(ros::Time::now().toSec(), s.vx * vx, s.vy * vy, s.wz * wz);
    }

    bool motionStatusFresh(const ros::Time& now) const
    {
        return !last_motion_status_.isZero() &&
               (now - last_motion_status_).toSec() <= cfg_.velocity_fallback_timeout;
    }

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        if (cfg_.velocity_source == "motion_status")
        {
            if (!cfg_.velocity_fallback || motionStatusFresh(ros::Time::now()))
                return;
            if (!using_fallback_)
            {
                ROS_WARN("[pose_fusion] %s silent for > %.2f s, propagating with %s until it is back",
                         cfg_.motion_status_topic.c_str(), cfg_.velocity_fallback_timeout, cfg_.cmd_vel_topic.c_str());
                using_fallback_ = true;
            }
        }
        // Twist 没有时间戳，按到达时刻算
        addVelocity(cfg_.cmd_vel_scale, msg->linear.x, msg->linear.y, msg->angular.z);
    }

    void motionStatusCallback(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        // 也按到达时刻算，跟 cmd_vel 同一个时间基准，来回切换时样本才保持有序。
        // deep_bridge 本来就是按收到报文的时刻盖戳的，两者只差一次 ROS 传输。
        last_motion_status_ = ros::Time::now();
        if (using_fallback_)
        {
            ROS_WARN("[pose_fusion] %s is back, propagating with it again", cfg_.motion_status_topic.c_str());
            using_fallback_ = false;
        }
        addVelocity(cfg_.motion_status_scale, msg->twist.linear.x, msg->twist.linear.y, msg->twist.angular.z);
    }

    void lidarPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
    {
        const ros::Time now = ros::Time::now();
        resetIfTimeWentBack(now.toSec());

        const auto& pp = msg->pose.pose;
        Eigen::Quaterniond q_world_lidar(pp.orientation.w, pp.orientation.x, pp.orientation.y, pp.orientation.z);
        q_world_lidar.normalize();
        const Eigen::Matrix3d R_world_lidar = q_world_lidar.toRotationMatrix();
        // 同 HandLioNode：world_T_body = world_T_lidar * lidar_T_body
        const Eigen::Matrix3d R_world_body = R_world_lidar * lidar_R_body_;
        const Eigen::Vector3d p_world_body =
            R_world_lidar * lidar_t_body_ + Eigen::Vector3d(pp.position.x, pp.position.y, pp.position.z);

        PoseFusion::Pose body;
        body.x = p_world_body.x();
        body.y = p_world_body.y();
        body.z = p_world_body.z();
        body.yaw = yawOf(R_world_body);

        const PoseFusion::AnchorResult r = fusion_->onLidarPose(now.toSec(), msg->header.stamp.toSec(), body);
        last_R_world_body_ = R_world_body;
        last_yaw_ = body.yaw;
        last_cov0_ = msg->pose.covariance[0];
        last_lidar_arrival_ = now;

        double vx, vy, wz;
        const bool vel_ok = fusion_->velocityAt(now.toSec(), vx, vy, wz);
        if (r.jump)
            ROS_WARN("[pose_fusion] /lidar_pose jump (threshold %.2f m): prediction off by [%.2f %.2f %.2f] m after "
                     "%.2f s, output snapped to it",
                     cfg_.params.jump_threshold, r.dx, r.dy, r.dz, r.age);
        if (r.age_exceeds_buffer && !r.first)
            ROS_WARN_THROTTLE(2.0, "[pose_fusion] /lidar_pose is %.2f s old, older than the velocity buffer", r.age);

        PoseFusionDiag d;
        d.header.stamp = now;
        d.header.frame_id = world_frame_id_;
        d.lidar_stamp = msg->header.stamp;
        d.age = r.age;
        d.innovation_x = r.dx;
        d.innovation_y = r.dy;
        d.innovation_z = r.dz;
        d.innovation_yaw = r.dyaw;
        d.slope = r.slope;
        d.first = r.first;
        d.jump = r.jump;
        d.age_exceeds_buffer = r.age_exceeds_buffer;
        d.velocity_stale = !vel_ok;
        d.velocity_source = using_fallback_ || cfg_.velocity_source == "cmd_vel" ? "cmd_vel" : "motion_status";
        diag_pub_.publish(d);
    }

    Config cfg_;
    Eigen::Matrix3d lidar_R_body_;
    Eigen::Vector3d lidar_t_body_;
    std::string world_frame_id_;
    std::unique_ptr<PoseFusion> fusion_;

    ros::Publisher diag_pub_;
    ros::Subscriber lidar_sub_;
    ros::Subscriber cmd_sub_;
    ros::Subscriber motion_sub_;

    ros::Time last_lidar_arrival_;
    ros::Time last_motion_status_;
    bool using_fallback_ = false;
    Eigen::Matrix3d last_R_world_body_ = Eigen::Matrix3d::Identity();
    double last_yaw_ = 0.0;
    double last_cov0_ = 0.0;
};

}  // namespace hand_lio
