/*
 * pose_fusion_shadow_node
 *
 * 旁路验证用：把 PoseFusion（/lidar_pose 锚点 + 机体速度前推，见 PoseFusion.h）的结果
 * 发到单独的话题上，跟现有的 /hand_lio/odom_vehicle 并排录包对比。
 *
 * **不接入任何控制/规划通路**：不发 TF（world->body 已经由 hand_lio_node 在发，再发一份
 * 会打架），输出话题默认是 /hand_lio/odom_fused_shadow，没有任何节点订阅它。
 *
 * 速度源二选一（velocity_source）：
 *   cmd_vel        控制器下发的指令，没有时间戳，按到达时刻算。机器狗被挡住/打滑时
 *                  估计会按指令“先走”，最多走出 速度 x /lidar_pose 延迟 那么远。
 *   motion_status  deep_bridge 转发的本体运控状态上报（实测机体速度，10Hz）。
 *
 * 外参 lidar_R_body / lidar_t_body 直接读 hand_lio_node 的参数（extrinsic_ns），保证跟
 * /hand_lio/odom_vehicle 用的是同一组值，不另存一份。
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "hand_lio/PoseFusion.h"
#include "hand_lio/PoseFusionDiag.h"

namespace hand_lio {

class PoseFusionShadowNode {
public:
    PoseFusionShadowNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    {
        PoseFusion::Params p;
        pnh.param("tau", p.tau, p.tau);
        pnh.param("velocity_timeout", p.velocity_timeout, p.velocity_timeout);
        pnh.param("slope_window", p.slope_window, p.slope_window);
        pnh.param("slope_min_ds", p.slope_min_ds, p.slope_min_ds);
        pnh.param("slope_max", p.slope_max, p.slope_max);
        pnh.param("jump_threshold", p.jump_threshold, p.jump_threshold);
        pnh.param("max_vz", p.max_vz, p.max_vz);
        pnh.param("buffer_horizon", p.buffer_horizon, p.buffer_horizon);

        pnh.param("velocity_source", velocity_source_, velocity_source_);
        if (velocity_source_ != "cmd_vel" && velocity_source_ != "motion_status")
        {
            ROS_FATAL("[pose_fusion_shadow] velocity_source must be cmd_vel or motion_status, got '%s'",
                      velocity_source_.c_str());
            throw std::runtime_error("bad velocity_source");
        }
        // 两个速度源的比例分开配：cmd_vel 是“指令”，实测前进方向只走到约 0.87~0.90；
        // motion_status 是本体自己量的，先按 1.0。
        const std::string scale_ns = velocity_source_ + "_scale/";
        pnh.param(scale_ns + "vx", p.scale_vx, p.scale_vx);
        pnh.param(scale_ns + "vy", p.scale_vy, p.scale_vy);
        pnh.param(scale_ns + "wz", p.scale_wz, p.scale_wz);
        fusion_.reset(new PoseFusion(p));
        jump_threshold_ = p.jump_threshold;

        pnh.param("lidar_timeout", lidar_timeout_, lidar_timeout_);
        pnh.param("pose_cov_reject_thresh", pose_cov_reject_thresh_, pose_cov_reject_thresh_);
        pnh.param("world_frame_id", world_frame_id_, world_frame_id_);
        pnh.param("vehicle_frame_id", vehicle_frame_id_, vehicle_frame_id_);

        std::string extrinsic_ns = "/hand_lio_node";
        pnh.param("extrinsic_ns", extrinsic_ns, extrinsic_ns);
        loadExtrinsics(extrinsic_ns);

        std::string lidar_pose_topic = "/lidar_pose";
        std::string cmd_vel_topic = "/cmd_vel";
        std::string motion_status_topic = "/deep_bridge_node/motion_status";
        std::string output_topic = "/hand_lio/odom_fused_shadow";
        std::string diag_topic = "/hand_lio/pose_fusion_diag";
        double publish_rate = 100.0;
        pnh.param("lidar_pose_topic", lidar_pose_topic, lidar_pose_topic);
        pnh.param("cmd_vel_topic", cmd_vel_topic, cmd_vel_topic);
        pnh.param("motion_status_topic", motion_status_topic, motion_status_topic);
        pnh.param("output_topic", output_topic, output_topic);
        pnh.param("diag_topic", diag_topic, diag_topic);
        pnh.param("publish_rate", publish_rate, publish_rate);

        odom_pub_ = nh.advertise<nav_msgs::Odometry>(output_topic, 100);
        diag_pub_ = nh.advertise<PoseFusionDiag>(diag_topic, 50);

        const ros::TransportHints no_delay = ros::TransportHints().tcpNoDelay();
        lidar_sub_ = nh.subscribe(lidar_pose_topic, 20, &PoseFusionShadowNode::lidarPoseCallback, this, no_delay);
        if (velocity_source_ == "cmd_vel")
            vel_sub_ = nh.subscribe(cmd_vel_topic, 200, &PoseFusionShadowNode::cmdVelCallback, this, no_delay);
        else
            vel_sub_ = nh.subscribe(motion_status_topic, 50, &PoseFusionShadowNode::motionStatusCallback, this, no_delay);
        timer_ = nh.createTimer(ros::Duration(1.0 / publish_rate), &PoseFusionShadowNode::timerCallback, this);

        ROS_INFO_STREAM("[pose_fusion_shadow] SHADOW ONLY (no TF, nothing subscribes the output). lidar_pose="
                        << lidar_pose_topic << " velocity_source=" << velocity_source_ << " ("
                        << (velocity_source_ == "cmd_vel" ? cmd_vel_topic : motion_status_topic) << ", scale vx="
                        << p.scale_vx << " vy=" << p.scale_vy << " wz=" << p.scale_wz << ") output=" << output_topic
                        << " diag=" << diag_topic << " tau=" << p.tau << " jump_threshold=" << p.jump_threshold);
    }

private:
    void loadExtrinsics(const std::string& ns)
    {
        std::vector<double> r;
        std::vector<double> t;
        if (!ros::param::get(ns + "/lidar_R_body", r) || r.size() != 9 || !ros::param::get(ns + "/lidar_t_body", t) ||
            t.size() != 3)
        {
            ROS_FATAL("[pose_fusion_shadow] cannot read %s/lidar_R_body (9) and %s/lidar_t_body (3). They are "
                      "hand_lio_node's extrinsics; launch hand_lio first (or point extrinsic_ns at them).",
                      ns.c_str(), ns.c_str());
            throw std::runtime_error("missing extrinsics");
        }
        // 行优先，p_lidar = lidar_R_body * p_body + lidar_t_body（同 hand_lio.yaml）
        lidar_R_body_ << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
        lidar_t_body_ << t[0], t[1], t[2];
    }

    static double yawOf(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

    // 仿真时间回绕（rosbag play -l）或时钟被改：从头来
    bool timeWentBack(double now)
    {
        if (fusion_->initialized() && now < fusion_->lastTime() - 1.0)
        {
            ROS_WARN("[pose_fusion_shadow] time went backwards (%.3f -> %.3f), resetting", fusion_->lastTime(), now);
            fusion_->reset();
            have_lidar_ = false;
            return true;
        }
        return false;
    }

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        // Twist 没有时间戳，只能用到达时刻
        fusion_->addVelocity(ros::Time::now().toSec(), msg->linear.x, msg->linear.y, msg->angular.z);
    }

    void motionStatusCallback(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        fusion_->addVelocity(msg->header.stamp.toSec(), msg->twist.linear.x, msg->twist.linear.y,
                             msg->twist.angular.z);
    }

    void lidarPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
    {
        const ros::Time now = ros::Time::now();
        timeWentBack(now.toSec());

        const auto& pp = msg->pose.pose;
        Eigen::Quaterniond q_world_lidar(pp.orientation.w, pp.orientation.x, pp.orientation.y, pp.orientation.z);
        q_world_lidar.normalize();
        const Eigen::Matrix3d R_world_lidar = q_world_lidar.toRotationMatrix();
        // 同 HandLioNode::publishVehicleOdom：world_T_body = world_T_lidar * lidar_T_body
        const Eigen::Matrix3d R_world_body = R_world_lidar * lidar_R_body_;
        const Eigen::Vector3d p_world_body =
            R_world_lidar * lidar_t_body_ + Eigen::Vector3d(pp.position.x, pp.position.y, pp.position.z);

        PoseFusion::Pose body;
        body.x = p_world_body.x();
        body.y = p_world_body.y();
        body.z = p_world_body.z();
        body.yaw = yawOf(R_world_body);

        const PoseFusion::AnchorResult r = fusion_->onLidarPose(now.toSec(), msg->header.stamp.toSec(), body);

        anchor_R_world_body_ = R_world_body;
        anchor_yaw_ = body.yaw;
        lidar_cov0_ = msg->pose.covariance[0];
        last_lidar_arrival_ = now;
        have_lidar_ = true;

        double vx, vy, wz;
        const bool vel_ok = fusion_->velocityAt(now.toSec(), vx, vy, wz);

        if (r.jump)
            ROS_WARN("[pose_fusion_shadow] /lidar_pose jump (threshold %.2f m): prediction off by [%.2f %.2f %.2f] m "
                     "after %.2f s, output snapped to it",
                     jump_threshold_, r.dx, r.dy, r.dz, r.age);
        if (r.age_exceeds_buffer && !r.first)
            ROS_WARN_THROTTLE(2.0, "[pose_fusion_shadow] /lidar_pose is %.2f s old, older than the velocity buffer",
                              r.age);

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
        d.velocity_source = velocity_source_;
        diag_pub_.publish(d);
    }

    void timerCallback(const ros::TimerEvent&)
    {
        const ros::Time now = ros::Time::now();
        if (timeWentBack(now.toSec()) || !have_lidar_)
            return;

        fusion_->advance(now.toSec());
        const PoseFusion::Pose& p = fusion_->pose();

        // 横滚/俯仰取最新一条 /lidar_pose 的，航向换成前推后的
        const Eigen::Quaterniond q(
            Eigen::AngleAxisd(PoseFusion::wrapAngle(p.yaw - anchor_yaw_), Eigen::Vector3d::UnitZ()) *
            anchor_R_world_body_);

        nav_msgs::Odometry odom;
        odom.header.stamp = now;
        odom.header.frame_id = world_frame_id_;
        odom.child_frame_id = vehicle_frame_id_;
        odom.pose.pose.position.x = p.x;
        odom.pose.pose.position.y = p.y;
        odom.pose.pose.position.z = p.z;
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        // covariance[0] 的口径同 /hand_lio/odom_vehicle：定位质量 0~0.99，0.99 = 不可信。
        // /lidar_pose 断流时锚点已经过期，同样标成不可信。
        const double lidar_silence = (now - last_lidar_arrival_).toSec();
        if (lidar_silence > lidar_timeout_)
        {
            ROS_WARN_THROTTLE(2.0, "[pose_fusion_shadow] no /lidar_pose for %.1f s, output is pure dead reckoning",
                              lidar_silence);
            odom.pose.covariance[0] = pose_cov_reject_thresh_;
        }
        else
        {
            odom.pose.covariance[0] = lidar_cov0_;
        }
        // twist：正在用于前推的机体系速度（已乘比例），不是独立的测量
        double vx, vy, wz;
        fusion_->velocityAt(now.toSec(), vx, vy, wz);
        odom.twist.twist.linear.x = vx;
        odom.twist.twist.linear.y = vy;
        odom.twist.twist.angular.z = wz;
        odom_pub_.publish(odom);
    }

    std::unique_ptr<PoseFusion> fusion_;
    std::string velocity_source_ = "cmd_vel";
    double jump_threshold_ = 0.5;
    double lidar_timeout_ = 3.0;
    double pose_cov_reject_thresh_ = 0.99;
    std::string world_frame_id_ = "world";
    std::string vehicle_frame_id_ = "body";

    Eigen::Matrix3d lidar_R_body_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lidar_t_body_ = Eigen::Vector3d::Zero();

    bool have_lidar_ = false;
    Eigen::Matrix3d anchor_R_world_body_ = Eigen::Matrix3d::Identity();
    double anchor_yaw_ = 0.0;
    double lidar_cov0_ = 0.0;
    ros::Time last_lidar_arrival_;

    ros::Publisher odom_pub_;
    ros::Publisher diag_pub_;
    ros::Subscriber lidar_sub_;
    ros::Subscriber vel_sub_;
    ros::Timer timer_;
};

} // namespace hand_lio

int main(int argc, char** argv) {
    ros::init(argc, argv, "pose_fusion_shadow_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    try {
        hand_lio::PoseFusionShadowNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("[pose_fusion_shadow] %s", e.what());
        return 1;
    }
    return 0;
}
