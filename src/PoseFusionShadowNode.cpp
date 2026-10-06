/*
 * pose_fusion_shadow_node
 *
 * 旁路验证用：把 PoseFusion（/lidar_pose 锚点 + 机体速度前推，见 PoseFusion.h）的结果
 * 发到单独的话题上，跟现有的 /hand_lio/odom_vehicle 并排录包对比。
 *
 * **不接入任何控制/规划通路**：不发 TF（world->body 由 hand_lio_node 在发，再发一份
 * 会打架），输出话题默认是 /hand_lio/odom_fused_shadow，没有任何节点订阅它。
 *
 * hand_lio_node 设成 pose_source=fused 之后，它自己就在发同样的融合位姿和
 * /hand_lio/pose_fusion_diag，这时不要再起本节点（诊断话题会重名）。
 *
 * 输入侧（/lidar_pose、速度源、诊断）跟 hand_lio_node 共用 FusionFrontend，参数键名
 * 也相同，见 config/pose_fusion_shadow.yaml。航向用高频流增量（yaw_from_stream）时
 * 订阅 /latest_imu_odom。外参 imu_R_lidar / lidar_R_body / lidar_t_body 直接读
 * hand_lio_node 的参数（extrinsic_ns），保证跟 /hand_lio/odom_vehicle 用同一组值。
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "hand_lio/FusionFrontend.h"

namespace hand_lio {

class PoseFusionShadowNode {
public:
    PoseFusionShadowNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    {
        const FusionFrontend::Config cfg = FusionFrontend::loadConfig(pnh);

        pnh.param("pose_cov_reject_thresh", pose_cov_reject_thresh_, pose_cov_reject_thresh_);
        pnh.param("world_frame_id", world_frame_id_, world_frame_id_);
        pnh.param("vehicle_frame_id", vehicle_frame_id_, vehicle_frame_id_);

        std::string extrinsic_ns = "/hand_lio_node";
        pnh.param("extrinsic_ns", extrinsic_ns, extrinsic_ns);
        loadExtrinsics(extrinsic_ns);

        std::string stream_topic = "/latest_imu_odom";
        std::string output_topic = "/hand_lio/odom_fused_shadow";
        double publish_rate = 100.0;
        pnh.param("stream_topic", stream_topic, stream_topic);
        pnh.param("output_topic", output_topic, output_topic);
        pnh.param("publish_rate", publish_rate, publish_rate);

        frontend_.reset(new FusionFrontend(nh, nullptr, cfg, lidar_R_body_, lidar_t_body_, world_frame_id_));
        odom_pub_ = nh.advertise<nav_msgs::Odometry>(output_topic, 100);
        stream_sub_ = nh.subscribe(stream_topic, 400, &PoseFusionShadowNode::streamCallback, this,
                                   ros::TransportHints().tcpNoDelay());
        timer_ = nh.createTimer(ros::Duration(1.0 / publish_rate), &PoseFusionShadowNode::timerCallback, this);

        ROS_INFO_STREAM("[pose_fusion_shadow] SHADOW ONLY (no TF, nothing subscribes the output). velocity_source="
                        << cfg.velocity_source << (cfg.velocity_fallback ? " (fallback cmd_vel)" : "")
                        << " yaw_from_stream=" << cfg.params.yaw_from_stream << " output=" << output_topic
                        << " diag=" << cfg.diag_topic << " tau=" << cfg.params.tau);
    }

private:
    void loadExtrinsics(const std::string& ns)
    {
        std::vector<double> rb, tb, ri;
        if (!ros::param::get(ns + "/lidar_R_body", rb) || rb.size() != 9 || !ros::param::get(ns + "/lidar_t_body", tb) ||
            tb.size() != 3)
        {
            ROS_FATAL("[pose_fusion_shadow] cannot read %s/lidar_R_body (9) and %s/lidar_t_body (3). They are "
                      "hand_lio_node's extrinsics; launch hand_lio first (or point extrinsic_ns at them).",
                      ns.c_str(), ns.c_str());
            throw std::runtime_error("missing extrinsics");
        }
        // 行优先，p_lidar = lidar_R_body * p_body + lidar_t_body（同 hand_lio.yaml）
        lidar_R_body_ << rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7], rb[8];
        lidar_t_body_ << tb[0], tb[1], tb[2];
        // imu_R_lidar 只用来把高频流的航向换算到机体；没配就按单位阵（hand_lio 的默认值）
        if (ros::param::get(ns + "/imu_R_lidar", ri) && ri.size() == 9)
            imu_R_lidar_ << ri[0], ri[1], ri[2], ri[3], ri[4], ri[5], ri[6], ri[7], ri[8];
    }

    // 高频流：只取机体姿态，喂给融合当航向增量、并给输出提供横滚/俯仰
    void streamCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        const auto& pp = msg->pose.pose;
        Eigen::Quaterniond q(pp.orientation.w, pp.orientation.x, pp.orientation.y, pp.orientation.z);
        q.normalize();
        const Eigen::Vector3d p(pp.position.x, pp.position.y, pp.position.z);
        const Eigen::Matrix3d R_body = q.toRotationMatrix() * imu_R_lidar_ * lidar_R_body_;
        const bool fresh = !have_stream_ || (p - last_stream_p_).norm() > 1e-9 ||
                           q.angularDistance(last_stream_q_) > 1e-9;
        frontend_->fusion().addStreamYaw(msg->header.stamp.toSec(), FusionFrontend::yawOf(R_body), fresh);
        last_stream_p_ = p;
        last_stream_q_ = q;
        stream_R_body_ = R_body;
        last_stream_arrival_ = ros::Time::now();
        have_stream_ = true;
    }

    void timerCallback(const ros::TimerEvent&)
    {
        const ros::Time now = ros::Time::now();
        if (frontend_->resetIfTimeWentBack(now.toSec()) || !frontend_->fusion().initialized())
            return;

        PoseFusion& fusion = frontend_->fusion();
        fusion.advance(now.toSec());
        const PoseFusion::Pose& p = fusion.pose();

        // 横滚/俯仰：高频流在就用它的（200Hz），否则用最新一条 /lidar_pose 的；航向换成融合的
        const bool stream_ok = have_stream_ && (now - last_stream_arrival_).toSec() < 0.1;
        const Eigen::Matrix3d R_ref = stream_ok ? stream_R_body_ : frontend_->lastLidarBodyR();
        const Eigen::Quaterniond q(
            Eigen::AngleAxisd(PoseFusion::wrapAngle(p.yaw - FusionFrontend::yawOf(R_ref)), Eigen::Vector3d::UnitZ()) *
            R_ref);

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
        if (!frontend_->active(now))
        {
            ROS_WARN_THROTTLE(2.0, "[pose_fusion_shadow] no /lidar_pose for %.1f s, output is pure dead reckoning",
                              frontend_->lidarSilence(now));
            odom.pose.covariance[0] = pose_cov_reject_thresh_;
        }
        else
        {
            odom.pose.covariance[0] = frontend_->lastLidarCov0();
        }
        // twist：正在用于前推的机体系速度（已乘比例），不是独立的测量
        double vx, vy, wz;
        fusion.velocityAt(now.toSec(), vx, vy, wz);
        odom.twist.twist.linear.x = vx;
        odom.twist.twist.linear.y = vy;
        odom.twist.twist.angular.z = wz;
        odom_pub_.publish(odom);
    }

    std::unique_ptr<FusionFrontend> frontend_;
    double pose_cov_reject_thresh_ = 0.99;
    std::string world_frame_id_ = "world";
    std::string vehicle_frame_id_ = "body";

    Eigen::Matrix3d imu_R_lidar_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d lidar_R_body_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d lidar_t_body_ = Eigen::Vector3d::Zero();

    bool have_stream_ = false;
    Eigen::Vector3d last_stream_p_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond last_stream_q_ = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d stream_R_body_ = Eigen::Matrix3d::Identity();
    ros::Time last_stream_arrival_;

    ros::Publisher odom_pub_;
    ros::Subscriber stream_sub_;
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
