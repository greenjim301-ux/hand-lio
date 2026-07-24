#include "hand_lio/HandLioNode.h"

#include <algorithm>
#include <vector>

#include <geometry_msgs/TransformStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace hand_lio
{

    HandLioNode::HandLioNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        std::vector<double> R_vec{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<double> t_vec{0.0, 0.0, 0.0};
        pnh.param("imu_R_lidar", R_vec, R_vec);
        pnh.param("imu_t_lidar", t_vec, t_vec);
        if (R_vec.size() == 9)
        {
            imu_R_lidar_ << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5], R_vec[6], R_vec[7], R_vec[8];
        }
        else
        {
            ROS_WARN("[hand_lio] imu_R_lidar size != 9, fallback to identity");
        }
        if (t_vec.size() == 3)
        {
            imu_t_lidar_ << t_vec[0], t_vec[1], t_vec[2];
        }
        else
        {
            ROS_WARN("[hand_lio] imu_t_lidar size != 3, fallback to zero");
        }

        std::vector<double> body_R_vec{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<double> body_t_vec{0.0, 0.0, 0.0};
        pnh.param("lidar_R_body", body_R_vec, body_R_vec);
        pnh.param("lidar_t_body", body_t_vec, body_t_vec);
        if (body_R_vec.size() == 9)
        {
            lidar_R_body_ << body_R_vec[0], body_R_vec[1], body_R_vec[2], body_R_vec[3], body_R_vec[4], body_R_vec[5],
                body_R_vec[6], body_R_vec[7], body_R_vec[8];
        }
        else
        {
            ROS_WARN("[hand_lio] lidar_R_body size != 9, fallback to identity");
        }
        if (body_t_vec.size() == 3)
        {
            lidar_t_body_ << body_t_vec[0], body_t_vec[1], body_t_vec[2];
        }
        else
        {
            ROS_WARN("[hand_lio] lidar_t_body size != 3, fallback to zero");
        }

        pnh.param("blind", blind_, blind_);
        pnh.param("point_filter_num", point_filter_num_, point_filter_num_);
        pnh.param("buffer_horizon_sec", buffer_horizon_sec_, buffer_horizon_sec_);
        pnh.param("pose_cov_reject_thresh", pose_cov_reject_thresh_, pose_cov_reject_thresh_);
        pnh.param("world_frame_id", world_frame_id_, world_frame_id_);
        pnh.param("vehicle_frame_id", vehicle_frame_id_, vehicle_frame_id_);
        pnh.param("n_scans", n_scans_, n_scans_);

        std::string lidar_topic = "/livox/lidar";
        std::string odom_topic = "/latest_imu_odom";
        std::string output_topic = "/hand_lio/clouds_world";
        std::string vehicle_odom_topic = "/LIO/odom_vehicle";
        pnh.param("lidar_topic", lidar_topic, lidar_topic);
        pnh.param("odom_topic", odom_topic, odom_topic);
        pnh.param("output_topic", output_topic, output_topic);
        pnh.param("vehicle_odom_topic", vehicle_odom_topic, vehicle_odom_topic);

        // /latest_imu_odom 200Hz，缓冲队列要能跟上；/livox/lidar 10Hz。
        odom_sub_ = nh.subscribe(odom_topic, 2000, &HandLioNode::odomCallback, this);
        lidar_sub_ = nh.subscribe(lidar_topic, 10, &HandLioNode::lidarCallback, this);
        cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 10);
        vehicle_odom_pub_ = nh.advertise<nav_msgs::Odometry>(vehicle_odom_topic, 200);

        ROS_INFO_STREAM("[hand_lio] lidar_topic=" << lidar_topic << " odom_topic=" << odom_topic << " output_topic="
                                                  << output_topic << " vehicle_odom_topic=" << vehicle_odom_topic
                                                  << " world_frame_id=" << world_frame_id_);
    }

    void HandLioNode::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        PoseSample s;
        s.t = msg->header.stamp.toSec();
        s.p = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        s.q = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                                 msg->pose.pose.orientation.z);
        s.q.normalize();
        // hand-topic.csv: covariance[0] 是定位方差，0.0~0.99，越大越不可信，0.99 表示定位失败
        s.cov0 = msg->pose.covariance[0];

        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (!odom_buf_.empty() && s.t <= odom_buf_.back().t)
            {
                // 时间戳回退（例如播包重播），历史插值区间已经不连续，清空重新累积
                ROS_WARN_THROTTLE(1.0, "[hand_lio] /latest_imu_odom timestamp went backwards, clearing pose buffer");
                odom_buf_.clear();
            }
            odom_buf_.push_back(s);
            while (!odom_buf_.empty() && (s.t - odom_buf_.front().t) > buffer_horizon_sec_)
            {
                odom_buf_.pop_front();
            }
        } // odom_mutex_ 在此释放，publishVehicleOdom 不需要它

        publishVehicleOdom(msg);
    }

    bool HandLioNode::interpolatePose(double t, PoseSample &out) const
    {
        if (odom_buf_.size() < 2)
            return false;
        if (t < odom_buf_.front().t || t > odom_buf_.back().t)
            return false;

        auto it = std::lower_bound(odom_buf_.begin(), odom_buf_.end(), t,
                                   [](const PoseSample &s, double tt)
                                   { return s.t < tt; });
        if (it == odom_buf_.begin())
        {
            out = *it;
            return true;
        }
        const PoseSample &hi = *it;
        const PoseSample &lo = *(it - 1);
        const double span = hi.t - lo.t;
        const double ratio = span > 1e-9 ? (t - lo.t) / span : 0.0;

        out.t = t;
        out.p = (1.0 - ratio) * lo.p + ratio * hi.p;
        out.q = lo.q.slerp(ratio, hi.q);
        out.cov0 = std::max(lo.cov0, hi.cov0);
        return true;
    }

    void HandLioNode::lidarCallback(const CustomMsgConstPtr &msg)
    {
        const int point_num = static_cast<int>(msg->point_num);
        if (point_num <= 1 || msg->points.empty())
            return;

        const double base_time = msg->header.stamp.toSec();
        const double end_time = base_time + static_cast<double>(msg->points.back().offset_time) * 1e-9;

        std::lock_guard<std::mutex> lock(odom_mutex_);

        // 只要求覆盖到这一帧的起始时刻 base_time：end_time 约等于"此刻"，
        // /latest_imu_odom 作为下游定位估计天然会比物理时刻滞后（计算耗时+网络延迟），
        // 要求整帧覆盖到 end_time 会导致逐帧必然被丢弃。末尾这几毫秒对应的点在下面
        // 逐点插值时会自然被跳过（interpolatePose 超出范围返回 false），不必在此整帧拒绝。
        if (odom_buf_.empty() || odom_buf_.front().t > base_time || odom_buf_.back().t < base_time)
        {
            if (odom_buf_.empty())
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] /latest_imu_odom buffer is empty, drop frame");
            }
            else
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] /latest_imu_odom buffer time range [%.3f, %.3f] does not cover this lidar scan start time %.3f, drop frame",
                                  odom_buf_.front().t, odom_buf_.back().t, base_time);
            }
            return;
        }

        // 用缓冲区里最新的样本做定位质量把关（而不是插值到 end_time，理由同上）
        const double latest_cov0 = odom_buf_.back().cov0;
        if (latest_cov0 >= pose_cov_reject_thresh_)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] localization covariance too high (%.3f >= %.3f), drop frame", latest_cov0,
                              pose_cov_reject_thresh_);
            return;
        }

        pcl::PointCloud<pcl::PointXYZI> cloud_world;
        cloud_world.reserve(point_num);

        uint32_t valid_num = 0;
        for (int i = 1; i < point_num; ++i)
        { // 第 0 个点跳过，Livox CustomMsg 的常见约定（同 Elevator-LIO）
            const auto &pt = msg->points[i];

            const bool tag_ok = (pt.tag & tag_mask_) == 0x10 || (pt.tag & tag_mask_) == 0x00;
            if (!(pt.line < n_scans_ && tag_ok))
                continue;

            ++valid_num;
            if (point_filter_num_ > 1 && (valid_num % point_filter_num_) != 0)
                continue;

            const double r2 = static_cast<double>(pt.x) * pt.x + static_cast<double>(pt.y) * pt.y +
                              static_cast<double>(pt.z) * pt.z;
            if (r2 <= blind_ * blind_)
                continue;

            const double t_point = base_time + static_cast<double>(pt.offset_time) * 1e-9;
            PoseSample pose_i;
            if (!interpolatePose(t_point, pose_i))
                continue;

            // 去畸变 + 转 world 系一步完成：用该点自己采集时刻的插值位姿直接变换
            const Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
            const Eigen::Vector3d p_imu = imu_R_lidar_ * p_lidar + imu_t_lidar_;
            const Eigen::Vector3d p_world = pose_i.q * p_imu + pose_i.p;

            pcl::PointXYZI out_pt;
            out_pt.x = static_cast<float>(p_world.x());
            out_pt.y = static_cast<float>(p_world.y());
            out_pt.z = static_cast<float>(p_world.z());
            out_pt.intensity = static_cast<float>(pt.reflectivity);
            cloud_world.push_back(out_pt);
        }

        if (cloud_world.empty())
            return;
        cloud_world.width = cloud_world.size();
        cloud_world.height = 1;
        cloud_world.is_dense = true;

        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(cloud_world, out_msg);
        out_msg.header.stamp = ros::Time(end_time);
        out_msg.header.frame_id = world_frame_id_;
        cloud_pub_.publish(out_msg);
    }

    void HandLioNode::publishVehicleOdom(const nav_msgs::Odometry::ConstPtr &msg)
    {
        // world_T_imu：直接用这一条 /latest_imu_odom 自己的 pose，不需要缓冲区插值
        Eigen::Matrix4d world_T_imu = Eigen::Matrix4d::Identity();
        const Eigen::Quaterniond q_world_imu(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        world_T_imu.block<3, 3>(0, 0) = q_world_imu.normalized().toRotationMatrix();
        world_T_imu.block<3, 1>(0, 3) =
            Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);

        Eigen::Matrix4d imu_T_lidar = Eigen::Matrix4d::Identity();
        imu_T_lidar.block<3, 3>(0, 0) = imu_R_lidar_;
        imu_T_lidar.block<3, 1>(0, 3) = imu_t_lidar_;

        Eigen::Matrix4d lidar_T_body = Eigen::Matrix4d::Identity();
        lidar_T_body.block<3, 3>(0, 0) = lidar_R_body_;
        lidar_T_body.block<3, 1>(0, 3) = lidar_t_body_;

        // 参照 Elevator-LIO::publish_body_odometry 的合成方式
        const Eigen::Matrix4d world_T_body = world_T_imu * imu_T_lidar * lidar_T_body;
        const Eigen::Quaterniond body_q(world_T_body.block<3, 3>(0, 0));
        const Eigen::Vector3d body_p = world_T_body.block<3, 1>(0, 3);

        nav_msgs::Odometry odom_body;
        odom_body.header.stamp = msg->header.stamp;
        odom_body.header.frame_id = world_frame_id_;
        odom_body.child_frame_id = vehicle_frame_id_;
        odom_body.pose.pose.position.x = body_p.x();
        odom_body.pose.pose.position.y = body_p.y();
        odom_body.pose.pose.position.z = body_p.z();
        odom_body.pose.pose.orientation.w = body_q.w();
        odom_body.pose.pose.orientation.x = body_q.x();
        odom_body.pose.pose.orientation.y = body_q.y();
        odom_body.pose.pose.orientation.z = body_q.z();
        // 定位质量透传给下游，而不是直接丢帧：/LIO/odom_vehicle 是持续反馈给规划/控制的话题，
        // 静默断流比让下游自己判断 covariance[0] 风险更大。twist 保持全零，
        // 跟 Elevator-LIO::publish_body_odometry 实际行为一致（它也从没写过 twist 字段）。
        odom_body.pose.covariance[0] = msg->pose.covariance[0];
        if (msg->pose.covariance[0] >= pose_cov_reject_thresh_)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] /LIO/odom_vehicle localization covariance too high (%.3f >= %.3f)",
                              msg->pose.covariance[0], pose_cov_reject_thresh_);
        }
        vehicle_odom_pub_.publish(odom_body);

        geometry_msgs::TransformStamped trans;
        trans.header.stamp = odom_body.header.stamp;
        trans.header.frame_id = world_frame_id_;
        trans.child_frame_id = vehicle_frame_id_;
        trans.transform.translation.x = body_p.x();
        trans.transform.translation.y = body_p.y();
        trans.transform.translation.z = body_p.z();
        trans.transform.rotation = odom_body.pose.pose.orientation;
        tf_broadcaster_.sendTransform(trans);
    }

} // namespace hand_lio
