#include "hand_lio/HandLioNode.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <geometry_msgs/TransformStamped.h>
#include <pcl/common/transforms.h>
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

        pnh.param("buffer_horizon_sec", buffer_horizon_sec_, buffer_horizon_sec_);
        pnh.param("pose_cov_reject_thresh", pose_cov_reject_thresh_, pose_cov_reject_thresh_);
        pnh.param("correction_smooth_tau", correction_smooth_tau_, correction_smooth_tau_);
        pnh.param("correction_jump_pos", correction_jump_pos_, correction_jump_pos_);
        pnh.param("correction_jump_ang_deg", correction_jump_ang_deg_, correction_jump_ang_deg_);
        pnh.param("correction_timeout_sec", correction_timeout_sec_, correction_timeout_sec_);
        pnh.param("world_frame_id", world_frame_id_, world_frame_id_);
        pnh.param("vehicle_frame_id", vehicle_frame_id_, vehicle_frame_id_);

        std::string odom_topic = "/latest_imu_odom";
        std::string elio_odom_topic = "/LIO/odom_imu";
        std::string elio_cloud_topic = "/LIO/clouds_lidar";
        std::string output_topic = "/hand_lio/clouds_lidar";
        std::string vehicle_odom_topic = "/hand_lio/odom_vehicle";
        pnh.param("odom_topic", odom_topic, odom_topic);
        pnh.param("elio_odom_topic", elio_odom_topic, elio_odom_topic);
        pnh.param("elio_cloud_topic", elio_cloud_topic, elio_cloud_topic);
        pnh.param("output_topic", output_topic, output_topic);
        pnh.param("vehicle_odom_topic", vehicle_odom_topic, vehicle_odom_topic);

        // /latest_imu_odom 200Hz，缓冲队列要能跟上；Elevator-LIO 的话题跟随每帧 lidar ~10Hz
        //（high_frequency_odom 开启时 odom_imu 会更高频，队列给足余量）。
        map_odom_sub_ = nh.subscribe(odom_topic, 2000, &HandLioNode::mapOdomCallback, this);
        elio_odom_sub_ = nh.subscribe(elio_odom_topic, 200, &HandLioNode::elioOdomCallback, this);
        elio_cloud_sub_ = nh.subscribe(elio_cloud_topic, 10, &HandLioNode::elioCloudCallback, this);
        cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 10);
        vehicle_odom_pub_ = nh.advertise<nav_msgs::Odometry>(vehicle_odom_topic, 200);

        ROS_INFO_STREAM("[hand_lio] odom_topic=" << odom_topic << " elio_odom_topic=" << elio_odom_topic
                                                 << " elio_cloud_topic=" << elio_cloud_topic << " output_topic="
                                                 << output_topic << " vehicle_odom_topic=" << vehicle_odom_topic
                                                 << " world_frame_id=" << world_frame_id_);
    }

    bool HandLioNode::pushSample(std::deque<PoseSample> &buf, const nav_msgs::Odometry::ConstPtr &msg, double horizon,
                                 const char *name)
    {
        PoseSample s;
        s.t = msg->header.stamp.toSec();
        s.p = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        s.q = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                                 msg->pose.pose.orientation.z);
        s.q.normalize();
        s.cov0 = msg->pose.covariance[0];

        bool cleared = false;
        if (!buf.empty() && s.t <= buf.back().t)
        {
            // 时间戳回退（播包重播/节点重启）：历史插值区间不连续，且发布方的坐标系
            // 可能已经换了原点，清空重新累积
            ROS_WARN_THROTTLE(1.0, "[hand_lio] %s timestamp went backwards, clearing pose buffer", name);
            buf.clear();
            cleared = true;
        }
        buf.push_back(s);
        while (!buf.empty() && (s.t - buf.front().t) > horizon)
        {
            buf.pop_front();
        }
        return cleared;
    }

    bool HandLioNode::interpolatePose(const std::deque<PoseSample> &buf, double t, PoseSample &out)
    {
        if (buf.size() < 2)
            return false;
        if (t < buf.front().t || t > buf.back().t)
            return false;

        auto it = std::lower_bound(buf.begin(), buf.end(), t,
                                   [](const PoseSample &s, double tt)
                                   { return s.t < tt; });
        if (it == buf.begin())
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

    void HandLioNode::mapOdomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            if (pushSample(map_odom_buf_, msg, buffer_horizon_sec_, "/latest_imu_odom"))
            {
                correction_valid_ = false;
            }
            tryUpdateCorrection();
        } // buf_mutex_ 在此释放，publishVehicleOdom 不需要它

        publishVehicleOdom(msg);
    }

    void HandLioNode::elioOdomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(buf_mutex_);
        if (pushSample(elio_odom_buf_, msg, buffer_horizon_sec_, "elio odom_imu"))
        {
            correction_valid_ = false;
        }
        tryUpdateCorrection();
    }

    void HandLioNode::tryUpdateCorrection()
    {
        if (map_odom_buf_.size() < 2 || elio_odom_buf_.size() < 2)
            return;

        // 配对时刻取两条流共同覆盖的最新时刻：/latest_imu_odom 滞后时它是 map_odom_buf_
        // 的队尾（拿滞后消息查及时缓存，查过去总能查到——跟旧方案逐点查未来正相反）；
        // 若某侧反而更旧，则退到那一侧队尾，插值区间依然两边都覆盖。
        const double t_pair = std::min(map_odom_buf_.back().t, elio_odom_buf_.back().t);
        if (correction_valid_ && t_pair <= correction_pair_t_)
            return;

        PoseSample map_pose, elio_pose;
        if (!interpolatePose(map_odom_buf_, t_pair, map_pose) || !interpolatePose(elio_odom_buf_, t_pair, elio_pose))
            return;

        // 全局定位失败期间冻结修正量：继续沿用上一次的值，超过 correction_timeout_sec_
        // 后由点云回调那侧兜底丢帧
        if (map_pose.cov0 >= pose_cov_reject_thresh_)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] map localization covariance too high (%.3f >= %.3f), freeze correction",
                              map_pose.cov0, pose_cov_reject_thresh_);
            return;
        }

        // T_map_elio = map_T_imu(t) * elio_T_imu(t)^-1
        const Eigen::Quaterniond q_new = (map_pose.q * elio_pose.q.conjugate()).normalized();
        const Eigen::Vector3d p_new = map_pose.p - q_new * elio_pose.p;

        if (!correction_valid_)
        {
            corr_q_ = q_new;
            corr_p_ = p_new;
            correction_valid_ = true;
        }
        else
        {
            const double pos_jump = (p_new - corr_p_).norm();
            const double ang_jump_deg = corr_q_.angularDistance(q_new) * 180.0 / M_PI;
            if (pos_jump > correction_jump_pos_ || ang_jump_deg > correction_jump_ang_deg_)
            {
                // 大幅跳变说明全局重定位刚修正过（这正是用 /latest_imu_odom 的意义所在），
                // 平滑过渡反而会让点云在错误位置多停留一个时间常数，直接跟上
                ROS_INFO("[hand_lio] correction jump detected (%.2fm / %.1fdeg), snapping to new value", pos_jump,
                         ang_jump_deg);
                corr_q_ = q_new;
                corr_p_ = p_new;
            }
            else
            {
                // 修正量只随 Elevator-LIO 的漂移变化，是慢变量，指数平滑压掉两侧估计的抖动
                const double dt = t_pair - correction_pair_t_;
                const double alpha = 1.0 - std::exp(-dt / std::max(correction_smooth_tau_, 1e-3));
                corr_p_ = (1.0 - alpha) * corr_p_ + alpha * p_new;
                corr_q_ = corr_q_.slerp(alpha, q_new).normalized();
            }
        }
        correction_pair_t_ = t_pair;
    }

    void HandLioNode::elioCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        Eigen::Quaterniond q;
        Eigen::Vector3d p;
        const double cloud_t = msg->header.stamp.toSec();
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            if (!correction_valid_)
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] no map<-elio correction yet, drop cloud");
                return;
            }
            // 修正量过旧（/latest_imu_odom 断流或长时间定位失败）时，漂移增量不再可控，丢帧
            if (cloud_t - correction_pair_t_ > correction_timeout_sec_)
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] correction is stale (%.2fs > %.2fs), drop cloud",
                                  cloud_t - correction_pair_t_, correction_timeout_sec_);
                return;
            }
            if (!map_odom_buf_.empty() && map_odom_buf_.back().cov0 >= pose_cov_reject_thresh_)
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] localization covariance too high (%.3f >= %.3f), drop cloud",
                                  map_odom_buf_.back().cov0, pose_cov_reject_thresh_);
                return;
            }
            q = corr_q_;
            p = corr_p_;
        }

        // Elevator-LIO 的点是 PointXYZINormal，fromROSMsg 到 PointXYZI 只取 x/y/z/intensity，
        // 多余字段自动忽略
        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(*msg, cloud);
        if (cloud.empty())
            return;

        const Eigen::Affine3f T = Eigen::Translation3f(p.cast<float>()) * q.cast<float>();
        pcl::transformPointCloud(cloud, cloud, T);

        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(cloud, out_msg);
        out_msg.header.stamp = msg->header.stamp;
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
        // 定位质量透传给下游，而不是直接丢帧：odom_vehicle 是持续反馈给规划/控制的话题，
        // 静默断流比让下游自己判断 covariance[0] 风险更大。twist 保持全零，
        // 跟 Elevator-LIO::publish_body_odometry 实际行为一致（它也从没写过 twist 字段）。
        odom_body.pose.covariance[0] = msg->pose.covariance[0];
        if (msg->pose.covariance[0] >= pose_cov_reject_thresh_)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] odom_vehicle localization covariance too high (%.3f >= %.3f)",
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
