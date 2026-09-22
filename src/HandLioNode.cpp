#include "hand_lio/HandLioNode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
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
        pnh.param("pending_timeout_sec", pending_timeout_sec_, pending_timeout_sec_);
        pnh.param("world_frame_id", world_frame_id_, world_frame_id_);
        pnh.param("vehicle_frame_id", vehicle_frame_id_, vehicle_frame_id_);
        pnh.param("n_scans", n_scans_, n_scans_);
        pnh.param("enable_virtual_obstacles", enable_virtual_obstacles_, enable_virtual_obstacles_);
        pnh.param("virtual_obstacle_topic", virtual_obstacle_topic_, virtual_obstacle_topic_);
        pnh.param("virtual_obstacle_max_range", virtual_obstacle_max_range_, virtual_obstacle_max_range_);
        pnh.param("virtual_obstacle_density_gain", virtual_obstacle_density_gain_, virtual_obstacle_density_gain_);
        pnh.param("virtual_obstacle_max_copies", virtual_obstacle_max_copies_, virtual_obstacle_max_copies_);
        pnh.param("virtual_obstacle_max_points", virtual_obstacle_max_points_, virtual_obstacle_max_points_);

        std::string lidar_topic = "/livox/lidar";
        std::string odom_topic = "/latest_imu_odom";
        std::string output_topic = "/hand_lio/clouds_lidar";
        std::string vehicle_odom_topic = "/hand_lio/odom_vehicle";
        pnh.param("lidar_topic", lidar_topic, lidar_topic);
        pnh.param("odom_topic", odom_topic, odom_topic);
        pnh.param("output_topic", output_topic, output_topic);
        pnh.param("vehicle_odom_topic", vehicle_odom_topic, vehicle_odom_topic);

        // /latest_imu_odom 200Hz，缓冲队列要能跟上；/livox/lidar 10Hz。
        odom_sub_ = nh.subscribe(odom_topic, 2000, &HandLioNode::odomCallback, this);
        lidar_sub_ = nh.subscribe(lidar_topic, 10, &HandLioNode::lidarCallback, this);
        cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 10);
        vehicle_odom_pub_ = nh.advertise<nav_msgs::Odometry>(vehicle_odom_topic, 200);
        if (enable_virtual_obstacles_)
        {
            // 对面是 latched 的，队列 1 就够：连上立刻收到当前这一份，之后只有
            // 用户改禁行区才会再发。
            virtual_obstacle_sub_ =
                nh.subscribe(virtual_obstacle_topic_, 1, &HandLioNode::virtualObstacleCallback, this);
        }

        ROS_INFO_STREAM("[hand_lio] lidar_topic=" << lidar_topic << " odom_topic=" << odom_topic << " output_topic="
                                                  << output_topic << " vehicle_odom_topic=" << vehicle_odom_topic
                                                  << " world_frame_id=" << world_frame_id_
                                                  << " virtual_obstacles="
                                                  << (enable_virtual_obstacles_ ? virtual_obstacle_topic_ : std::string("off")));
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
            if (!odom_buf_.empty() && s.t < odom_buf_.back().t)
            {
                // 判据是**严格**回退。重复的时间戳是无害的：interpolatePose 的
                // span > 1e-9 已经挡住了零长度区间，lower_bound 对非递减序列（含
                // 重复值）也照样正确。用 <= 的话，一条重复位姿（回放循环、上游
                // 重发同一帧）就会连 pending_frames_ 一起清掉，把在途的那帧雷达
                // 直接丢了——清空之后 processFrame 的覆盖检查必然不通过。
                // 只有严格回退才真的让历史插值区间不连续，那时才该清。
                ROS_WARN_THROTTLE(1.0, "[hand_lio] /latest_imu_odom timestamp went backwards, clearing pose buffer");
                odom_buf_.clear();
                pending_frames_.clear();
            }
            odom_buf_.push_back(s);
            while (!odom_buf_.empty() && (s.t - odom_buf_.front().t) > buffer_horizon_sec_)
            {
                odom_buf_.pop_front();
            }
            drainPendingFrames();
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
        // point_num 是消息自己声称的点数，processFrame 拿它当 msg->points[i] 的上界。
        // 声称的比实际带的多就会越界读，所以在入口就夹到 points.size()，
        // 后面那一路不用再各自防一遍。
        const size_t declared = msg->point_num;
        if (declared > msg->points.size())
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] lidar frame claims %zu points but carries %zu, clamping",
                              declared, msg->points.size());
        }
        const int point_num = static_cast<int>(std::min(declared, msg->points.size()));
        if (point_num <= 1)
            return;

        PendingFrame frame;
        frame.msg = msg;
        frame.point_num = point_num;
        frame.end_time = msg->header.stamp.toSec() +
                         static_cast<double>(msg->points[point_num - 1].offset_time) * 1e-9;
        frame.arrival_wall_time = ros::Time::now();

        std::lock_guard<std::mutex> lock(odom_mutex_);
        pending_frames_.push_back(frame);
        // 正常滞后 2~20ms 时队列里最多 1 帧；积压说明 odom 断流已久，只留最新的几帧
        while (pending_frames_.size() > 5)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] pending frame queue overflow, dropping oldest lidar frame");
            pending_frames_.pop_front();
        }
        drainPendingFrames();
    }

    void HandLioNode::drainPendingFrames()
    {
        while (!pending_frames_.empty())
        {
            const PendingFrame &front = pending_frames_.front();
            const bool covered = !odom_buf_.empty() && odom_buf_.back().t >= front.end_time;
            // 实测 /latest_imu_odom 滞后 2~20ms，正常情况下等一两条 odom 就覆盖到帧尾了；
            // 超时兜底只在 odom 停更时触发：按当前覆盖范围尽力处理，未覆盖的点被
            // interpolatePose 跳过，行为退化为旧版"到即处理"，不无限积压
            const bool timed_out = (ros::Time::now() - front.arrival_wall_time).toSec() > pending_timeout_sec_;
            if (!covered && !timed_out)
                return;
            if (timed_out && !covered)
            {
                ROS_WARN_THROTTLE(1.0, "[hand_lio] odom did not cover lidar frame end within %.2fs, processing partially",
                                  pending_timeout_sec_);
            }
            PendingFrame frame = pending_frames_.front();
            pending_frames_.pop_front();
            processFrame(frame);
        }
    }

    void HandLioNode::virtualObstacleCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        // navibot 只发 xyz float32（见它的 ros_bridge.publish_virtual_obstacles），
        // 但别假设字段顺序/步长，按 fields 里的 offset 取，多一个 intensity 之类也不会错。
        int off_x = -1, off_y = -1, off_z = -1;
        int off_nx = -1, off_ny = -1, off_nz = -1;
        for (const auto &f : msg->fields)
        {
            if (f.datatype != sensor_msgs::PointField::FLOAT32)
                continue;
            if (f.name == "x")
                off_x = static_cast<int>(f.offset);
            else if (f.name == "y")
                off_y = static_cast<int>(f.offset);
            else if (f.name == "z")
                off_z = static_cast<int>(f.offset);
            else if (f.name == "normal_x")
                off_nx = static_cast<int>(f.offset);
            else if (f.name == "normal_y")
                off_ny = static_cast<int>(f.offset);
            else if (f.name == "normal_z")
                off_nz = static_cast<int>(f.offset);
        }
        const bool has_normal = (off_nx >= 0 && off_ny >= 0 && off_nz >= 0);
        if (off_x < 0 || off_y < 0 || off_z < 0 || msg->point_step == 0)
        {
            ROS_WARN_THROTTLE(1.0, "[hand_lio] virtual obstacle cloud has no float32 x/y/z, ignore");
            return;
        }

        // width/height 和各字段 offset 全来自消息本身，必须当不可信输入校验：
        // 越界的 offset 会读到下一个点甚至缓冲区外，而 data 短于声明的点数时
        // base 直接跑飞。上面只校验了"有没有 x/y/z"，挡不住这两种。
        const int max_off = std::max({off_x, off_y, off_z,
                                      has_normal ? off_nx : 0,
                                      has_normal ? off_ny : 0,
                                      has_normal ? off_nz : 0});
        if (max_off + 4 > static_cast<int>(msg->point_step))
        {
            ROS_WARN_THROTTLE(1.0,
                              "[hand_lio] virtual obstacle cloud field offset %d+4 exceeds point_step %u, ignore",
                              max_off, msg->point_step);
            return;
        }

        size_t n = static_cast<size_t>(msg->width) * msg->height;
        const size_t n_fit = msg->data.size() / msg->point_step;
        if (n > n_fit)
        {
            ROS_WARN_THROTTLE(1.0,
                              "[hand_lio] virtual obstacle cloud claims %zu points but data holds only %zu, truncating",
                              n, n_fit);
            n = n_fit;
        }

        std::vector<VirtualObstaclePoint> pts;
        pts.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            const uint8_t *base = msg->data.data() + i * msg->point_step;
            VirtualObstaclePoint vp;
            std::memcpy(&vp.p.x(), base + off_x, 4);
            std::memcpy(&vp.p.y(), base + off_y, 4);
            std::memcpy(&vp.p.z(), base + off_z, 4);
            if (!vp.p.allFinite())
                continue;
            if (has_normal)
            {
                std::memcpy(&vp.n.x(), base + off_nx, 4);
                std::memcpy(&vp.n.y(), base + off_ny, 4);
                std::memcpy(&vp.n.z(), base + off_nz, 4);
                if (!vp.n.allFinite())
                    vp.n.setZero();
            }
            pts.push_back(vp);
        }
        if (!has_normal && n > 0)
        {
            // 没有法向就没法做背面剔除, 整圈墙会全注入 —— 射向远侧墙面的光束会
            // 把近侧的墙投票投没(见 appendVirtualObstacles)。这时候宁可全注入也
            // 不要全丢: 至少禁行区还有一部分能挡住, 但要吵一点让人发现版本对不上。
            ROS_WARN_THROTTLE(5.0,
                              "[hand_lio] virtual obstacle cloud has no normal_x/y/z, backface culling disabled "
                              "(navibot too old?)");
        }

        // frame_id 必须是世界系：这些点直接就当 map 系点用，不做任何变换。对不上
        // 就整批丢掉——静默接受会在地图上凭空造出一堵位置完全错误的墙。
        if (!msg->header.frame_id.empty() && msg->header.frame_id != world_frame_id_)
        {
            ROS_WARN_THROTTLE(5.0,
                              "[hand_lio] virtual obstacle frame_id '%s' != world_frame_id '%s', ignore this cloud",
                              msg->header.frame_id.c_str(), world_frame_id_.c_str());
            return;
        }

        const size_t n_kept = pts.size();
        auto shared = std::make_shared<const std::vector<VirtualObstaclePoint>>(std::move(pts));
        {
            std::lock_guard<std::mutex> lock(virtual_obstacle_mutex_);
            virtual_obstacle_pts_ = std::move(shared);
        }
        // 空点云是正常状态（没有禁行区 / 全删了），照收不误，把上一批清掉。
        ROS_INFO("[hand_lio] virtual obstacles updated: %zu points", n_kept);
    }

    void HandLioNode::appendVirtualObstacles(const Eigen::Vector3d &sensor_pos,
                                             pcl::PointCloud<pcl::PointXYZI> &cloud) const
    {
        std::shared_ptr<const std::vector<VirtualObstaclePoint>> cache;
        {
            std::lock_guard<std::mutex> lock(virtual_obstacle_mutex_);
            cache = virtual_obstacle_pts_;
        }
        if (!cache || cache->empty())
            return;
        const std::vector<VirtualObstaclePoint> &pts = *cache;
        const ros::WallTime t_begin = ros::WallTime::now();

        // ---- 1) 量程剔除 ----
        // 超过 grid_map 的 max_ray_length 的点它本来也不会当成有效命中来用, 发过去
        // 只是白占带宽。

        // ---- 2) 背面剔除 ----
        // **这一步不能省。** navibot 发的是整圈闭合的墙(它不知道机器人在哪)。原样
        // 全注入的话, 从传感器射向**远侧**墙面的那些光束会穿过**近侧**墙面所在的
        // 体素, 给近侧记上 miss —— grid_map 每个更新周期按
        // count_hit >= count_hit_and_miss - count_hit 投票(grid_map.cpp:664),
        // 自己的点把自己的墙投没了。真实的墙不会这样: 雷达根本看不见墙背面。
        //
        // 判据就是经典的背面剔除: 法向朝着传感器(dot > 0)才留。navibot 给每个点带
        // 了朝外法向(见它的 virtual_obstacles._outward_normals)。
        //
        // 试过按 (方位角, 俯仰角) 分桶做深度缓冲, 不行: 桶要比墙面采样的角间距粗
        // 才挡得住, 而那个角间距随距离变 —— 2m 处 0.025m 的采样张 0.72°, 比雷达
        // 自己的角分辨率还粗, 光束直接从近侧点之间漏过去。合成数据上实测远侧墙
        // 192 个点一个没剔掉。背面剔除没有这个尺度问题, 也不需要跨仓库对齐常数。
        //
        // 残留的情况(凹多边形自遮挡、一块挡住另一块)只会多注入一点点, 方向是
        // "墙更结实", 不会反过来把墙投没。
        std::vector<size_t> visible;
        std::vector<double> range2(pts.size(), 0.0);
        visible.reserve(pts.size());
        const double max_r2 = virtual_obstacle_max_range_ * virtual_obstacle_max_range_;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            const Eigen::Vector3d to_sensor = sensor_pos - pts[i].p.cast<double>();
            const double r2 = to_sensor.squaredNorm();
            range2[i] = r2;
            if (r2 > max_r2 || r2 < 1e-4)
                continue;
            // 法向是零向量(navibot 没给 / 退化多边形)时不剔除, 宁可多注入
            if (pts[i].n.squaredNorm() > 1e-6 && pts[i].n.cast<double>().dot(to_sensor) <= 0.0)
                continue;
            visible.push_back(i);
        }
        if (visible.empty())
            return;

        // ---- 3) 按距离加密 ----
        // grid_map 的投票展开就是 `注入点数 >= 穿过该体素的真实光束数`。虚拟墙所在
        // 的位置现实里是空的，每帧都有真实光束穿过去打到后面的地面，每条贡献一个
        // miss。真实光束密度 ∝ 1/r²，所以注入份数也按 1/r² 来：density_gain 就是
        // "1m 处每个可见点复制几份"。
        //
        // 默认 12 的来历：Mid-360 约 2 万点/帧、FOV 360°x59°(≈5.7 sr) → 约 3500
        // 点/sr；一个 0.05m 的体素在 1m 处张 (0.05/1)² = 2.5e-3 sr，约 8.7 条光束
        // 穿过，取 12 留一点余量。**这是按包络算的，没在真机上量过** —— 上机第一
        // 件事就是看虚拟墙在 grid_map 里稳不稳（/grid_map/occupancy 里那堵墙会不会
        // 一闪一闪），不稳就把这个数调大。
        //
        // 复制点不加抖动：grid_map 的 setCacheOccupancy 是按点计数的，同一个坐标
        // 重复 N 次就是 N 个 hit，抖动没有额外收益，只会让 rviz 里看着更乱。
        // 先把份数算出来并一次性 reserve, 再 push。cloud 在 processFrame 里只按真实
        // 点数 reserve 过, 直接 push 四万个点会触发好几次扩容, 每次都把整个 buffer
        // (最后有 60000*32B ≈ 1.9MB)复制一遍 —— 实测这部分占了注入开销的一多半。
        std::vector<int> copies_of(visible.size(), 0);
        int planned = 0;
        for (size_t k = 0; k < visible.size() && planned < virtual_obstacle_max_points_; ++k)
        {
            const double r2 = std::max(range2[visible[k]], 1e-4);
            int c = static_cast<int>(std::lround(virtual_obstacle_density_gain_ / r2));
            c = std::min(std::max(c, 1), virtual_obstacle_max_copies_);
            c = std::min(c, virtual_obstacle_max_points_ - planned);
            copies_of[k] = c;
            planned += c;
        }
        cloud.reserve(cloud.size() + static_cast<size_t>(planned));

        int budget = virtual_obstacle_max_points_;
        pcl::PointXYZI out;
        out.intensity = 0.0f;
        for (size_t k = 0; k < visible.size(); ++k)
        {
            const size_t i = visible[k];
            const int copies = copies_of[k];
            if (copies <= 0)
                break;
            out.x = pts[i].p.x();
            out.y = pts[i].p.y();
            out.z = pts[i].p.z();
            for (int k = 0; k < copies; ++k)
                cloud.push_back(out);
            budget -= copies;
        }
        if (budget <= 0)
        {
            ROS_WARN_THROTTLE(5.0, "[hand_lio] virtual obstacle injection hit the %d-point budget, wall may be thinned",
                              virtual_obstacle_max_points_);
        }
        // 上机排查"墙没出现"时要能一眼看出是哪一环: 缓存里有多少、这一帧过了多少、
        // 注入了多少、花了多久。节流到 5s 一条, 正常跑的时候不吵。
        ROS_INFO_THROTTLE(5.0,
                          "[hand_lio] virtual obstacles: cached=%zu visible=%zu injected=%d took=%.2fms",
                          pts.size(), visible.size(), virtual_obstacle_max_points_ - budget,
                          (ros::WallTime::now() - t_begin).toSec() * 1e3);
    }

    void HandLioNode::packXYZI(const pcl::PointCloud<pcl::PointXYZI> &cloud,
                               sensor_msgs::PointCloud2 &out)
    {
        // 不用 pcl::toROSMsg：sizeof(pcl::PointXYZI) 是 32，它照搬内存布局，于是
        // point_step=32 而只声明 x@0 y@4 z@8 intensity@16 —— offset 12 那 4 字节
        // （PCL 的 data[3]，构造函数塞的 1.0f）和尾部 20..31 那 12 字节没有任何
        // 字段声明，也没有任何一行代码写过。实测尾部就是未初始化的堆内存，指针
        // 形状的值都能直接读出来：既让同一段输入两次跑出的字节不一样（没法做
        // 校验和比对），又把 12/32 的带宽花在没人要的字节上。
        // 这里手工打包成紧凑的 16 字节，两个问题一起消掉。
        out.fields.clear();
        out.fields.resize(4);
        const char *names[4] = {"x", "y", "z", "intensity"};
        for (int i = 0; i < 4; ++i)
        {
            out.fields[i].name = names[i];
            out.fields[i].offset = static_cast<uint32_t>(4 * i);
            out.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
            out.fields[i].count = 1;
        }
        out.height = 1;
        out.width = static_cast<uint32_t>(cloud.size());
        out.point_step = 16;
        out.row_step = out.point_step * out.width;
        out.is_bigendian = false;
        out.is_dense = cloud.is_dense;
        out.data.resize(static_cast<size_t>(out.row_step));
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            float v[4] = {cloud[i].x, cloud[i].y, cloud[i].z, cloud[i].intensity};
            std::memcpy(out.data.data() + i * 16, v, 16);
        }
    }

    void HandLioNode::processFrame(const PendingFrame &frame)
    {
        const CustomMsgConstPtr &msg = frame.msg;
        const int point_num = frame.point_num;   // 已夹过界，见 PendingFrame 的注释
        const double base_time = msg->header.stamp.toSec();

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

        // 用缓冲区里最新的样本做定位质量把关
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

            // 去畸变 + 转 map 系一步完成：用该点自己采集时刻的插值位姿直接变换
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

        // 把 navibot 圈的禁行区混进来，让 SCAN-Planner 的局部避障也看得见（它自己
        // 的实时栅格图不知道那些区域，沟这种负障碍更是压根没有占据体素）。
        //
        // **必须混在同一条消息里**，不能另开一路发到同一个话题：grid_map 的
        // cloudCallback 每次是从 proj_points_cnt = 0 开始**覆盖** md_.proj_points_
        // （不累加），而 updateOccupancyCallback 是定时器、每次只消费最近那一帧 ——
        // 另发一路会把真实雷达帧顶掉，是安全倒退。混在这里还顺带保证了 stamp 和
        // /grid_map/sensor_pose 天然对齐，不会有中继节点那种一帧错位。
        //
        // 放在 empty 判断之后：真实点一个都没有的帧本来就该丢，不能只靠虚拟点
        // 凭空造出一帧来。
        if (enable_virtual_obstacles_)
        {
            PoseSample pose_end;
            if (interpolatePose(frame.end_time, pose_end))
            {
                // 用雷达原点而不是 IMU 原点：可见性剔除是"从雷达看过去"的几何，
                // 两者差几厘米，算上也不费事。
                const Eigen::Vector3d lidar_origin = pose_end.q * imu_t_lidar_ + pose_end.p;
                appendVirtualObstacles(lidar_origin, cloud_world);
            }
        }

        cloud_world.width = cloud_world.size();
        cloud_world.height = 1;
        cloud_world.is_dense = true;

        sensor_msgs::PointCloud2 out_msg;
        packXYZI(cloud_world, out_msg);
        out_msg.header.stamp = ros::Time(frame.end_time);
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
