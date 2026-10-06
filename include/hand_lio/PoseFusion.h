/*
 * PoseFusion —— 用 /lidar_pose 做锚点、机体速度做前推的位姿估计（纯算法，不依赖 ROS）。
 *
 * 背景（见 nav_analysis/20261001_evening 三条 bag 的分析）：/latest_imu_odom 的“突变”
 * 99% 是 /lidar_pose 到达时的修正，坏的是两次修正之间的前推——楼梯上它的 z 向速度
 * 能偏到 -1~-1.5 m/s，/lidar_pose 延迟超过约 1 s 时它干脆不前推。这里保留同一个
 * 锚点，只把前推换掉：
 *
 *   锚点位姿 = 最新一条 /lidar_pose（换算到机体） + 从它的消息头时刻到现在的速度积分
 *   输出位姿 = 跟着同一份速度往前走，再以时间常数 tau 向锚点位姿靠拢
 *
 * 第二步是为了把每次 /lidar_pose 到达时的修正量摊开，输出里不留台阶。
 *
 * 水平用机体系速度 (vx, vy) 前推；z 没有速度源，按最近一段 /lidar_pose 的坡度 (dz/ds)
 * 乘以前推的水平路程。
 *
 * 航向两种来源（yaw_from_stream）：
 *   false  用速度源的 wz 积分。
 *   true   用 200Hz 高频流（/latest_imu_odom，陀螺）自己的航向增量。10-06 两段 M20S
 *          楼梯实录上，它前推 1.4 s 的航向误差 90 分位 1.3°，wz 积分是 4~5°。
 *          但高频流有“位置保持”模式（/lidar_pose 延迟超过约 1 s 时，10-01 下楼那条
 *          61% 的帧位置不变），保持帧里航向也一起冻住。所以只用“新鲜”样本（位置或
 *          航向变了的），两条新鲜样本隔得超过 stream_gap_max 时那一小段退回 wz。
 *          stream_yaw_rate_max 只挡离谱的毛刺，**不要调低**：高频流的机体航向有几十毫秒
 *          内来回 3~4° 的快速抖动（折算 4~12 rad/s），而且是来回的。增量能首尾相消，
 *          锚点航向才等于“/lidar_pose 航向 + 期间真实转角”；按 3 rad/s 剔掉抖动的一半会
 *          破坏相消，实测（10-06 两段）航向误差 90 分位从 1.3~1.5° 变差到 2.1~3.2°。
 *
 * 修不了的：/lidar_pose 自身的跳变（地图重影/重定位）。判到跳变时标记 jump 并让输出
 * 直接跟过去，不做平滑——把 1 m 的台阶摊成 0.3 s 的“运动”只会造出一段 3 m/s 的假速度。
 * 判据水平和高度分开：
 *   水平：新锚点与前推预测相差超过 jump_threshold。速度前推在水平上很准（三条实录
 *         bag 回放，这个差的 90 分位 0.07 m、最大 0.16 m），差得多就是 /lidar_pose 跳了。
 *   高度：相邻两条 /lidar_pose 的 z 之差超过 jump_threshold + max_vz * 间隔。**不能**
 *         用“与预测的差”：z 是靠坡度外推的，延迟 1 s 以上时外推自己就能错 0.5 m
 *         （实录里出现过一次 -0.58 m），会把自己的误差当成跳变。
 *
 * 所有时间是同一条时钟上的秒（节点里是 ros::Time）。
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <deque>

namespace hand_lio {

class PoseFusion {
public:
    struct Params {
        double tau = 0.3;              // [s] 输出向锚点靠拢的时间常数
        double velocity_timeout = 0.5; // [s] 一个速度样本最多用多久，之后按零速
        double scale_vx = 1.0;         // 速度源到实际机体速度的比例
        double scale_vy = 1.0;
        double scale_wz = 1.0;
        double slope_window = 1.2;     // [s] 估坡度用最近多长一段 /lidar_pose
        double slope_min_ds = 0.15;    // [m] 这段水平位移小于它就不估坡度（按 0）
        double slope_max = 0.8;        // |dz/ds| 的上限
        double jump_threshold = 0.5;   // [m] /lidar_pose 跳变判据，见文件头
        double max_vz = 0.5;           // [m/s] 高度判据里允许的最大真实升降速度
        double buffer_horizon = 5.0;   // [s] 速度样本保留多久，要盖住 /lidar_pose 的最大延迟
        bool yaw_from_stream = false;  // 航向用高频流增量而不是 wz 积分，见文件头
        double stream_gap_max = 0.05;  // [s] 两条新鲜高频流样本最大间隔，超过按断流处理
        double stream_yaw_rate_max = 20.0; // [rad/s] 增量折算角速度超过它当毛刺，不用。别调低，见文件头
    };

    // 机体位姿里参与前推的部分；横滚/俯仰由节点自己取
    struct Pose {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double yaw = 0.0;
    };

    struct AnchorResult {
        bool first = false;      // 第一条锚点，没有可比的预测
        bool jump = false;
        double age = 0.0;        // 到达时刻 - 消息头时刻 [s]
        bool age_exceeds_buffer = false; // 延迟超过速度缓冲，前推不完整
        double dx = 0.0;         // 新锚点 - 前推预测（世界系）
        double dy = 0.0;
        double dz = 0.0;
        double dyaw = 0.0;
        double slope = 0.0;
    };

    explicit PoseFusion(const Params& params) : p_(params) {}

    static double wrapAngle(double a)
    {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a < -M_PI)
            a += 2.0 * M_PI;
        return a;
    }

    // 机体系速度样本，t 是它开始生效的时刻。时间倒退的样本丢弃。
    void addVelocity(double t, double vx, double vy, double wz)
    {
        if (!vel_.empty() && t < vel_.back().t)
            return;
        vel_.push_back({t, p_.scale_vx * vx, p_.scale_vy * vy, p_.scale_wz * wz});
        // 留一条早于窗口的样本，窗口起点处才查得到速度
        while (vel_.size() > 2 && vel_[1].t < t - p_.buffer_horizon)
            vel_.pop_front();
    }

    // 高频流（/latest_imu_odom 换算到机体后）的航向样本。fresh=false 表示这条跟上一条
    // 位置、航向都一样（位置保持帧），不能当作“这段时间没转”来用。
    void addStreamYaw(double t, double yaw, bool fresh)
    {
        if (!fresh || (!yaw_buf_.empty() && t <= yaw_buf_.back().t))
            return;
        double unwrapped = yaw;
        if (!yaw_buf_.empty())
            unwrapped = yaw_buf_.back().yaw + wrapAngle(yaw - last_stream_yaw_raw_);
        last_stream_yaw_raw_ = yaw;
        yaw_buf_.push_back({t, unwrapped});
        while (yaw_buf_.size() > 2 && yaw_buf_[1].t < t - p_.buffer_horizon)
            yaw_buf_.pop_front();
    }

    // 把锚点和输出都推进到 t_now。还没有锚点时什么都不做。
    void advance(double t_now)
    {
        if (!initialized_ || t_now <= last_t_)
            return;
        const double dt = t_now - last_t_;
        integrate(anchor_, last_t_, t_now);
        integrate(out_, last_t_, t_now);
        const double a = p_.tau > 1e-6 ? 1.0 - std::exp(-dt / p_.tau) : 1.0;
        out_.x += a * (anchor_.x - out_.x);
        out_.y += a * (anchor_.y - out_.y);
        out_.z += a * (anchor_.z - out_.z);
        out_.yaw = wrapAngle(out_.yaw + a * wrapAngle(anchor_.yaw - out_.yaw));
        last_t_ = t_now;
    }

    // 一条 /lidar_pose（已换算到机体）在 t_now 到达，消息头时刻 t_stamp。
    AnchorResult onLidarPose(double t_now, double t_stamp, const Pose& body)
    {
        AnchorResult r;
        advance(t_now);

        r.age = std::max(0.0, t_now - t_stamp);
        r.age_exceeds_buffer = vel_.empty() || vel_.front().t > t_stamp;

        // 坡度：窗口内最早一条到这一条
        while (!hist_.empty() && (hist_.front().t < t_stamp - p_.slope_window || hist_.front().t >= t_stamp))
            hist_.pop_front();
        if (!hist_.empty())
        {
            const double ds = std::hypot(body.x - hist_.front().pose.x, body.y - hist_.front().pose.y);
            if (ds > p_.slope_min_ds)
                slope_ = std::max(-p_.slope_max, std::min(p_.slope_max, (body.z - hist_.front().pose.z) / ds));
            else
                slope_ = 0.0;
        }
        hist_.push_back({t_stamp, body});

        Pose anchor = body;
        integrate(anchor, t_now - r.age, t_now);

        if (!initialized_)
        {
            r.first = true;
        }
        else
        {
            r.dx = anchor.x - anchor_.x;
            r.dy = anchor.y - anchor_.y;
            r.dz = anchor.z - anchor_.z;
            r.dyaw = wrapAngle(anchor.yaw - anchor_.yaw);
            const double dt_lidar = std::max(0.0, t_stamp - prev_lidar_t_);
            r.jump = std::hypot(r.dx, r.dy) > p_.jump_threshold ||
                     std::abs(body.z - prev_lidar_z_) > p_.jump_threshold + p_.max_vz * dt_lidar;
        }
        prev_lidar_t_ = t_stamp;
        prev_lidar_z_ = body.z;

        if (r.jump)
        {
            // 跳变前后的 /lidar_pose 不在同一个高度基准上，不能拿来估坡度
            hist_.clear();
            hist_.push_back({t_stamp, body});
            slope_ = 0.0;
            anchor = body;
            integrate(anchor, t_now - r.age, t_now);
        }

        anchor_ = anchor;
        if (r.first || r.jump)
            out_ = anchor_;
        initialized_ = true;
        last_t_ = t_now;
        r.slope = slope_;
        return r;
    }

    void reset()
    {
        initialized_ = false;
        vel_.clear();
        hist_.clear();
        yaw_buf_.clear();
        slope_ = 0.0;
    }

    bool initialized() const { return initialized_; }
    const Pose& pose() const { return out_; }
    const Pose& anchor() const { return anchor_; }
    double lastTime() const { return last_t_; }

    // t 时刻生效的机体系速度（已乘比例）；超时或没有样本时为零，返回 false
    bool velocityAt(double t, double& vx, double& vy, double& wz) const
    {
        vx = vy = wz = 0.0;
        const int i = indexAt(t);
        if (i < 0 || t - vel_[i].t >= p_.velocity_timeout)
            return false;
        vx = vel_[i].vx;
        vy = vel_[i].vy;
        wz = vel_[i].wz;
        return true;
    }

private:
    struct VelSample {
        double t, vx, vy, wz;
    };
    struct LidarSample {
        double t;
        Pose pose;
    };
    struct YawSample {
        double t, yaw;  // yaw 已展开（不回绕）
    };

    // 高频流在 [t0, t1] 上的航向增量。区间要被新鲜样本覆盖、样本间隔不超过
    // stream_gap_max、折算角速度不超过 stream_yaw_rate_max，否则返回 false。
    bool streamYawDelta(double t0, double t1, double& d) const
    {
        if (yaw_buf_.size() < 2 || t0 < yaw_buf_.front().t || t1 > yaw_buf_.back().t || t1 <= t0)
            return false;
        auto cmp = [](const YawSample& s, double v) { return s.t < v; };
        auto hi0 = std::lower_bound(yaw_buf_.begin(), yaw_buf_.end(), t0, cmp);
        auto hi1 = std::lower_bound(hi0, yaw_buf_.end(), t1, cmp);
        auto lo0 = hi0 == yaw_buf_.begin() ? hi0 : hi0 - 1;
        for (auto it = lo0; it != hi1; ++it)
        {
            if ((it + 1)->t - it->t > p_.stream_gap_max)
                return false;
        }
        auto at = [&](double t, std::deque<YawSample>::const_iterator hi) {
            if (hi == yaw_buf_.begin() || hi->t <= t)
                return hi->yaw;
            const auto lo = hi - 1;
            return lo->yaw + (hi->yaw - lo->yaw) * (t - lo->t) / (hi->t - lo->t);
        };
        d = at(t1, hi1) - at(t0, hi0);
        return std::abs(d) <= p_.stream_yaw_rate_max * (t1 - t0);
    }

    // 最后一条 t <= 查询时刻的样本下标，没有则 -1
    int indexAt(double t) const
    {
        auto it = std::upper_bound(vel_.begin(), vel_.end(), t,
                                   [](double v, const VelSample& s) { return v < s.t; });
        return static_cast<int>(it - vel_.begin()) - 1;
    }

    // 用缓冲里的速度把 pose 从 t0 积分到 t1：样本零阶保持，过了 velocity_timeout 按零速。
    // z 按当前坡度乘以水平路程。航向用高频流增量时按 5ms 小步走，跟上 200Hz 的航向。
    void integrate(Pose& pose, double t0, double t1) const
    {
        constexpr double kStreamStep = 0.005;
        double t = t0;
        int i = indexAt(t);
        const int n = static_cast<int>(vel_.size());
        while (t < t1)
        {
            double seg_end = t1;
            if (i + 1 < n)
                seg_end = std::min(seg_end, vel_[i + 1].t);
            bool moving = false;
            if (i >= 0)
            {
                const double expire = vel_[i].t + p_.velocity_timeout;
                if (t < expire)
                {
                    moving = true;
                    seg_end = std::min(seg_end, expire);
                }
            }
            if (p_.yaw_from_stream)
            {
                seg_end = std::min(seg_end, t + kStreamStep);
                // 最新一条高频流样本总比“现在”早一点（实测约 1ms）。在它那里切开：覆盖到的
                // 部分用高频流增量，只有最后这一小截退回 wz。不切的话整步都因为没覆盖而退回，
                // 每一拍推进到“现在”时高频流航向就一点都用不上。
                if (!yaw_buf_.empty() && t < yaw_buf_.back().t && yaw_buf_.back().t < seg_end)
                    seg_end = yaw_buf_.back().t;
            }
            const double dt = seg_end - t;
            if (dt > 0.0)
            {
                // 速度源超时时机体按静止算，但航向照样跟高频流走（原地转时也可能断了速度源）
                double dyaw = moving ? vel_[i].wz * dt : 0.0;
                double d = 0.0;
                if (p_.yaw_from_stream && streamYawDelta(t, seg_end, d))
                    dyaw = d;
                if (moving)
                {
                    const VelSample& s = vel_[i];
                    const double yaw_mid = pose.yaw + 0.5 * dyaw;
                    const double c = std::cos(yaw_mid);
                    const double sn = std::sin(yaw_mid);
                    pose.x += (c * s.vx - sn * s.vy) * dt;
                    pose.y += (sn * s.vx + c * s.vy) * dt;
                    pose.z += slope_ * std::hypot(s.vx, s.vy) * dt;
                }
                pose.yaw = wrapAngle(pose.yaw + dyaw);
            }
            t = seg_end;
            if (i + 1 < n && vel_[i + 1].t <= t)
                ++i;
        }
    }

    Params p_;
    std::deque<VelSample> vel_;
    std::deque<LidarSample> hist_;
    std::deque<YawSample> yaw_buf_;
    double last_stream_yaw_raw_ = 0.0;
    double slope_ = 0.0;
    double prev_lidar_t_ = 0.0; // 上一条 /lidar_pose 的消息头时刻和机体 z
    double prev_lidar_z_ = 0.0;
    bool initialized_ = false;
    double last_t_ = 0.0;
    Pose anchor_;
    Pose out_;
};

} // namespace hand_lio
