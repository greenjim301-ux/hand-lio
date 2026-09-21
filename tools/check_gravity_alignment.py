#!/usr/bin/env python3
"""检查 /latest_imu_odom 的 map 系是不是重力对齐的。

判据
----
设备静止时加速度计量的是比力（specific force），在 IMU 系里方向沿"上"。
把它用 /latest_imu_odom 报的姿态转到 world 系：

    a_world = world_R_imu * a_imu

若 map 系重力对齐（Z 轴与重力反向），a_world 必须指向 +Z。

注意两点：

1. map 系跟重力系差一个**航向角**（绕重力轴转）不影响判据，也不影响 hand_lio
   ——那仍然算重力对齐，a_world 照样指 +Z。
2. 单次静止测到 a_world ≈ +Z，就足以说明"此刻姿态是重力锁定的"。之所以还要
   换几个姿态各测一次，是为了排除另一种情况：姿态估计只在初始化那个姿态附近
   准，一动就飘。设备是要动着用的，所以这一点必须验。

用法
----
    rosrun hand_lio check_gravity_alignment.py
    # 或直接 ./check_gravity_alignment.py

跑起来之后：把设备静止放好 -> 出一行结果 -> 换一个明显不同的姿态（比如
俯仰 20°、横滚 30°、转个方向）再静止 -> 再出一行。攒够 3~4 个姿态后 Ctrl-C
看结论。
"""

import sys
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

ODOM_TOPIC = "/latest_imu_odom"
IMU_TOPIC = "/livox/imu"

STILL_GYRO_THRESH = 0.02      # [rad/s] 陀螺模长低于此认为静止
STILL_MIN_SEC = 1.0           # 连续静止多久算一个有效稳定段
PAIR_MAX_DT = 0.02            # [s] odom 与 imu 配对的最大时间差
ATTITUDE_NEW_DEG = 8.0        # 与已记录姿态差异超过这个角度才算"新姿态"
VERDICT_OK_DEG = 3.0          # 偏差小于此视为对齐


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def roll_pitch_deg(R):
    """从旋转矩阵取 roll/pitch，仅用于把姿态打出来看，不参与判据。"""
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch


def angle_between_deg(u, v):
    cu, cv = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(np.dot(cu, cv), -1.0, 1.0)))


class Checker(object):
    def __init__(self):
        self.odom_buf = []            # (t, R)
        self.still_since = None
        self.samples = []             # 当前稳定段内的 (R, a_imu)
        self.holds = []               # 每个稳定段一条结果
        self.last_t = 0.0
        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_cb, queue_size=200)
        rospy.Subscriber(IMU_TOPIC, Imu, self.imu_cb, queue_size=200)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.odom_buf.append((msg.header.stamp.to_sec(), quat_to_R(q.w, q.x, q.y, q.z)))
        if len(self.odom_buf) > 2000:
            self.odom_buf = self.odom_buf[-1000:]

    def nearest_odom(self, t):
        if not self.odom_buf:
            return None
        i = int(np.argmin([abs(ot - t) for ot, _ in self.odom_buf]))
        ot, R = self.odom_buf[i]
        return R if abs(ot - t) <= PAIR_MAX_DT else None

    def imu_cb(self, msg):
        t = msg.header.stamp.to_sec()
        g = msg.angular_velocity
        a = msg.linear_acceleration
        gyro = np.linalg.norm([g.x, g.y, g.z])
        a_imu = np.array([a.x, a.y, a.z])

        if gyro > STILL_GYRO_THRESH or np.linalg.norm(a_imu) < 1e-6:
            self.try_finish_hold()
            self.still_since = None
            self.samples = []
            return

        R = self.nearest_odom(t)
        if R is None:
            return
        if self.still_since is None:
            self.still_since = t
        self.last_t = t
        self.samples.append((R, a_imu))

    def try_finish_hold(self):
        """静止时长够 STILL_MIN_SEC 才算一个有效稳定段。"""
        if self.still_since is None or len(self.samples) < 10:
            return
        if self.last_t - self.still_since < STILL_MIN_SEC:
            return
        self.finish_hold()

    def finish_hold(self):
        Rs = np.array([s[0] for s in self.samples])
        As = np.array([s[1] for s in self.samples])
        R_mean = Rs.mean(axis=0)
        a_world = np.array([R.dot(a) for R, a in self.samples]).mean(axis=0)
        dev = angle_between_deg(a_world, np.array([0.0, 0.0, 1.0]))
        roll, pitch = roll_pitch_deg(R_mean)
        mag = np.linalg.norm(As, axis=1).mean()

        # 姿态跟已有的太接近就不算新姿态，提醒用户换一个
        is_new = all(angle_between_deg(R_mean[:, 2], h["z_axis"]) > ATTITUDE_NEW_DEG
                     for h in self.holds)
        self.holds.append({"roll": roll, "pitch": pitch, "dev": dev,
                           "mag": mag, "n": len(self.samples),
                           "z_axis": R_mean[:, 2], "new": is_new})
        tag = "" if is_new else "   <- 姿态与之前某次接近，换个明显不同的姿态再测一次"
        print("稳定段 #%d  设备姿态 roll=%+6.1f° pitch=%+6.1f°   "
              "世界系重力偏离 +Z: %5.2f°   |a|=%.3f  (%d 样本)%s"
              % (len(self.holds), roll, pitch, dev, mag, len(self.samples), tag))
        sys.stdout.flush()
        self.samples = []
        self.still_since = None

    def verdict(self):
        print("\n" + "=" * 78)
        if not self.holds:
            print("没有采到任何稳定段。确认两个话题都在发，且设备真的静止过 >= %.1fs。"
                  % STILL_MIN_SEC)
            return
        devs = [h["dev"] for h in self.holds]
        n_new = sum(1 for h in self.holds if h["new"])
        spread = 0.0
        for i in range(len(self.holds)):
            for j in range(i + 1, len(self.holds)):
                spread = max(spread, angle_between_deg(self.holds[i]["z_axis"],
                                                       self.holds[j]["z_axis"]))
        print("共 %d 个稳定段，其中 %d 个姿态互不相同；姿态最大差异 %.1f°"
              % (len(self.holds), n_new, spread))
        print("重力偏离 +Z：最小 %.2f°  最大 %.2f°  均值 %.2f°"
              % (min(devs), max(devs), float(np.mean(devs))))
        mags = [h["mag"] for h in self.holds]
        print("加速度模长均值 %.3f（≈1 说明 Livox 报的是 g，≈9.81 是 m/s²，不影响判据）"
              % float(np.mean(mags)))
        print("-" * 78)
        one_attitude = spread < ATTITUDE_NEW_DEG
        if max(devs) < VERDICT_OK_DEG:
            print("结论：/latest_imu_odom 的 map 系**是重力对齐的**"
                  "（最大偏离 %.2f° < %.1f°）。" % (max(devs), VERDICT_OK_DEG))
            if one_attitude:
                print("      但只测到一个姿态（差异仅 %.1f°），无法排除'姿态一变就飘'。" % spread)
                print("      请把设备摆成俯仰/横滚各 20° 以上的几个姿态再各测一次。")
            else:
                print("      跨 %d 个姿态、差异达 %.1f°，重力方向始终锁在 +Z。"
                      % (len(self.holds), spread))
        else:
            # 偏差恒定 = map 系整体歪了；偏差随姿态变 = 姿态估计本身不可靠
            drift = max(devs) - min(devs)
            print("结论：/latest_imu_odom **不满足重力对齐**，最大偏离 %.2f°。" % max(devs))
            if one_attitude:
                print("      只测了一个姿态，还分不清是 map 系整体歪了、还是姿态估计不准；")
                print("      换几个姿态再测一次就能区分。")
            elif drift < VERDICT_OK_DEG:
                print("      偏离在各姿态下基本恒定（%.2f°~%.2f°）-> map 系整体相对重力"
                      "歪了约 %.1f°，" % (min(devs), max(devs), float(np.mean(devs))))
                print("      是个固定倾斜，可以用一个常量旋转补偿。")
            else:
                print("      偏离随姿态变化（%.2f°~%.2f°，波动 %.2f°）-> 姿态估计本身"
                      "没有锁住重力，" % (min(devs), max(devs), drift))
                print("      补一个常量旋转救不了，得从定位那端查。")
            print("      无论哪种，hand_lio 输出的点云都不会是水平的，")
            print("      imu_R_lidar 也就不能简单留单位阵了。")
        print("=" * 78)


def main():
    rospy.init_node("check_gravity_alignment", anonymous=True)
    print(__doc__)
    print("监听 %s 与 %s ...\n" % (ODOM_TOPIC, IMU_TOPIC))
    c = Checker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        c.try_finish_hold()
        c.verdict()


if __name__ == "__main__":
    main()
