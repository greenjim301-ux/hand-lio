#!/usr/bin/env python3
"""测量 /latest_imu_odom 位姿台阶的步进大小——即运动拖影的直接预测量。

背景：/latest_imu_odom 以 200Hz 发布但内容只有 ~10Hz 更新（台阶状）。hand_lio
逐点插值时，帧内点拿到的是"保持中"的旧位姿，所以单帧点云的拖影量 ≈ 相邻两次
位姿更新之间设备实际移动的距离/转角。本脚本对流去重（只统计位姿发生变化的
时刻），打印每次更新的平移步进和旋转步进分布：

    step_t：相邻两个不同位姿之间的平移距离
    step_r：相邻两个不同位姿之间的旋转角
    hold  ：同一位姿被重复发布的持续时长（台阶宽度）

用法（不挑环境，静止/行走/原地转身各测一段）：

    python3 tools/measure_odom_steps.py
    python3 tools/measure_odom_steps.py --odom /latest_imu_odom --report 5

判读：运动段的 step_t / step_r 就是对应运动下单帧点云的拖影上限。
step_t 中位数到分米级、或 step_r 使远处（乘上到墙距离）超过几厘米，
拖影就值得修（odom 去重 + 真插值）。Ctrl+C 打印最终汇总。
"""

import argparse
import math
import signal
import sys
import threading
import time

import rospy
from nav_msgs.msg import Odometry


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def fmt_stats(name, vals, scale, unit):
    if not vals:
        return "  %-6s: (无样本)" % name
    v = sorted(x * scale for x in vals)
    return "  %-6s: n=%-5d min=%7.2f%s  median=%7.2f%s  mean=%7.2f%s  p95=%7.2f%s  max=%7.2f%s" % (
        name, len(v), v[0], unit, percentile(v, 50), unit, sum(v) / len(v), unit,
        percentile(v, 95), unit, v[-1], unit)


class StepMeter(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.msg_count = 0
        self.last_pose = None          # (x,y,z,qx,qy,qz,qw)
        self.last_change_stamp = None  # 上一次位姿变化时的消息时间戳
        self.start_mono = time.monotonic()
        self.step_t = []  # 平移步进 [m]
        self.step_r = []  # 旋转步进 [rad]
        self.holds = []   # 台阶宽度 [s]

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            self.msg_count += 1
            if self.last_pose is None:
                self.last_pose = pose
                self.last_change_stamp = stamp
                return
            if pose == self.last_pose:
                return
            lp = self.last_pose
            dt = math.sqrt((p.x - lp[0]) ** 2 + (p.y - lp[1]) ** 2 + (p.z - lp[2]) ** 2)
            # 四元数夹角: theta = 2*acos(|<q1,q2>|)
            dot = abs(lp[3] * q.x + lp[4] * q.y + lp[5] * q.z + lp[6] * q.w)
            dr = 2.0 * math.acos(min(1.0, dot))
            self.step_t.append(dt)
            self.step_r.append(dr)
            self.holds.append(stamp - self.last_change_stamp)
            self.last_pose = pose
            self.last_change_stamp = stamp

    def report(self, final=False):
        with self.lock:
            elapsed = time.monotonic() - self.start_mono
            lines = [
                "%s====== 位姿台阶统计 (运行 %.0fs) ======" % ("\n最终" if final else "\n", elapsed),
                "  消息数=%d (%.1fHz)  位姿更新数=%d (%.1fHz)" % (
                    self.msg_count, self.msg_count / elapsed if elapsed > 0 else 0.0,
                    len(self.step_t), len(self.step_t) / elapsed if elapsed > 0 else 0.0),
                fmt_stats("step_t", self.step_t, 1000.0, "mm"),
                fmt_stats("step_r", self.step_r, 180.0 / math.pi, "deg"),
                fmt_stats("hold", self.holds, 1000.0, "ms"),
                "  (step_t/step_r 即该运动状态下单帧点云的拖影上限；step_r 乘上到目标距离折算成位移)",
            ]
        print("\n".join(lines))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odom", default="/latest_imu_odom", help="nav_msgs/Odometry topic")
    parser.add_argument("--report", type=float, default=10.0, help="周期性打印间隔 [s]")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("measure_odom_steps", anonymous=True)
    meter = StepMeter()
    rospy.Subscriber(args.odom, Odometry, meter.odom_cb, queue_size=2000)
    rospy.loginfo("listening: %s", args.odom)

    def on_sigint(_sig, _frame):
        meter.report(final=True)
        rospy.signal_shutdown("done")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    rate = rospy.Rate(1.0 / max(args.report, 0.1))
    while not rospy.is_shutdown():
        rate.sleep()
        meter.report()


if __name__ == "__main__":
    main()
