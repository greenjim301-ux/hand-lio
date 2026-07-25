#!/usr/bin/env python3
"""测量 /latest_imu_odom 相对 /livox/lidar 的滞后。

针对 hand_lio 延迟处理设计关心的两个量，逐帧统计并打印分布：

1. gap（时间戳域缺口）：lidar 帧到达那一刻，帧尾采集时刻超出当时最新 odom
   时间戳多少。gap > 0 的部分就是旧版"到即处理"设计会丢掉的帧尾时长。
2. wait（墙钟等待）：从 lidar 帧到达，到第一条 stamp >= 帧尾时刻的 odom 到达，
   实际要等多久。这正是 hand_lio 待处理队列引入的额外延迟。

lidar 侧用 rospy.AnyMsg 直接解析 livox CustomMsg 的原始字节（只取 header.stamp
和最后一个点的 offset_time），不依赖任何消息定义包，source 不 source devel 都能跑：

    python3 tools/measure_odom_lag.py
    python3 tools/measure_odom_lag.py --odom /latest_imu_odom --lidar /livox/lidar --report 5

运行期间每 --report 秒打印一次累计统计，Ctrl+C 退出时打印最终汇总。
"""

import argparse
import signal
import struct
import sys
import threading
import time

import rospy
from nav_msgs.msg import Odometry

CUSTOM_POINT_BYTES = 19  # uint32 offset_time + 3x float32 xyz + uint8 reflectivity/tag/line


def parse_custom_msg(buff):
    """从 livox_ros_driver(/2) CustomMsg 的序列化字节中取 (base_time_sec, end_offset_sec)。

    布局: Header(uint32 seq, uint32 secs, uint32 nsecs, uint32 len + frame_id)
          + uint64 timebase + uint32 point_num + uint8 lidar_id + uint8[3] rsvd
          + uint32 array_len + CustomPoint[array_len]
    """
    _seq, secs, nsecs, frame_id_len = struct.unpack_from("<IIII", buff, 0)
    off = 16 + frame_id_len
    off += 8 + 4 + 1 + 3  # timebase + point_num + lidar_id + rsvd
    (array_len,) = struct.unpack_from("<I", buff, off)
    off += 4
    end_offset_ns = 0
    if array_len > 0:
        (end_offset_ns,) = struct.unpack_from("<I", buff, off + (array_len - 1) * CUSTOM_POINT_BYTES)
    return secs + nsecs * 1e-9, end_offset_ns * 1e-9

# pending 帧超过该墙钟时长仍未被 odom 覆盖，视为"从未覆盖"（odom 断流），不再计入 wait 分布
NEVER_COVERED_SEC = 5.0


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def fmt_stats(name, vals_sec):
    if not vals_sec:
        return "  %-6s: (无样本)" % name
    v = sorted(x * 1000.0 for x in vals_sec)  # -> ms
    return "  %-6s: n=%-5d min=%6.1fms  median=%6.1fms  mean=%6.1fms  p95=%6.1fms  max=%6.1fms" % (
        name, len(v), v[0], percentile(v, 50), sum(v) / len(v), percentile(v, 95), v[-1])


class LagMeter(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.newest_odom_stamp = None
        self.odom_count = 0
        self.lidar_count = 0
        self.start_mono = time.monotonic()
        self.gaps = []      # 时间戳域缺口 [s]，可为负（到达时已被覆盖）
        self.waits = []     # 墙钟等待 [s]
        self.pending = []   # [(end_stamp, arrival_mono), ...] 尚未被 odom 覆盖的帧
        self.never_covered = 0

    def odom_cb(self, msg):
        t = msg.header.stamp.to_sec()
        now = time.monotonic()
        with self.lock:
            self.odom_count += 1
            if self.newest_odom_stamp is None or t > self.newest_odom_stamp:
                self.newest_odom_stamp = t
            still_pending = []
            for end_stamp, arrival in self.pending:
                if t >= end_stamp:
                    self.waits.append(now - arrival)
                elif now - arrival > NEVER_COVERED_SEC:
                    self.never_covered += 1
                else:
                    still_pending.append((end_stamp, arrival))
            self.pending = still_pending

    def lidar_cb(self, msg):
        try:
            base, end_offset = parse_custom_msg(msg._buff)
        except struct.error:
            rospy.logwarn_throttle(5.0, "lidar 消息字节解析失败，话题类型可能不是 livox CustomMsg")
            return
        end = base + end_offset
        now = time.monotonic()
        with self.lock:
            self.lidar_count += 1
            if self.newest_odom_stamp is not None:
                self.gaps.append(end - self.newest_odom_stamp)
                if self.newest_odom_stamp >= end:
                    self.waits.append(0.0)  # 到达时已被覆盖，无需等待
                else:
                    self.pending.append((end, now))
            else:
                self.pending.append((end, now))

    def report(self, final=False):
        with self.lock:
            elapsed = time.monotonic() - self.start_mono
            lines = [
                "%s====== odom 滞后统计 (运行 %.0fs) ======" % ("\n最终" if final else "\n", elapsed),
                "  消息数: lidar=%d (%.1fHz)  odom=%d (%.1fHz)  等待中=%d  从未覆盖(>%.0fs)=%d" % (
                    self.lidar_count, self.lidar_count / elapsed if elapsed > 0 else 0.0,
                    self.odom_count, self.odom_count / elapsed if elapsed > 0 else 0.0,
                    len(self.pending), NEVER_COVERED_SEC, self.never_covered),
                fmt_stats("gap", self.gaps),
                fmt_stats("wait", self.waits),
                "  (gap>0 的部分是旧版会丢的帧尾时长；wait 是延迟处理设计的实际额外延迟)",
            ]
        print("\n".join(lines))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odom", default="/latest_imu_odom", help="nav_msgs/Odometry topic")
    parser.add_argument("--lidar", default="/livox/lidar", help="livox CustomMsg topic")
    parser.add_argument("--report", type=float, default=10.0, help="周期性打印间隔 [s]")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("measure_odom_lag", anonymous=True)
    meter = LagMeter()
    rospy.Subscriber(args.odom, Odometry, meter.odom_cb, queue_size=2000)
    rospy.Subscriber(args.lidar, rospy.AnyMsg, meter.lidar_cb, queue_size=10)
    rospy.loginfo("listening: odom=%s lidar=%s", args.odom, args.lidar)

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
