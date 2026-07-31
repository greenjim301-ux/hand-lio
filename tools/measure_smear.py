#!/usr/bin/env python3
"""逐帧测量 /hand_lio/clouds_lidar 中一面墙的"厚度"，用于判断运动拖影是否显著。

原理：拖影是帧内畸变——运动时若去畸变位姿不准（如 /latest_imu_odom 的 10Hz 台阶），
同一帧里先扫到和后扫到的墙点会错位，表现为单帧内墙面沿法向变"厚"。
本脚本对每帧落在指定包围盒内的点做 PCA 平面拟合，报告沿法向的散布：

    thickness_std：点到拟合平面距离的标准差
    spread95    ：点到平面距离的 2.5%~97.5% 分位区间宽度

用法：先在 rviz 里读出目标墙面的大致 world 坐标范围，然后：

    python3 tools/measure_smear.py --xmin 2.0 --xmax 3.0 --ymin -2.0 --ymax 2.0 --zmin 0.3 --zmax 1.5

静止时先跑一段记下基线（约 1~3cm），再平行于墙行走/原地转身各跑一段。
运动时 thickness_std 明显超过基线（2 倍以上）即拖影显著。Ctrl+C 打印汇总。
"""

import argparse
import signal
import sys

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


class SmearMeter(object):
    def __init__(self, box, min_points):
        self.box = box  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.min_points = min_points
        self.stds = []

    def cloud_cb(self, msg):
        pts = np.array(
            [p for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float64)
        if pts.size == 0:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.box
        mask = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
                (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))
        sel = pts[mask]
        if len(sel) < self.min_points:
            rospy.loginfo_throttle(2.0, "包围盒内点数不足 (%d < %d)，检查范围是否套住了墙面", len(sel), self.min_points)
            return

        # PCA 平面拟合：最小特征值方向即法向，点在该方向上的散布就是"厚度"
        centered = sel - sel.mean(axis=0)
        cov = centered.T @ centered / len(sel)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        dist = centered @ normal
        std = float(dist.std())
        lo, hi = np.percentile(dist, [2.5, 97.5])
        self.stds.append(std)
        rospy.loginfo("n=%5d  thickness_std=%6.1fmm  spread95=%6.1fmm", len(sel), std * 1000.0, (hi - lo) * 1000.0)

    def summary(self):
        if not self.stds:
            print("没有有效帧")
            return
        v = np.array(self.stds) * 1000.0
        print("\n====== 汇总 (%d 帧) ======" % len(v))
        print("thickness_std: min=%.1fmm  median=%.1fmm  mean=%.1fmm  p95=%.1fmm  max=%.1fmm" % (
            v.min(), np.median(v), v.mean(), np.percentile(v, 95), v.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloud", default="/hand_lio/clouds_lidar")
    parser.add_argument("--xmin", type=float, required=True)
    parser.add_argument("--xmax", type=float, required=True)
    parser.add_argument("--ymin", type=float, required=True)
    parser.add_argument("--ymax", type=float, required=True)
    parser.add_argument("--zmin", type=float, required=True)
    parser.add_argument("--zmax", type=float, required=True)
    parser.add_argument("--min-points", type=int, default=100, help="单帧参与拟合的最少点数")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("measure_smear", anonymous=True)
    meter = SmearMeter((args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax), args.min_points)
    rospy.Subscriber(args.cloud, PointCloud2, meter.cloud_cb, queue_size=5)
    rospy.loginfo("listening: %s  box=x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]",
                  args.cloud, args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)

    def on_sigint(_sig, _frame):
        meter.summary()
        rospy.signal_shutdown("done")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    rospy.spin()


if __name__ == "__main__":
    main()
