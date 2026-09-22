#!/usr/bin/env python3
"""对比 hand_lio 两种启动方式实际跑在什么参数上：roslaunch(带 yaml) vs rosrun(C++ 默认值)。

背景
----
节点的参数有两个来源：

  1. `config/hand_lio.yaml` —— 只有 roslaunch 会加载（launch 里 rosparam 到 ~ 命名空间）
  2. `include/hand_lio/HandLioNode.h` 的成员默认值 —— rosrun / 直接跑可执行文件时用

两处不一致时，同一个节点换一种启动方式就换了参数，而且不会有任何提示。已经踩过
两次：`blind`（0.8 / 0.35）、`imu_t_lidar`（工厂值 / 零）。这个脚本把两种方式各起
一遍，从**实际输出**反推节点的有效参数，两两对比，最后跟 yaml 对一次。

怎么反推（喂一帧合成数据，不需要真传感器）
----------------------------------------
    有效 imu_R_lidar / imu_t_lidar  <- 单位位姿下四个已知点的落点解出来
    有效盲区                        <- 落在 -x 轴上那一串点里最近的一个
    有效机体位置                    <- /hand_lio/odom_vehicle.position
                                       = imu_t_lidar + imu_R_lidar*lidar_t_body

构造：一帧里先放 4 个锚点 (1,0,0)(0,1,0)(0,0,1)(1,1,1)（半径都 ≥1m，任何 blind
都留得住），再放一串 -x 轴上半径 0.2~1.0m 的点。锚点解出仿射映射
`p_world = R*p_lidar + t`（4 个非共面点足够，R 由前三列给出、t 由四式联立），
再把 -x 那串代回去换算成**雷达系半径**——那就是节点实际用的盲区。

用法
----
    python3 tools/check_start_paths.py                 # 自己起 roscore，两种方式各一趟
    python3 tools/check_start_paths.py --config <hand_lio.yaml>
    python3 tools/check_start_paths.py --keep          # 结束后不杀节点，便于手查

**会向 /latest_imu_odom 和 /livox/lidar 发合成数据**，只能在没有真传感器、没有导航
在跑的环境里用（开发机 schroot，或临时起的 roscore）。真机上先停 hand_lio.service。

退出码 0 = 两种启动方式测得一致、且都与 yaml 相符；1 = 有差异（脚本会把差异打出来）。
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rospy
import yaml
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from hand_lio.msg import CustomMsg, CustomPoint

# 锚点：半径都 ≥1m，不会被 blind 滤掉；四个不共面，足够解出 R 和 t。
ANCHORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
PROBE_RADII = [0.2, 0.34, 0.36, 0.5, 0.79, 0.81, 1.0]

TOL = 1e-3        # [m] 判定"一致"的容差：点云是 float32，插值本身是精确的
FRAME_STAMP_AGE = 0.02   # [s] 合成帧的 header.stamp 往前放一点，保证 odom 覆盖得到


def quat_to_R(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def blind_interval(measured_min):
    """测得的最小存活半径 -> (盲区上界, 盲区下界] 的区间写法，盲区只能测到这个精度。"""
    lower = [r for r in PROBE_RADII if r < measured_min - 1e-4]
    return (max(lower) if lower else 0.0, measured_min)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    R = lambda v: np.array(v, dtype=np.float64).reshape(3, 3)
    return {
        "blind": float(cfg["blind"]),
        "imu_R_lidar": R(cfg["imu_R_lidar"]),
        "imu_t_lidar": np.array(cfg["imu_t_lidar"], dtype=np.float64),
        "lidar_R_body": R(cfg["lidar_R_body"]),
        "lidar_t_body": np.array(cfg["lidar_t_body"], dtype=np.float64),
    }


class Probe(object):
    """订阅输出、发合成输入，反推节点实际用的外参与盲区。"""

    def __init__(self):
        self.lock_n = 0
        self.cloud = None
        self.body_pos = None
        self.body_R = None
        self.odom_pub = rospy.Publisher("/latest_imu_odom", Odometry, queue_size=50)
        self.lidar_pub = rospy.Publisher("/livox/lidar", CustomMsg, queue_size=5)
        rospy.Subscriber("/hand_lio/clouds_lidar", PointCloud2, self.cloud_cb, queue_size=5)
        rospy.Subscriber("/hand_lio/odom_vehicle", Odometry, self.body_cb, queue_size=50)
        self.stop = False

    def cloud_cb(self, msg):
        off = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.width * msg.height, msg.point_step)
        cols = []
        for axis in ("x", "y", "z"):
            o = off[axis]
            cols.append(raw[:, o:o + 4].copy().view("<f4"))
        self.cloud = np.column_stack(cols).astype(np.float64)
        self.lock_n += 1

    def body_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.body_pos = np.array([p.x, p.y, p.z])
        self.body_R = quat_to_R(q.w, q.x, q.y, q.z)

    def publish_odom(self):
        m = Odometry()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "world"
        m.child_frame_id = "imu"
        m.pose.pose.orientation.w = 1.0
        m.pose.covariance[0] = 0.02
        self.odom_pub.publish(m)

    def publish_frame(self, n_scans=4):
        pts = [CustomPoint(x=9.0, y=9.0, z=9.0, offset_time=0, reflectivity=1, tag=0x10, line=0)]
        seq = ANCHORS + [(-r, 0.0, 0.0) for r in PROBE_RADII]
        for i, v in enumerate(seq):
            pts.append(CustomPoint(x=v[0], y=v[1], z=v[2], offset_time=int(i * 2e6),
                                   reflectivity=1, tag=0x10, line=i % n_scans))
        m = CustomMsg()
        m.header.stamp = rospy.Time.from_sec(rospy.Time.now().to_sec() - FRAME_STAMP_AGE)
        m.header.frame_id = "livox_frame"
        m.point_num = len(pts)
        m.lidar_id = 0
        m.rsvd = [0, 0, 0]
        m.points = pts
        self.lidar_pub.publish(m)

    def measure(self, timeout):
        """发到收到一帧新点云为止，返回 (R_eff, t_eff, blind_eff, body_eff, n_points)。"""
        start = rospy.Time.now().to_sec()
        deadline = start + timeout
        n0 = self.lock_n
        next_frame = start
        rate = rospy.Rate(100)
        while rospy.Time.now().to_sec() < deadline and not rospy.is_shutdown():
            self.publish_odom()
            if rospy.Time.now().to_sec() >= next_frame:      # 5Hz 合成帧，够节点处理过来
                self.publish_frame()
                next_frame = rospy.Time.now().to_sec() + 0.2
            if self.lock_n > n0 and self.cloud is not None and len(self.cloud) >= len(ANCHORS) + 1:
                break
            rate.sleep()
        if self.lock_n <= n0 or self.cloud is None:
            return None
        d = self.cloud[:len(ANCHORS), :]
        t = (d[0] + d[1] + d[2] - d[3]) / 2.0     # d_i = R*e_i + t, d_4 = R*(1,1,1) + t
        R = np.column_stack([d[0] - t, d[1] - t, d[2] - t])
        fam = self.cloud[len(ANCHORS):, :]
        lidar_frame = (fam - t).dot(R)            # = R^T (p - t)
        blind = float(np.linalg.norm(lidar_frame, axis=1).min()) if len(fam) else float("nan")
        return R, t, blind, self.body_pos, self.body_R, len(self.cloud)


def start_node(mode, log_path):
    log = open(log_path, "wb")
    if mode == "roslaunch":
        cmd = ["roslaunch", "hand_lio", "hand_lio.launch"]
    else:
        cmd = ["rosrun", "hand_lio", "hand_lio_node"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid, env=os.environ.copy())


def stop_node(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        for _ in range(40):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass


def ensure_master():
    """没有 master 就自己起一个（跟 tools/hand_lio.sh 同一个套路），返回要负责关掉的进程。"""
    for _ in range(3):
        try:
            if subprocess.call(["timeout", "3", "rostopic", "list"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                return None
        except OSError:
            break
        time.sleep(1)
    print("ROS master 不可达，后台起一个临时 roscore")
    proc = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    time.sleep(4)
    return proc


def fmt_vec(v):
    return "(" + ", ".join("%+.4f" % x for x in v) + ")"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "hand_lio.yaml")
    ap.add_argument("--config", default=default_cfg, help="hand_lio.yaml 路径")
    ap.add_argument("--timeout", type=float, default=20.0, help="每种启动方式最多等多久 [s]")
    ap.add_argument("--keep", action="store_true", help="结束后不杀节点")
    args = ap.parse_args(rospy.myargv()[1:])

    cfg = load_config(args.config)
    print("yaml 里的值：blind=%.3f  imu_t_lidar=%s" % (cfg["blind"], fmt_vec(cfg["imu_t_lidar"])))
    if cfg["blind"] >= 1.0:
        print("警告：blind ≥ 1m，锚点(半径1m)可能被滤掉，这个检查不适用")
        return 1

    master = ensure_master()
    rospy.init_node("check_start_paths", anonymous=True)
    probe = Probe()

    logdir = "/tmp/hand_lio_check_start_paths"
    os.makedirs(logdir, exist_ok=True)
    results = {}
    try:
        for mode in ("roslaunch", "rosrun"):
            log_path = os.path.join(logdir, mode + ".log")
            print("\n--- 启动方式：%s（日志 %s）" % (mode, log_path))
            proc = start_node(mode, log_path)
            try:
                m = probe.measure(args.timeout)
            finally:
                if not args.keep:
                    stop_node(proc)
            if m is None:
                print("  没收到 /hand_lio/clouds_lidar，看看日志")
                results[mode] = None
                continue
            R, t, blind, body, body_R, npts = m
            results[mode] = dict(R=R, t=t, blind=blind, body=body, body_R=body_R, n=npts)
            lo, hi = blind_interval(blind)
            print("  收到 %d 点：imu_R_lidar 与 yaml 最大偏差 %.5f，imu_t_lidar=%s，"
                  "盲区 ∈ (%.3f, %.3f] m，机体位置=%s，机体姿态与 yaml 最大偏差 %.5f"
                  % (npts, np.abs(R - cfg["imu_R_lidar"]).max(), fmt_vec(t), lo, hi,
                     fmt_vec(body), np.abs(body_R - cfg["imu_R_lidar"].dot(cfg["lidar_R_body"])).max()
                     if body_R is not None else float("nan")))
    finally:
        if not args.keep:
            if master is not None:
                try:
                    os.killpg(os.getpgid(master.pid), signal.SIGINT)
                except OSError:
                    pass

    print("\n" + "=" * 78)
    bad = []
    ok = [m for m in results if results[m] is not None]
    if len(ok) < 2:
        print("两种启动方式没有都测到，无法两两对比")
        return 1
    body_ref = cfg["imu_t_lidar"] + cfg["imu_R_lidar"].dot(cfg["lidar_t_body"])
    body_R_ref = cfg["imu_R_lidar"].dot(cfg["lidar_R_body"])
    # 盲区只能测成区间：最小存活半径就是探测点里第一个大于 blind 的那个。
    exp_min = min([r for r in PROBE_RADII if r > cfg["blind"]], default=None)
    if exp_min is None:
        print("探测半径最大 %.2fm 小于 blind=%.2f，量不到盲区（把 PROBE_RADII 调大）"
              % (max(PROBE_RADII), cfg["blind"]))
        return 1
    print("yaml 的 blind=%.3f 对应'最小存活半径=%.3f'（盲区只能测到探测点的精度）"
          % (cfg["blind"], exp_min))
    for mode in ("roslaunch", "rosrun"):
        r = results.get(mode)
        if r is None:
            bad.append("%s 没测到" % mode)
            continue
        dR = np.abs(r["R"] - cfg["imu_R_lidar"]).max()
        dt = np.abs(r["t"] - cfg["imu_t_lidar"]).max()
        db = abs(r["blind"] - exp_min)
        dbody = np.abs(r["body"] - body_ref).max() if r["body"] is not None else float("nan")
        dbodyR = (np.abs(r["body_R"] - body_R_ref).max()
                  if r["body_R"] is not None else float("nan"))
        worst = max(dR, dt, db, dbody, dbodyR)
        tag = "OK " if worst <= TOL else "差异"
        print("%-5s %s  imu_R_lidar Δ=%.5f  imu_t_lidar Δ=%.5f  最小存活半径 %.3f(Δ=%.3f)  "
              "机体位置 Δ=%.5f  机体姿态 Δ=%.5f"
              % (tag, mode, dR, dt, r["blind"], db, dbody, dbodyR))
        if worst > TOL:
            bad.append("%s 的实际参数与 yaml 不符（Δ 见上）" % mode)
    dcross = max(np.abs(results["roslaunch"]["R"] - results["rosrun"]["R"]).max(),
                 np.abs(results["roslaunch"]["t"] - results["rosrun"]["t"]).max(),
                 abs(results["roslaunch"]["blind"] - results["rosrun"]["blind"]),
                 np.abs(results["roslaunch"]["body"] - results["rosrun"]["body"]).max(),
                 np.abs(results["roslaunch"]["body_R"] - results["rosrun"]["body_R"]).max())
    print("两种启动方式之间最大差异：%.5f" % dcross)
    if dcross > TOL:
        bad.append("roslaunch 与 rosrun 跑在不同参数上")
    print("-" * 78)
    if bad:
        print("结论：**不一致** —— %s" % "；".join(bad))
        print("        yaml 与 C++ 成员默认值必须同步（见 HandLioNode.h 参数段注释）。")
        return 1
    print("结论：两种启动方式参数一致，且都与 yaml 相符。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
