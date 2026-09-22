#!/usr/bin/env python3
"""喂合成输入，逐项核对 hand_lio 的输出对不对（点云 + 机体位姿）。

为什么需要
----------
现成那四个工具都是**只读**的：看拖影、看退化、看重力对齐。这个是一个**回环**检查：
自己发 /latest_imu_odom 与 /livox/lidar（外加 /navibot/virtual_obstacles），
拿解析真值去核对 /hand_lio/clouds_lidar 与 /hand_lio/odom_vehicle。用来验证任何
改动（去畸变、外参、虚拟障碍注入、跨进程输入校验、定位质量门控……）没有把输出搞坏。

期望值从 `config/hand_lio.yaml` 现算，不写死常数：imu_R_lidar / imu_t_lidar /
lidar_R_body / lidar_t_body / blind / n_scans / point_filter_num / world_frame_id /
虚拟障碍那几个参数。**所以它假设节点是 roslaunch 起的、yaml 生效**——若怀疑两种
启动方式参数不一致，先跑 tools/check_start_paths.py。

逐项检查
--------
    1  点云布局：point_step=16、字段 x/y/z/intensity @0/4/8/12、字节数一致且无填充垃圾
    2  点数 = 帧内点数-1（第 0 点按约定跳过）
    3  去畸变几何：逐点与"每点各自时刻插值位姿"的解析真值比（容差 1mm）
    4  intensity = 原始 reflectivity
    5  时间戳 = 帧尾时刻（不是帧首）
    6  /hand_lio/odom_vehicle = world_T_imu * imu_T_lidar * lidar_T_body
    7  虚拟障碍：注入份数 = 可见性×1/r² 模型；真实点一个不丢；frame_id 不符整批忽略；
       空点云清缓存；没法向时不做背面剔除
    8  跨进程输入加固：字段 offset 越界拒收、width 谎报按 data 截断、lidar point_num 谎报夹住
    9  定位质量门控：cov ≥ 阈值丢帧，但 odom_vehicle 照发并透传 cov；恢复后继续发
   10  odom 断流：不发布、也不报假成功；恢复后继续发
   11  重复 odom 时间戳：不丢帧（判据是严格回退），不应出现 went backwards 告警
   12  (信息项) NaN 位姿：记录当前行为，不计入结论

用法
----
    tools/hand_lio.sh start                       # 或 roslaunch hand_lio hand_lio.launch
    python3 tools/check_node_output.py
    python3 tools/check_node_output.py --log <roslaunch 日志>   # 额外核对日志里的加固告警

**会往 /latest_imu_odom、/livox/lidar、/navibot/virtual_obstacles 发合成数据**，
只能在真传感器和导航都停掉的环境里跑（开发机 schroot，或临时 roscore）。默认启动
前有 5 秒倒计时可以 Ctrl-C，`--yes` 跳过。

退出码 0 = 全部通过（信息项不影响）。
"""

import argparse
import math
import os
import signal
import sys
import threading
import time

import numpy as np
import rospy
import yaml
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from hand_lio.msg import CustomMsg, CustomPoint

FRAME_SEC = 0.1          # 合成帧的扫描时长 [s]
N_POINTS = 2000          # 每帧点数；第 0 个点按约定会被节点跳过
# 目标点（雷达系）：都在 1m 开外，不会撞上 blind
TARGETS = np.array([[1.0, 0, 0], [0, 1.5, 0.2], [-2.0, 0, -0.1], [0, -3.0, 0.3], [2.5, 2.5, 0],
                    [-1.0, -1.0, 0.5], [3.0, 0.5, -0.5], [-0.5, 2.0, 0], [1.2, -2.2, 0.1], [-3.5, 1.0, 0]])

checks, failures = [], []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    if not ok:
        failures.append(name)
    print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    sys.stdout.flush()


def note(text):
    print("[----] %s" % text)
    sys.stdout.flush()


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


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_config(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    R = lambda v: np.array(v, dtype=np.float64).reshape(3, 3)
    return dict(
        blind=float(c["blind"]),
        imu_R_lidar=R(c["imu_R_lidar"]), imu_t_lidar=np.array(c["imu_t_lidar"], float),
        lidar_R_body=R(c["lidar_R_body"]), lidar_t_body=np.array(c["lidar_t_body"], float),
        n_scans=int(c["n_scans"]), point_filter_num=int(c["point_filter_num"]),
        world_frame_id=str(c["world_frame_id"]), vehicle_frame_id=str(c["vehicle_frame_id"]),
        cov_thresh=float(c["pose_cov_reject_thresh"]), buffer_horizon=float(c["buffer_horizon_sec"]),
        vo_enable=bool(c["enable_virtual_obstacles"]),
        vo_range=float(c["virtual_obstacle_max_range"]),
        vo_gain=float(c["virtual_obstacle_density_gain"]),
        vo_max_copies=int(c["virtual_obstacle_max_copies"]),
        vo_max_points=int(c["virtual_obstacle_max_points"]),
    )


class Runner(object):
    """发合成输入、收输出。模型的位姿是解析式，所以期望值可以精确算出来。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.clouds, self.odoms, self.frame_ends = [], [], []
        self.tfs = []
        self.cov0, self.odom_enabled, self.nan_mode = 0.05, True, False
        self.last_stamp = None
        self.stop = threading.Event()
        self.T0 = rospy.Time.now().to_sec()
        self.freeze_t = None
        self.odom_pub = rospy.Publisher("/latest_imu_odom", Odometry, queue_size=200)
        self.lidar_pub = rospy.Publisher("/livox/lidar", CustomMsg, queue_size=20)
        self.vo_pub = rospy.Publisher("/navibot/virtual_obstacles", PointCloud2, queue_size=1, latch=True)
        rospy.Subscriber("/hand_lio/clouds_lidar", PointCloud2, self.cloud_cb, queue_size=30)
        rospy.Subscriber("/hand_lio/odom_vehicle", Odometry, self.vehicle_cb, queue_size=400)

    # ---- 解析真值：匀速平动 + 匀角速度偏航，lerp/slerp 正好能精确复现 ----
    def model(self, t):
        if self.freeze_t is not None and t >= self.freeze_t:
            t = self.freeze_t
        dt = t - self.T0
        return np.array([5.0 + 0.4 * dt, 1.0 - 0.2 * dt, 0.5]), 0.2 + 0.3 * dt

    def expected_world(self, t, p_lidar):
        p, yaw = self.model(t)
        return rz(yaw).dot(self.cfg["imu_R_lidar"].dot(p_lidar) + self.cfg["imu_t_lidar"]) + p

    # ---- 回调 ----
    def cloud_cb(self, msg):
        off = {f.name: f.offset for f in msg.fields}
        n = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        grid = raw.reshape(n, msg.point_step)
        xyz = np.column_stack([grid[:, off["x"]:off["x"] + 4], grid[:, off["y"]:off["y"] + 4],
                               grid[:, off["z"]:off["z"] + 4]]).copy().view("<f4").reshape(n, 3).astype(np.float64)
        o = off["intensity"]
        inten = grid[:, o:o + 4].copy().view("<f4").reshape(n).astype(np.float64)
        with self.lock:
            self.clouds.append(dict(stamp=msg.header.stamp.to_sec(), pts=xyz, inten=inten,
                                    frame=msg.header.frame_id, step=msg.point_step,
                                    fields=[(f.name, f.offset, f.datatype) for f in msg.fields],
                                    raw=raw.copy()))

    def vehicle_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with self.lock:
            self.odoms.append(dict(stamp=msg.header.stamp.to_sec(),
                                   pos=np.array([p.x, p.y, p.z]), quat=(q.w, q.x, q.y, q.z),
                                   cov=msg.pose.covariance[0]))

    # ---- 发布 ----
    def odom_loop(self):
        rate = rospy.Rate(200)
        while not self.stop.is_set():
            t = rospy.Time.now().to_sec()
            p, yaw = self.model(t)
            m = Odometry()
            m.header.stamp = rospy.Time.from_sec(t)
            m.header.frame_id = self.cfg["world_frame_id"]
            m.child_frame_id = "imu"
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = p
            if self.nan_mode:
                m.pose.pose.orientation.x = float("nan")
                m.pose.pose.orientation.y = float("nan")
                m.pose.pose.orientation.z = float("nan")
                m.pose.pose.orientation.w = float("nan")
            else:
                m.pose.pose.orientation.z = math.sin(yaw / 2.0)
                m.pose.pose.orientation.w = math.cos(yaw / 2.0)
            m.pose.covariance[0] = self.cov0
            if self.odom_enabled:
                self.odom_pub.publish(m)
                self.last_stamp = t
            rate.sleep()

    def make_frame(self, n=N_POINTS, sec=FRAME_SEC, last_offset=None, declared_extra=0):
        base = rospy.Time.now().to_sec() - sec
        pts = []
        for k in range(n):
            tgt = TARGETS[k % len(TARGETS)]
            off = last_offset if (last_offset is not None and k == n - 1) else k * sec / max(n - 1, 1)
            pts.append(CustomPoint(x=float(tgt[0]), y=float(tgt[1]), z=float(tgt[2]),
                                   offset_time=int(round(off * 1e9)), reflectivity=100,
                                   tag=0x10, line=k % self.cfg["n_scans"]))
        m = CustomMsg()
        m.header.stamp = rospy.Time.from_sec(base)
        m.header.frame_id = "livox_frame"
        m.timebase = int(base * 1e9)
        m.point_num = n + declared_extra
        m.lidar_id = 0
        m.rsvd = [0, 0, 0]
        m.points = pts
        return m, base + pts[-1].offset_time * 1e-9

    def lidar_loop(self):
        rate = rospy.Rate(10)
        while not self.stop.is_set():
            m, end = self.make_frame()
            self.lidar_pub.publish(m)
            with self.lock:
                self.frame_ends.append(end)
            rate.sleep()

    def publish_vo(self, pts, normals, frame_id=None, width=None, point_step=None, fields=None):
        m = PointCloud2()
        m.header.frame_id = self.cfg["world_frame_id"] if frame_id is None else frame_id
        m.header.stamp = rospy.Time.now()
        m.height = 1
        m.width = len(pts) if width is None else width
        m.fields = fields if fields is not None else [
            PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1), PointField("normal_x", 12, PointField.FLOAT32, 1),
            PointField("normal_y", 16, PointField.FLOAT32, 1), PointField("normal_z", 20, PointField.FLOAT32, 1)]
        m.point_step = point_step if point_step is not None else 24
        m.row_step = m.point_step * m.width
        m.is_bigendian = False
        m.is_dense = True
        m.data = np.hstack([pts, normals]).astype("<f4").tobytes()
        self.vo_pub.publish(m)

    def snap(self):
        with self.lock:
            return list(self.clouds), list(self.odoms), list(self.frame_ends)

    # ---- 模型：可见性 + 1/r² 加密（跟 appendVirtualObstacles 的规则一致）----
    def vo_model(self, sensor_pos, pts, normals, quiet=True):
        vis = []
        for p, n in zip(pts, normals):
            d = sensor_pos - p
            r2 = float(d.dot(d))
            if r2 > self.cfg["vo_range"] ** 2 or r2 < 1e-4:
                continue
            if np.linalg.norm(n) > 1e-3 and float(n.dot(d)) <= 0.0:
                continue
            vis.append(r2)
        planned = 0
        for r2 in vis:
            c = min(max(int(round(self.cfg["vo_gain"] / max(r2, 1e-4))), 1), self.cfg["vo_max_copies"])
            planned += min(c, self.cfg["vo_max_points"] - planned)
        if not quiet:
            print("    模型：可见 %d 点 -> 注入 %d 点" % (len(vis), planned))
        return planned

    def ring(self, cx, cy):
        """机器人前方一个闭合矩形禁行区（外法向朝外）：只有近侧那面墙看得见。"""
        step = 0.025
        pts, nrm = [], []
        x0, x1, y0, y1 = cx + 1.0, cx + 3.0, cy - 5.0, cy + 5.0
        for y in np.arange(y0, y1 + step / 2, step):
            pts += [[x0, y, 0.5], [x1, y, 0.5]]
            nrm += [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        for x in np.arange(x0, x1 + step / 2, step):
            pts += [[x, y0, 0.5], [x, y1, 0.5]]
            nrm += [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
        return np.array(pts), np.array(nrm)


def injected(cloud):
    return len(cloud["pts"]) - int(np.sum(cloud["inten"] != 0.0))


def frames_missing(ends, clouds, t_from, t_to, tol=2e-6):
    stamps = [c["stamp"] for c in clouds]
    return [e for e in ends if t_from <= e <= t_to and not any(abs(e - s) <= tol for s in stamps)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "hand_lio.yaml")
    ap.add_argument("--config", default=default_cfg, help="hand_lio.yaml 路径")
    ap.add_argument("--log", default=None, help="roslaunch 日志路径；给了就额外核对日志里的加固告警")
    ap.add_argument("--yes", action="store_true", help="跳过启动前的 5 秒倒计时")
    args = ap.parse_args(rospy.myargv()[1:])

    cfg = load_config(args.config)
    print(__doc__.split("用法")[0].split("为什么需要")[0].strip())
    print("\n配置：blind=%.3f  n_scans=%d  point_filter_num=%d  world=%s  虚拟障碍=%s"
          % (cfg["blind"], cfg["n_scans"], cfg["point_filter_num"], cfg["world_frame_id"], cfg["vo_enable"]))
    if cfg["blind"] >= 1.0:
        print("blind ≥ 1m：合成点(≥1m)会被滤掉，本检查不适用")
        return 1
    if cfg["point_filter_num"] != 1:
        print("point_filter_num=%d ≠ 1：本检查的点数/顺序判据不适用" % cfg["point_filter_num"])
        return 1
    if not args.yes:
        print("\n**会向 /latest_imu_odom、/livox/lidar、/navibot/virtual_obstacles 发合成数据**，"
              "5 秒后开始，Ctrl-C 取消……")
        for _ in range(5):
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

    rospy.init_node("check_node_output", anonymous=True)
    r = Runner(cfg)
    for fn in (r.odom_loop, r.lidar_loop):
        th = threading.Thread(target=fn)
        th.daemon = True
        th.start()

    # ---- 1~6：布局、几何、点数、时间戳、机体位姿 ----
    rospy.sleep(3.0)
    clouds, odoms, ends = r.snap()
    check("点云在发布", len(clouds) >= 15, "3s 收到 %d 帧" % len(clouds))
    if not clouds:
        print("没有输出，后面的检查跳过；先看节点日志")
        return 1
    c = clouds[-1]
    check("布局 point_step=16 且字段在 0/4/8/12",
          c["step"] == 16 and [f[1] for f in c["fields"]] == [0, 4, 8, 12]
          and [f[0] for f in c["fields"]] == ["x", "y", "z", "intensity"],
          "point_step=%d fields=%s" % (c["step"], c["fields"]))
    expected = np.array([r.expected_world(c["stamp"] - FRAME_SEC + k * FRAME_SEC / (N_POINTS - 1),
                                          TARGETS[k % len(TARGETS)]) for k in range(1, N_POINTS)])
    check("点数 = 帧内点数-1（第 0 点跳过）", len(c["pts"]) == len(expected),
          "%d vs %d" % (len(c["pts"]), len(expected)))
    delta = np.zeros(3)
    if len(c["pts"]) == len(expected):
        err = np.linalg.norm(c["pts"] - expected, axis=1)
        check("去畸变+外参几何与解析真值一致", err.max() < 1e-3,
              "最大 %.7f m  平均 %.7f m" % (err.max(), err.mean()))
        # 真实点整体相对期望的平移量 = 节点实际用的外参与 yaml 的差。
        # 虚拟障碍那几项要用它推算**节点自己的**传感器位置，否则外参一错，
        # 那几项会跟着一起红，报出来的是同一个根因的 5 个症状。
        delta = c["pts"].mean(axis=0) - expected.mean(axis=0)
    packed = np.hstack([c["pts"], c["inten"].reshape(-1, 1)]).astype("<f4").tobytes()
    check("线上字节 = 紧凑打包（无填充垃圾）", c["raw"].tobytes() == packed,
          "data=%d 期望=%d" % (c["raw"].size, len(packed)))
    check("intensity = 原始 reflectivity", np.all(c["inten"] == 100), "取值 %s" % np.unique(c["inten"])[:3])
    check("时间戳 = 帧尾（不是帧首）", min(abs(c["stamp"] - e) for e in ends[-8:]) < 1e-6
          and all(abs(c["stamp"] - (e - FRAME_SEC)) > 1e-6 for e in ends[-8:]),
          "stamp=%.6f" % c["stamp"])
    if odoms:
        e_pos = e_rot = 0.0
        for o in odoms[10:]:
            p_ref, yaw_ref = r.model(o["stamp"])
            pos_ref = p_ref + rz(yaw_ref).dot(cfg["imu_t_lidar"] + cfg["imu_R_lidar"].dot(cfg["lidar_t_body"]))
            R_ref = rz(yaw_ref).dot(cfg["imu_R_lidar"]).dot(cfg["lidar_R_body"])
            e_pos = max(e_pos, float(np.linalg.norm(o["pos"] - pos_ref)))
            e_rot = max(e_rot, float(np.abs(quat_to_R(*o["quat"]) - R_ref).max()))
        check("odom_vehicle = world_T_imu*imu_T_lidar*lidar_T_body", max(e_pos, e_rot) < 1e-3,
              "位置 %.6f m  姿态 %.6f" % (e_pos, e_rot))

    # ---- 7：虚拟障碍 ----
    r.freeze_t = rospy.Time.now().to_sec()      # 冻住位姿，可见性判据才可精确预测
    rospy.sleep(0.7)
    sensor = r.expected_world(r.freeze_t + 1.0, np.zeros(3)) + delta
    pts_v, nrm_v = r.ring(sensor[0], sensor[1])
    if cfg["vo_enable"]:
        r.publish_vo(pts_v, nrm_v)
        rospy.sleep(2.0)
        c2, _, _ = r.snap()
        new2 = [x for x in c2 if x["stamp"] > c["stamp"]]
        exp_inj = r.vo_model(sensor, pts_v, nrm_v, quiet=False)
        if new2:
            got = injected(new2[-1])
            check("虚拟障碍注入份数 = 可见性×1/r² 模型", abs(got - exp_inj) <= max(0.05 * exp_inj, 3),
                  "实测 %d 模型 %d" % (got, exp_inj))
            check("注入时真实点一个不丢", int(np.sum(new2[-1]["inten"] != 0.0)) == N_POINTS - 1, "")
            check("混合后 frame_id 仍是 %s" % cfg["world_frame_id"], new2[-1]["frame"] == cfg["world_frame_id"], "")
        else:
            check("虚拟障碍注入", False, "注入后没有新点云")
        base_inj = injected(r.snap()[0][-1])
        r.publish_vo(pts_v, nrm_v, frame_id="not_" + cfg["world_frame_id"])
        rospy.sleep(1.5)
        check("frame_id 不符时整批忽略（保留旧缓存）", injected(r.snap()[0][-1]) == base_inj,
              "之前 %d 之后 %d" % (base_inj, injected(r.snap()[0][-1])))
        r.publish_vo(np.zeros((0, 3)), np.zeros((0, 3)))
        rospy.sleep(1.5)
        check("空点云清掉缓存（禁行区删干净后不留旧墙）", injected(r.snap()[0][-1]) == 0, "")
        vis = [i for i in range(len(pts_v)) if nrm_v[i][0] == -1.0 and abs(pts_v[i][1] - sensor[1]) < 1.0][:5]
        r.publish_vo(pts_v[vis], nrm_v[vis])
        rospy.sleep(1.5)
        check("没有法向时不做背面剔除（全注入）",
              injected(r.snap()[0][-1]) == r.vo_model(sensor, pts_v[vis], nrm_v[vis]), "")
        base_inj = injected(r.snap()[0][-1])

        # ---- 8：跨进程输入加固 ----
        r.publish_vo(pts_v, nrm_v, point_step=8,
                     fields=[PointField("x", 0, PointField.FLOAT32, 1),
                             PointField("y", 4, PointField.FLOAT32, 1),
                             PointField("z", 8, PointField.FLOAT32, 1)], width=len(pts_v))
        rospy.sleep(1.5)
        check("字段 offset 越界（offset+4 > point_step）拒收", injected(r.snap()[0][-1]) == base_inj, "")
        small_pts, small_nrm = pts_v[vis].copy(), nrm_v[vis].copy()
        r.publish_vo(small_pts, small_nrm, width=500)
        rospy.sleep(1.5)
        check("width 谎报时按 data 容量截断",
              injected(r.snap()[0][-1]) == r.vo_model(sensor, small_pts, small_nrm),
              "实测 %d 模型 %d" % (injected(r.snap()[0][-1]), r.vo_model(sensor, small_pts, small_nrm)))
        n_c, n_e = len(r.snap()[0]), len(r.snap()[2])
        m, end = r.make_frame(n=500, sec=0.02, last_offset=0.02, declared_extra=1000)
        r.lidar_pub.publish(m)
        with r.lock:
            r.frame_ends.append(end)
        rospy.sleep(2.5)
        c8 = [x for x in r.snap()[0][n_c:] if abs(x["stamp"] - end) < 1e-6]
        check("lidar point_num 谎报时夹到实际数组（不越界读）",
              bool(c8) and int(np.sum(c8[0]["inten"] != 0.0)) == 499,
              "真实点 %d" % (int(np.sum(c8[0]["inten"] != 0.0)) if c8 else -1))
    else:
        note("config 里 enable_virtual_obstacles=false，跳过虚拟障碍相关的 4 组检查")

    # ---- 9：定位质量门控 ----
    t_gate = rospy.Time.now().to_sec()
    r.cov0 = cfg["cov_thresh"]
    rospy.sleep(1.2)
    c9, o9, _ = r.snap()
    check("cov >= 阈值时丢帧（不喂假点云给规划器）",
          not [x for x in c9 if t_gate + 0.3 < x["stamp"] < t_gate + 1.0], "")
    vg = [o for o in o9 if t_gate + 0.3 < o["stamp"] < t_gate + 1.0]
    check("定位失败时 odom_vehicle 照发并透传 cov", len(vg) > 100 and all(abs(o["cov"] - cfg["cov_thresh"]) < 1e-6 for o in vg),
          "%d 条" % len(vg))
    r.cov0 = 0.05
    n0 = len(c9)
    rospy.sleep(1.5)
    check("定位恢复后点云继续发", len(r.snap()[0]) - n0 >= 8, "")

    # ---- 10：odom 断流 ----
    r.odom_enabled = False
    rospy.sleep(1.5)
    n_s = len(r.snap()[0])
    rospy.sleep(1.0)
    check("odom 断流时不发陈旧点云", len([x for x in r.snap()[0][n_s:] if x["stamp"] < rospy.Time.now().to_sec()]) <= 2, "")
    r.odom_enabled = True
    n_r = len(r.snap()[0])
    rospy.sleep(1.5)
    check("odom 恢复后点云继续发", len(r.snap()[0]) - n_r >= 8, "")

    # ---- 11：重复 odom 时间戳 ----
    mark = rospy.Time.now().to_sec()
    n_c, n_e = len(r.snap()[0]), len(r.snap()[2])
    p_dup, yaw_dup = r.model(r.last_stamp)
    m = Odometry()
    m.header.stamp = rospy.Time.from_sec(r.last_stamp)
    m.header.frame_id = cfg["world_frame_id"]
    m.child_frame_id = "imu"
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = p_dup
    m.pose.pose.orientation.z = math.sin(yaw_dup / 2.0)
    m.pose.pose.orientation.w = math.cos(yaw_dup / 2.0)
    m.pose.covariance[0] = 0.05
    r.odom_pub.publish(m)
    rospy.sleep(2.8)
    c11, _, ends11 = r.snap()
    # 窗口从重复戳之前一点开始（那帧正是旧代码会丢的），到快照前 0.2s 结束
    # （最后一帧的点云可能还没到，算进去会假报丢帧）
    window = (mark - 0.15, min(mark + 2.5, rospy.Time.now().to_sec() - 0.2))
    missing = frames_missing(ends11[n_e:], c11[n_c:], window[0], window[1])
    check("重复 odom 时间戳不丢帧（判据是严格回退）", not missing,
          "%.1fs 窗口内 %d 帧没发出来" % (window[1] - window[0], len(missing)))

    # ---- 12：信息项，NaN 位姿 ----
    n_c12 = len(r.snap()[0])
    with r.lock:
        n_tf = len(r.tfs) if hasattr(r, "tfs") else 0
    t_nan = rospy.Time.now().to_sec()
    r.nan_mode = True
    rospy.sleep(0.6)
    r.nan_mode = False
    rospy.sleep(1.2)
    c12, o12, _ = r.snap()
    bad_clouds = [x for x in c12[n_c12:] if t_nan <= x["stamp"] <= t_nan + 1.8 and not np.all(np.isfinite(x["pts"]))]
    bad_odom = [o for o in o12 if t_nan <= o["stamp"] <= t_nan + 0.8 and not np.all(np.isfinite(o["pos"]))]
    if bad_clouds or bad_odom:
        note("已知缺口（不计入结论）：odom 位姿没有有限性校验 —— 0.6s 的 NaN 位姿"
             "（cov 仍是 0.05，门控拦不住）导致 %d 帧点云含非有限点、%d 条 odom_vehicle 位置为 NaN；"
             "TF 同理。见 config 里 pose_cov_reject_thresh 的说明。" % (len(bad_clouds), len(bad_odom)))
    else:
        note("NaN 位姿没有渗进点云/odom_vehicle（已经被拦住了）")

    # ---- 可选：核对日志 ----
    if args.log and os.path.exists(args.log):
        text = open(args.log, "rb").read().decode("utf8", "replace")
        check("日志里没有'时间戳回退'告警（重复戳不该判回退）", "timestamp went backwards" not in text, "")
        check("日志里有越界拒收/截断/夹住的告警",
              all(k in text for k in ("exceeds point_step", "truncating", "clamping")), "")
    elif args.log:
        note("--log 指定的文件不存在：%s" % args.log)

    r.stop.set()
    print("\n%d/%d 项通过" % (len(checks) - len(failures), len(checks)))
    if failures:
        print("未通过：%s" % "、".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(130)
