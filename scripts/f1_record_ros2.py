"""
f1_record_ros2.py —— F1 PICO 遥操录制（ROS2 版，含头部摄像头同步）

链路: PICO 手柄 → pico_teleop_node → /realtime_joint_cmd → 本脚本订阅 → HDF5 → LeRobot
      F1 头部相机 (Orbbec Gemini 335) → 本脚本抓帧 → 与关节帧对齐存 HDF5

用法（机器人上）:
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash
  python3 f1_record_ros2.py --cam /dev/video6    # 关节 + 机载画面同步录制（推荐）
  python3 f1_record_ros2.py --cam rtsp://192.168.1.16:8554/live   # 拉 RTSP 流录制
  python3 f1_record_ros2.py                       # 只录关节（摄像头不可用时）

PC 端转 LeRobot（PC 不需要 ROS）:
  python f1_record_ros2.py --convert recordings/session_xxx.h5
      ↑ 自动把 HDF5 里的机载画面解码进数据集（视觉 + 关节天然同步）
  python f1_record_ros2.py --convert xxx.h5 --video pico.mp4
      ↑ HDF5 没画面时，用 PICO 录屏均匀采样对齐（备用方案）

关节顺序（14 DOF，单位 rad）:
  [J1_right..J7_right, J1_left..J7_left] —— 右臂在前。
  躯干 3 DOF（腰/升降/头）不在手臂 topic 里，反馈原样存 HDF5（observation.motor_*）。

依赖:
  机器人端: rclpy + numpy + h5py（--cam 模式必需: pip install h5py）+ cv2（--cam 模式）
  PC 端:   numpy + h5py + lerobot（HDF5 带画面时还需 cv2）
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 14-DOF 关节名（右臂在前，左臂在后，与旧数据集命名一致）
JOINT_NAMES_14 = [
    "J1_right_joint", "J2_right_joint", "J3_right_joint", "J4_right_joint",
    "J5_right_joint", "J6_right_joint", "J7_right_joint",
    "J1_left_joint", "J2_left_joint", "J3_left_joint", "J4_left_joint",
    "J5_left_joint", "J6_left_joint", "J7_left_joint",
]

N_JOINTS = 14
CAM_KEY = "observation.images.cam"


def pad7(arr, fill=np.nan):
    """补到 7 维（反馈数据长度可能不固定，统一 pad 成 7）"""
    a = np.asarray(arr, dtype=np.float32)
    if a.size >= 7:
        return a[:7].astype(np.float32)
    p = np.full(7, fill, dtype=np.float32)
    p[: a.size] = a
    return p


# ---------------------------------------------------------------- 摄像头抓帧


class CameraGrabber:
    """后台线程持续抓摄像头最新帧，主线程取最新一张。

    source: V4L2 设备（如 /dev/video6）或 RTSP URL（cv2 都支持）。
    帧以 JPEG 字节存储（省空间），解码放到 PC 端转换时做。
    """

    def __init__(self, source, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._jpeg = None          # 最新帧 JPEG 字节
        self._frame_shape = None   # (h, w)
        self.frames_captured = 0
        self.ok = False
        self.err = ""

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(50):        # 最多等 5 秒出第一帧
            if self.frames_captured > 0 or self.err:
                break
            time.sleep(0.1)
        return self.ok

    def _run(self):
        import cv2
        cap = None
        try:
            if self.source.startswith("/dev/"):
                cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.err = f"打不开 {self.source}"
                return

            ret, frame = cap.read()
            if not ret:            # 默认格式协商失败 → 退到 MJPG 重试
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, frame = cap.read()
            if not ret:
                self.err = f"读不到帧: {self.source}"
                return

            while ret:
                if frame.ndim == 3:                # 解码后的 BGR 帧 → 压成 JPEG
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        break
                    h, w = frame.shape[:2]
                    jpg = jpg.tobytes()
                else:                              # V4L2 MJPG 原生 JPEG 字节
                    probe = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    if probe is None:
                        break
                    h, w = probe.shape[:2]
                    jpg = frame.tobytes()
                with self._lock:
                    self._jpeg = jpg
                    self._frame_shape = (h, w)
                    self.frames_captured += 1
                self.ok = True
                ret, frame = cap.read()
        except Exception as e:
            self.err = f"相机线程异常: {e}"
        finally:
            if cap is not None:
                cap.release()

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def shape(self):
        with self._lock:
            return self._frame_shape


# ---------------------------------------------------------------- 录制（机器人端）


def record_ros2(args):
    """在机器人上录制。ROS 依赖全部延迟导入，PC 端 --convert 模式不需要。"""
    import rclpy
    from rclpy.node import Node
    from interface_pkg.msg import RealtimeJointCmd, Robotstatus, MotFeedback

    class F1Recorder(Node):
        def __init__(self):
            super().__init__("f1_record_ros2")
            self.frames = []
            self.start_time = time.time()

            # 附加观测的最新值缓存（不属于主 action 空间，随帧保存）
            self.latest_status = None
            self.latest_motor = None

            # 左右臂最后已知指令。PICO 若只发单臂（robot_id=1/2），
            # 另一侧填 NaN 而不是 0 —— 0 是"回零位"，会污染数据
            self.last_cmd_left = None
            self.last_cmd_right = None
            self.seen_robot_ids = set()

            self.sub_cmd = self.create_subscription(
                RealtimeJointCmd, "/realtime_joint_cmd", self.cmd_callback, 50)
            self.sub_status = self.create_subscription(
                Robotstatus, "/arm_status", self.status_callback, 10)
            self.sub_motor = self.create_subscription(
                MotFeedback, "/motor_feedback", self.motor_callback, 10)

            self.min_interval = 1.0 / args.fps if args.fps > 0 else 0.0
            self.last_frame_time = 0.0
            self.frame_no = 0
            self._last_autosave_frame = 0
            self._autosaved_file = None
            self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir = Path(args.out)
            self.out_dir.mkdir(parents=True, exist_ok=True)

            # 摄像头（可选）
            self.cam = None
            if args.cam:
                self.cam = CameraGrabber(args.cam)
                if self.cam.start():
                    h, w = self.cam.shape() or (0, 0)
                    self.get_logger().info(
                        f"相机就绪: {args.cam} ({w}x{h})，JPEG 随帧存储")
                else:
                    self.get_logger().warn(
                        f"相机不可用: {self.cam.err} —— 本次只录关节数据")
                    self.cam = None

            self.get_logger().info(
                f"订阅 /realtime_joint_cmd /arm_status /motor_feedback，"
                f"帧率上限 {args.fps if args.fps > 0 else '不限制'} Hz")
            self.get_logger().info(f"录制中... Ctrl+C 停止。输出目录: {self.out_dir}")

        # ---- 回调 ----
        def status_callback(self, msg):
            self.latest_status = msg

        def motor_callback(self, msg):
            self.latest_motor = msg

        def cmd_callback(self, msg):
            now = time.time()
            if now - self.last_frame_time < self.min_interval:
                return
            self.last_frame_time = now

            self.seen_robot_ids.add(int(msg.robot_id))

            left = np.asarray(msg.left_joints, dtype=np.float32)
            right = np.asarray(msg.right_joints, dtype=np.float32)

            # 按 robot_id 更新对应侧的最后已知指令（0=双臂 1=左 2=右）
            if msg.robot_id in (RealtimeJointCmd.ROBOT_LEFT, RealtimeJointCmd.ROBOT_BOTH):
                self.last_cmd_left = pad7(left, fill=np.nan)
            if msg.robot_id in (RealtimeJointCmd.ROBOT_RIGHT, RealtimeJointCmd.ROBOT_BOTH):
                self.last_cmd_right = pad7(right, fill=np.nan)

            l = self.last_cmd_left if self.last_cmd_left is not None \
                else np.full(7, np.nan, dtype=np.float32)
            r = self.last_cmd_right if self.last_cmd_right is not None \
                else np.full(7, np.nan, dtype=np.float32)

            joints = np.concatenate([r, l])  # 14 DOF，右臂在前

            frame = {
                "ts": now,
                "joints_rad": joints,
                "duration": float(msg.duration),
                "robot_id": int(msg.robot_id),
            }

            # 机载画面：随关节帧同步抓最新一张
            if self.cam is not None:
                jpg = self.cam.latest_jpeg()
                if jpg is not None:
                    frame["cam_jpeg"] = jpg

            # 手臂反馈（单位以 Robotstatus.msg 注释为准：度或弧度未明确，原始保存仅供核对）
            if self.latest_status is not None:
                s = self.latest_status
                frame["robot_left_joint_positions"] = pad7(s.left_joint_positions)
                frame["robot_right_joint_positions"] = pad7(s.right_joint_positions)
                frame["is_alarming"] = bool(s.is_alarming)
                frame["joint_move_complete"] = bool(s.joint_move_complete)
                frame["global_speed_ratio"] = float(s.global_speed_ratio)

            # 躯干反馈（腰 3 + 头 2 + 升降 1，仅供将来扩展 17-DOF 用）
            if self.latest_motor is not None:
                m = self.latest_motor
                frame["motor_waist_angles"] = np.array(
                    [m.back_wl_angle, m.back_wm_angle, m.back_wh_angle], dtype=np.float32)
                frame["motor_head_angles"] = np.array(
                    [m.back_hl_angle, m.back_hh_angle], dtype=np.float32)
                frame["motor_lift_position"] = np.float32(m.height_position)

            self.frames.append(frame)
            self.frame_no += 1

            if self.frame_no - self._last_autosave_frame >= 500:
                self._autosaved_file = self._autosave()
                self._last_autosave_frame = self.frame_no

            self._print_status(frame)

        # ---- 显示与保存 ----
        def _print_status(self, frame):
            elapsed = time.time() - self.start_time
            hz = self.frame_no / max(elapsed, 0.01)
            deg = np.degrees(frame["joints_rad"])
            # 只显示几个关键关节
            key = [(0, "J1R"), (3, "J4R"), (6, "J7R"), (7, "J1L"), (10, "J4L"), (13, "J7L")]
            joint_str = " ".join(f"{name}:{deg[k]:+6.1f}" for k, name in key)
            end_char = "\n" if self.frame_no % 10 == 0 else "\r"
            print(
                f"\r[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] "
                f"帧:{self.frame_no:5d} {hz:5.1f}Hz | {joint_str}",
                end=end_char, flush=True)

        def _autosave(self):
            """自动保存 —— 防止进程被强杀丢数据"""
            if not self.frames:
                return None
            path, _ = self._save_frames(suffix="_autosave")
            print(f"\n[AUTO] 已自动保存 {len(self.frames)} 帧 → {path.name}", flush=True)
            return path

        def _save_frames(self, suffix=""):
            """把 frames 写成 HDF5（h5py 缺失时降级 npz + json 元数据）"""
            n = len(self.frames)
            joints = np.array([f["joints_rad"] for f in self.frames], dtype=np.float32)
            actions = np.zeros_like(joints)
            actions[:-1] = joints[1:]
            actions[-1] = joints[-1]  # 末帧 action = 自身

            elapsed = self.frames[-1]["ts"] - self.frames[0]["ts"] if n > 1 else 0
            fps = (n - 1) / elapsed if elapsed > 0.01 else float(args.fps or 0)

            attrs = {
                "total_frames": n,
                "fps": float(fps),
                "recorded_at": datetime.now().isoformat(),
                "source": "f1_record_ros2",
                "joint_names": json.dumps(JOINT_NAMES_14),
                "joint_units": "rad",
                "joint_order": "right_arm(7) + left_arm(7)",
                "seen_robot_ids": json.dumps(sorted(self.seen_robot_ids)),
                "cam_source": self.cam.source if self.cam else "",
                "cam_format": "jpeg",
                "note": "action[i]=state[i+1]; 躯干不在此数据集内，见 motor_* 字段",
            }

            base = self.out_dir / f"session_{self.ts}{suffix}"

            try:
                import h5py
                with h5py.File(str(base) + ".h5", "w") as f:
                    f.create_dataset("timestamp", data=np.array(
                        [fr["ts"] - self.start_time for fr in self.frames], dtype=np.float32))
                    f.create_dataset("observation.state", data=joints)
                    f.create_dataset("action", data=actions)
                    f.create_dataset("duration", data=np.array(
                        [fr["duration"] for fr in self.frames], dtype=np.float32))
                    f.create_dataset("robot_id", data=np.array(
                        [fr["robot_id"] for fr in self.frames], dtype=np.int16))
                    # 机载画面（JPEG 字节，变长存储）
                    if "cam_jpeg" in self.frames[0]:
                        jpeg_dt = h5py.vlen_dtype(np.dtype("uint8"))
                        ds = f.create_dataset(CAM_KEY, (n,), dtype=jpeg_dt)
                        for i, fr in enumerate(self.frames):
                            ds[i] = np.frombuffer(fr["cam_jpeg"], dtype=np.uint8)
                        h, w = self.cam.shape() or (0, 0)
                        attrs["cam_width"], attrs["cam_height"] = w, h
                    # 附加观测（可能缺失的字段跳过）
                    if "robot_left_joint_positions" in self.frames[0]:
                        f.create_dataset("observation.robot_left_joint_positions", data=np.array(
                            [fr["robot_left_joint_positions"] for fr in self.frames], dtype=np.float32))
                        f.create_dataset("observation.robot_right_joint_positions", data=np.array(
                            [fr["robot_right_joint_positions"] for fr in self.frames], dtype=np.float32))
                    if "motor_waist_angles" in self.frames[0]:
                        f.create_dataset("observation.motor_waist_angles", data=np.array(
                            [fr["motor_waist_angles"] for fr in self.frames], dtype=np.float32))
                        f.create_dataset("observation.motor_head_angles", data=np.array(
                            [fr["motor_head_angles"] for fr in self.frames], dtype=np.float32))
                        f.create_dataset("observation.motor_lift_position", data=np.array(
                            [fr["motor_lift_position"] for fr in self.frames], dtype=np.float32))
                    for k, v in attrs.items():
                        f.attrs[k] = v
                return base.with_suffix(".h5"), "h5"
            except ImportError:
                # 机器人上没有 h5py —— 降级 npz（numpy 自带）+ json 元数据
                if self.cam is not None:
                    print("[WARN] 没装 h5py，机载画面无法存储，本次只保存关节数据")
                    print("[WARN] 装法: pip install h5py")
                np.savez(
                    str(base) + ".npz",
                    timestamp=np.array([fr["ts"] - self.start_time for fr in self.frames], dtype=np.float32),
                    state=joints, action=actions)
                meta = {k: v for k, v in attrs.items()}
                meta["format"] = "npz"
                (base.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                return base.with_suffix(".npz"), "npz"

        def finalize(self):
            """停止后保存最终文件并打印统计"""
            print()
            n = len(self.frames)
            if n == 0:
                self.get_logger().error(
                    "没收到任何 /realtime_joint_cmd 数据，不保存。排查：\n"
                    "  1. pico_teleop_node 是否在跑（ros2 node list）\n"
                    "  2. PICO 是否连上同一网络并在发数据（节点日志有 [RT] 打印）\n"
                    "  3. 若机器人是直接控制模式（发的是 /arm_move_joint 而不是 "
                    "/realtime_joint_cmd），本脚本录不到，需要改订阅")
                return None

            path, fmt = self._save_frames(suffix="")
            joints = np.array([f["joints_rad"] for f in self.frames], dtype=np.float32)
            nan_frames = int(np.isnan(joints).any(axis=1).sum())
            elapsed = time.time() - self.start_time
            print(f"[REC] 总计: {n} 帧, {elapsed:.1f} 秒, {n / max(elapsed, 0.01):.1f} Hz")
            print(f"[REC] 已保存 ({fmt}): {path}")
            if self.cam is not None:
                n_img = sum(1 for fr in self.frames if "cam_jpeg" in fr)
                print(f"[CAM] 画面帧: {n_img}/{n}（相机共抓 {self.cam.frames_captured} 张）")
            if nan_frames:
                print(f"[WARN] {nan_frames} 帧含 NaN（单臂未收到指令），转 LeRobot 时注意")
            if 0 not in self.seen_robot_ids:
                print(f"[WARN] 全程没收到双臂指令 (robot_id=0)，见过: {sorted(self.seen_robot_ids)}")

            # 删除 autosave 临时文件
            if self._autosaved_file and self._autosaved_file.exists():
                self._autosaved_file.unlink()
                print(f"[CLEAN] 已删除自动保存临时文件: {self._autosaved_file.name}")
            return path

    rclpy.init()
    node = F1Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        result = node.finalize()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # SIGINT 已触发过一次 shutdown，二次调用抛 RCLError，属正常
    return result


# ---------------------------------------------------------------- 转换（PC 端）


def convert_h5_to_lerobot(h5_path, dataset_name, out_dir, video_path=None,
                          task="teleop manipulation", nan_mode="keep", force=False):
    """PC 端：把录制的 HDF5 转成 LeRobot 数据集（不需要 ROS）。
    HDF5 带机载画面时自动解码进数据集（视觉与关节天然同步）；
    没画面时给 --video 则用录屏均匀采样对齐。

    nan_mode: keep=保留 NaN（默认，仅告警）/ drop=剔除含 NaN 帧 / interp=按关节列插值
    force: 目标数据集已存在时是否强制覆盖（默认拒绝，防止同名转换删旧数据）
    """
    import h5py

    cam_jpegs = None
    with h5py.File(str(h5_path), "r") as f:
        joints = f["observation.state"][:].astype(np.float32)
        actions = f["action"][:].astype(np.float32)
        attrs = dict(f.attrs)
        if CAM_KEY in f:
            cam_jpegs = [bytes(f[CAM_KEY][i]) for i in range(f[CAM_KEY].shape[0])]

    n = joints.shape[0]
    fps = float(attrs.get("fps", 30))

    nan_frames = int(np.isnan(joints).any(axis=1).sum())
    if nan_frames:
        print(f"[WARN] {nan_frames}/{n} 帧含 NaN 关节角（单臂未收到指令）")

    # ---- NaN 处理：剔除 / 插值 / 保留 ----
    if nan_mode == "drop" and nan_frames:
        keep = ~np.isnan(joints).any(axis=1)
        joints = joints[keep]
        actions = actions[keep]
        if cam_jpegs is not None:
            cam_jpegs = [j for j, k in zip(cam_jpegs, keep) if k] or None
        print(f"[NAN] drop: 剔除 {n - int(keep.sum())} 帧，剩 {int(keep.sum())} 帧")
    elif nan_mode == "interp" and nan_frames:
        for j in range(N_JOINTS):
            col = joints[:, j]
            bad = np.isnan(col)
            if bad.any() and (~bad).any():
                col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
        # 插值后按录制契约重建 action（action[i] = state[i+1]）
        actions = np.vstack([joints[1:], joints[-1:]])
        remain = int(np.isnan(joints).any(axis=1).sum())
        if remain:
            print(f"[WARN] interp 后仍有 {remain} 帧含 NaN（整列缺失，该臂全程无指令），保留原值")
        else:
            print(f"[NAN] interp: 已按关节列插值 {nan_frames} 帧 NaN")
    elif nan_frames:
        print("[NAN] keep: NaN 原样进数据集 —— 训练前需自行清洗")
    n = joints.shape[0]
    if n == 0:
        print("[ERR] NaN 处理后没有可用帧")
        return None

    features = {
        "observation.state": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES_14},
        "action": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES_14},
    }

    # ---- 视觉来源：HDF5 机载画面优先，--video 录屏其次 ----
    images = None
    if cam_jpegs is not None:
        import cv2
        if len(cam_jpegs) < n:        # 相机启动滞后，前几帧无画面 → 用首帧填充
            pad = n - len(cam_jpegs)
            print(f"[CAM] 前 {pad} 帧无画面（相机启动滞后），用第一帧填充")
            cam_jpegs = [cam_jpegs[0]] * pad + cam_jpegs
        print(f"[CAM] HDF5 内机载画面 {len(cam_jpegs)} 帧，解码中...")
        images = []
        for jpg in cam_jpegs:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = images[0].shape[:2]
        print(f"[CAM] 解码完成: {w}x{h}")
        if video_path:
            print("[WARN] HDF5 已有机载画面，忽略 --video")
        features[CAM_KEY] = {
            "dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"],
        }
    elif video_path:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, f0 = cap.read()
        if not ret or vid_frames == 0:
            print(f"[ERR] 视频读不了: {video_path}")
            return None
        h, w = f0.shape[:2]
        step = vid_frames / n
        images = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
            ret, fr = cap.read()
            if not ret:
                images.append(images[-1].copy())  # 尾部兜底：复用上一帧
            else:
                images.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"[VIDEO] 抽了 {len(images)} 帧 ({w}x{h}), 视频共 {vid_frames} 帧")
        features[CAM_KEY] = {
            "dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"],
        }

    ds_dir = Path(out_dir) / dataset_name
    if ds_dir.exists() and not force:
        print(f"[ERR] 数据集已存在: {ds_dir} —— 直接转换会覆盖旧数据")
        print("      换一个 --dataset 名，或确认要覆盖时加 --force")
        return None
    if ds_dir.exists() and force:
        import shutil
        shutil.rmtree(ds_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("[ERR] 没装 lerobot（pip install lerobot），无法生成数据集")
        return None

    ds = LeRobotDataset.create(
        repo_id=dataset_name, fps=int(round(fps)), features=features,
        root=str(ds_dir), robot_type="F1", use_videos=False,
    )
    for i in range(n):
        frame = {
            "observation.state": joints[i].astype(np.float32),
            "action": actions[i].astype(np.float32),
            "task": task,
        }
        if images is not None:
            frame[CAM_KEY] = images[i]
        ds.add_frame(frame)
    # lerobot 0.4.x 的 save_episode() 不接受 task 参数，task 由 add_frame 逐帧写入
    ds.save_episode()
    ds.finalize()
    print(f"[LeRobot] 已生成: {ds_dir}/  ({n} 帧, {fps:.1f} fps)")
    return str(ds_dir)


# ---------------------------------------------------------------- 入口

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 PICO 遥操录制（ROS2）")
    parser.add_argument("--fps", type=int, default=50, help="录制帧率上限，0=不限制")
    parser.add_argument("--cam", type=str, default=None,
                        help="机载摄像头：V4L2 设备（/dev/video6）或 RTSP URL，画面与关节帧同步存储")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称")
    parser.add_argument("--out", type=str, default="./recordings", help="输出目录")
    parser.add_argument("--convert", type=str, default=None,
                        help="PC 端模式：指定 HDF5 路径，转成 LeRobot 数据集后退出")
    parser.add_argument("--video", type=str, default=None,
                        help="PC 端模式：PICO 录屏视频，HDF5 无画面时与 --convert 一起用")
    parser.add_argument("--task", type=str, default="teleop manipulation",
                        help="任务描述（进 tasks.jsonl，训练 prompt 用）")
    parser.add_argument("--nan", choices=["keep", "drop", "interp"], default="keep",
                        help="NaN 帧处理：keep=保留(默认) drop=剔除 interp=按列插值")
    parser.add_argument("--force", action="store_true",
                        help="目标数据集已存在时强制覆盖（默认拒绝，防同名转换删旧数据）")
    args = parser.parse_args()

    if args.convert:
        h5 = Path(args.convert)
        if not h5.exists():
            print(f"[ERR] 文件不存在: {h5}")
            raise SystemExit(1)
        name = args.dataset or f"f1_ros2_{h5.stem.removeprefix('session_')}"
        convert_h5_to_lerobot(h5, name, args.out, video_path=args.video,
                              task=args.task, nan_mode=args.nan, force=args.force)
    else:
        record_ros2(args)
