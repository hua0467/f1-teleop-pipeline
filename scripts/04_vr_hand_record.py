"""
F1 遥操数据采集管线 —— VR 手部追踪录制

从 Quest 3 (hand-tracking-streamer APK) 接收手部追踪数据，
录制到 HDF5 文件。

依赖:
    pip install hand-tracking-sdk

使用方式:
    # 1. 在 Quest 3 上启动 hand-tracking-streamer APK
    #    - 在头显内配置 PC 的 IP 地址和端口(默认 9000)
    #    - 选择双手追踪模式
    # 2. 在 PC 上运行:
    python 04_vr_hand_record.py --episode-name pick_cube_001 --fps 30 --duration 0
"""

import h5py
import numpy as np
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput


class VREpisodeRecorder:
    """从 VR 手部追踪录制遥操演示"""

    def __init__(self, output_dir="./recordings", fps=30, listen_port=9000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # 手部追踪客户端
        self.client = HTSClient(
            HTSClientConfig(
                transport_mode="udp",
                host="0.0.0.0",
                port=listen_port,
                output=StreamOutput.FRAMES,
                hand_filter="both",
                error_policy="tolerant",  # 容忍丢包,不崩溃
            )
        )

        # 21 个手部关节点 + 1 个手腕 = 22 个关键点, 每点 3D 坐标
        self.JOINT_NAMES = [j.value for j in JointName]
        self.N_JOINTS = len(self.JOINT_NAMES)  # 21 joints + wrist = 22 per hand

        self.frames = {
            "timestamp": [],
            # 左手数据
            "observation.left_hand.wrist_pose": [],    # 7: x,y,z,qx,qy,qz,qw
            "observation.left_hand.landmarks": [],     # 22*3 = 66: wrist + 21 joints
            # 右手数据
            "observation.right_hand.wrist_pose": [],   # 7
            "observation.right_hand.landmarks": [],    # 66
            # 机器人状态和动作 (暂时留空,接入 F1 后填充)
            "observation.state": [],
            "action": [],
            "observation.gripper_cmd": [],
            # LeRobot 元数据
            "episode_index": [],
            "task_index": [],
            "next.done": [],
            "next.reward": [],
        }
        self.latest_left_frame = None
        self.latest_right_frame = None
        self.recording = False
        self.episode_idx = 0
        self._frame_count = 0
        self._dof = 6  # F1 机械臂自由度,接入实际机器人后调整

    def _extract_hand_data(self, frame, side: str):
        """从一个 HandFrame 中提取手腕姿态 + 全部 landmarks"""
        if frame is None:
            wrist = np.zeros(7, dtype=np.float32)
            landmarks = np.zeros(self.N_JOINTS * 3, dtype=np.float32)
            return wrist, landmarks

        w = frame.wrist
        wrist = np.array([w.x, w.y, w.z, w.qx, w.qy, w.qz, w.qw], dtype=np.float32)

        landmarks = np.zeros(self.N_JOINTS * 3, dtype=np.float32)
        for i, name in enumerate(self.JOINT_NAMES):
            joint = frame.get_joint(JointName(name))
            landmarks[i * 3:i * 3 + 3] = joint

        return wrist, landmarks

    def start_episode(self, task_index=0):
        """开始新演示"""
        self.frames = {k: [] for k in self.frames}
        self.latest_left_frame = None
        self.latest_right_frame = None
        self.recording = True
        self._frame_count = 0
        self._start_time = time.time()
        print(f"\n[REC] Episode {self.episode_idx} started (task: {task_index})")
        print(f"   Listening on UDP port, waiting for Quest 3 data...")
        print(f"   Press Ctrl+C to stop\n")

    def _update_frames(self):
        """从 VR 客户端拉取最新帧，非阻塞"""
        try:
            for frame in self.client.iter_events():
                if frame.side.value == "left":
                    self.latest_left_frame = frame
                elif frame.side.value == "right":
                    self.latest_right_frame = frame
        except Exception:
            pass  # 超时或无数据,忽略

    def record_frame(self):
        """录制一帧(当前 VR 手部状态)"""
        if not self.recording:
            return

        self._update_frames()

        # 提取手部数据
        left_wrist, left_landmarks = self._extract_hand_data(self.latest_left_frame, "left")
        right_wrist, right_landmarks = self._extract_hand_data(self.latest_right_frame, "right")

        # 简单夹爪控制: 根据拇指和食指指尖距离估算(0=闭合,1=全开)
        left_gripper = self._estimate_gripper(self.latest_left_frame) if self.latest_left_frame else 0.5
        right_gripper = self._estimate_gripper(self.latest_right_frame) if self.latest_right_frame else 0.5
        gripper_cmd = max(left_gripper, right_gripper)

        t = time.time() - self._start_time

        self.frames["timestamp"].append(t)
        self.frames["observation.left_hand.wrist_pose"].append(left_wrist)
        self.frames["observation.left_hand.landmarks"].append(left_landmarks)
        self.frames["observation.right_hand.wrist_pose"].append(right_wrist)
        self.frames["observation.right_hand.landmarks"].append(right_landmarks)

        # 机器人状态和动作 (TODO: 接入 IK 解算和 F1 机器人后替换)
        self.frames["observation.state"].append(np.zeros(self._dof, dtype=np.float32))
        self.frames["action"].append(np.zeros(self._dof, dtype=np.float32))
        self.frames["observation.gripper_cmd"].append(gripper_cmd)

        self.frames["episode_index"].append(self.episode_idx)
        self.frames["task_index"].append(0)
        self.frames["next.done"].append(False)
        self.frames["next.reward"].append(0.0)

        self._frame_count += 1

        # 帧率控制
        elapsed = time.time() - self._start_time
        expected = self._frame_count * self.frame_interval
        if elapsed < expected:
            time.sleep(expected - elapsed)

    def _estimate_gripper(self, hand_frame) -> float:
        """根据拇指尖和食指尖距离估算夹爪开合度"""
        try:
            thumb_tip = np.array(hand_frame.get_joint(JointName.ThumbTip))
            index_tip = np.array(hand_frame.get_joint(JointName.IndexTip))
            dist = np.linalg.norm(thumb_tip - index_tip)
            # 经验映射: 0~5cm → 0~1 (闭合→全开)
            return float(np.clip(dist / 0.05, 0.0, 1.0))
        except Exception:
            return 0.5

    def stop_episode(self, success=True):
        """结束当前演示"""
        self.recording = False

        if len(self.frames["timestamp"]) == 0:
            print("[WARN] No frames recorded, skipping save")
            return None

        self.frames["next.done"][-1] = True
        if success:
            self.frames["next.reward"][-1] = 1.0

        h5_path = self.output_dir / f"episode_{self.episode_idx:06d}.h5"

        with h5py.File(str(h5_path), "w") as f:
            for key, values in self.frames.items():
                if len(values) > 0:
                    f.create_dataset(key, data=np.array(values))

            f.attrs["fps"] = self.fps
            f.attrs["total_frames"] = self._frame_count
            f.attrs["duration_sec"] = self.frames["timestamp"][-1]
            f.attrs["robot_type"] = "F1"
            f.attrs["source"] = "Quest3_hand_tracking"
            f.attrs["recorded_at"] = datetime.now().isoformat()

        duration = self.frames["timestamp"][-1]
        print(f"\n[STOP] Episode {self.episode_idx} saved")
        print(f"   File: {h5_path}")
        print(f"   Frames: {self._frame_count} | Duration: {duration:.1f}s | "
              f"FPS: {self._frame_count/duration:.1f}")

        self.episode_idx += 1
        return str(h5_path)


# ===== 主入口 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 VR 手部追踪录制")
    parser.add_argument("--episode-name", type=str, default="test_vr",
                       help="演示名称")
    parser.add_argument("--fps", type=int, default=30,
                       help="录制帧率")
    parser.add_argument("--duration", type=float, default=0,
                       help="录制时长（秒），0=手动 Ctrl+C 停止")
    parser.add_argument("--output", type=str, default="./recordings",
                       help="输出目录")
    parser.add_argument("--port", type=int, default=9000,
                       help="UDP 监听端口 (需与 Quest 3 APK 配置一致)")
    args = parser.parse_args()

    recorder = VREpisodeRecorder(
        output_dir=args.output,
        fps=args.fps,
        listen_port=args.port,
    )

    recorder.start_episode()

    try:
        t0 = time.time()
        while True:
            recorder.record_frame()

            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n[WARN] Manual interrupt")

    recorder.stop_episode()
    print("\n[OK] VR recording done")
