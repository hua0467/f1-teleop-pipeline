"""
F1 遥操数据采集管线 —— 数据录制模块

从 VR 手部追踪 + 机器人关节状态 → HDF5 文件

使用方式：
    python 02_record_episode.py --episode-name pick_cube_001 --fps 30 --duration 0

如果 duration=0，手动按 Ctrl+C 结束录制。
"""

import h5py
import numpy as np
import time
import argparse
import json
from pathlib import Path
from datetime import datetime


class EpisodeRecorder:
    """录制一条完整的遥操演示"""

    def __init__(self, output_dir="./recordings", fps=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_interval = 1.0 / fps

        self.frames = {
            "timestamp": [],
            "observation.state": [],       # 当前关节角度
            "action": [],                   # 目标关节角度
            "observation.hand_pose": [],    # VR 手部 6D 位姿
            "observation.gripper_cmd": [],  # 夹爪指令 (0~1)
            "episode_index": [],
            "task_index": [],
            "next.done": [],
            "next.reward": [],
        }
        self.recording = False
        self.episode_idx = 0
        self._frame_count = 0

    def start_episode(self, task_index=0):
        """开始新的一条演示"""
        self.frames = {k: [] for k in self.frames}
        self.recording = True
        self._frame_count = 0
        self._start_time = time.time()
        print(f"\n[REC] Episode {self.episode_idx} started (task: {task_index})")
        print(f"   Press Ctrl+C to stop\n")

    def record_frame(self, joint_state, target_action, hand_pose=None, gripper_cmd=None):
        """
        录一帧数据

        参数:
            joint_state:  list[float], 当前机器人关节角度
            target_action: list[float], IK 输出的目标关节角度
            hand_pose:     list[float], VR 手部 6D 位姿 (x,y,z,rx,ry,rz) 或 None
            gripper_cmd:   float, 夹爪开合指令 (0=闭合, 1=全开) 或 None
        """
        if not self.recording:
            return

        self.frames["timestamp"].append(time.time() - self._start_time)
        self.frames["observation.state"].append(np.array(joint_state, dtype=np.float32))
        self.frames["action"].append(np.array(target_action, dtype=np.float32))
        self.frames["observation.hand_pose"].append(
            np.array(hand_pose, dtype=np.float32) if hand_pose is not None
            else np.zeros(6, dtype=np.float32)
        )
        self.frames["observation.gripper_cmd"].append(
            gripper_cmd if gripper_cmd is not None else 0.0
        )
        self.frames["episode_index"].append(self.episode_idx)
        self.frames["task_index"].append(0)
        self.frames["next.done"].append(False)
        self.frames["next.reward"].append(0.0)

        self._frame_count += 1

        # 控制帧率
        elapsed = time.time() - self._start_time
        expected = self._frame_count * self.frame_interval
        if elapsed < expected:
            time.sleep(expected - elapsed)

    def stop_episode(self, success=True):
        """结束当前演示，保存到 HDF5"""
        self.recording = False

        if len(self.frames["timestamp"]) == 0:
            print("[WARN] No data, skipping save")
            return None

        # 标记最后一帧
        self.frames["next.done"][-1] = True
        if success:
            self.frames["next.reward"][-1] = 1.0

        # 保存为 HDF5
        h5_path = self.output_dir / f"episode_{self.episode_idx:06d}.h5"

        with h5py.File(str(h5_path), "w") as f:
            for key, values in self.frames.items():
                if len(values) > 0:
                    f.create_dataset(key, data=np.array(values))

            # 元数据
            f.attrs["fps"] = self.fps
            f.attrs["total_frames"] = self._frame_count
            f.attrs["duration_sec"] = self.frames["timestamp"][-1]
            f.attrs["robot_type"] = "F1"
            f.attrs["recorded_at"] = datetime.now().isoformat()

        duration = self.frames["timestamp"][-1]
        print(f"\n[STOP] Episode {self.episode_idx} saved")
        print(f"   File: {h5_path}")
        print(f"   Frames: {self._frame_count} | Duration: {duration:.1f}s | "
              f"FPS: {self._frame_count/duration:.1f}")

        self.episode_idx += 1
        return str(h5_path)


# ===== 测试用 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 遥操数据录制")
    parser.add_argument("--episode-name", type=str, default="test",
                       help="这条演示的名称")
    parser.add_argument("--fps", type=int, default=30,
                       help="录制帧率")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="录制时长（秒），0=手动 Ctrl+C 停止")
    parser.add_argument("--output", type=str, default="./recordings",
                       help="输出目录")
    args = parser.parse_args()

    recorder = EpisodeRecorder(output_dir=args.output, fps=args.fps)

    # 模拟录制（实际使用时，数据来自 VR + 机器人）
    recorder.start_episode()

    try:
        t0 = time.time()
        dof = 6  # F1 机械臂自由度，根据实际情况改

        while True:
            # ===== 这里替换为真实的 VR + 机器人数据 =====
            t = time.time() - t0
            joint_state = [np.sin(t * 2 + i) * 0.5 for i in range(dof)]       # 模拟
            target_action = [np.sin(t * 2 + i + 0.1) * 0.5 for i in range(dof)] # 模拟
            hand_pose = [0.5, 0.2, 0.3 + np.sin(t)*0.1, 0, 0, 0]              # 模拟
            gripper_cmd = 0.5 + 0.3 * np.sin(t * 3)                            # 模拟
            # =============================================

            recorder.record_frame(joint_state, target_action, hand_pose, gripper_cmd)

            if args.duration > 0 and t >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n[WARN] Manual interrupt")

    recorder.stop_episode()
    print("\n[OK] Recording test done (simulated data, replace with real data source)")
