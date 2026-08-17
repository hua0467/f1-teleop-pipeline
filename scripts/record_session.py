"""
最简录制脚本 —— 只做一件事：收 UDP → 解 IK → 存 HDF5

用法:
  python scripts/record_session.py

运行后在终端看到帧数滚动就说明在录制了，Ctrl+C 停止。
录制完成后自动生成含视觉的 LeRobot 数据集（需要 -v 指定视频）。

选项:
  --no-ik     跳过 IK，只存原始手部位姿
  --video MP4 录制停止后自动对齐视频、生成含视觉的 LeRobot
  --dataset N  数据集名称（默认 f1_session_YYYYMMDD_HHMMSS）
"""

from __future__ import annotations

import argparse
import atexit
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- 路径 ----
BASE_DIR = Path(__file__).parent.parent
RECORDINGS = BASE_DIR / "recordings"
VIDEOS_DIR = BASE_DIR / "videos"
DATASETS = BASE_DIR / "datasets"
RECORDINGS.mkdir(parents=True, exist_ok=True)


# ---- IK ----
class IKSolver:
    def __init__(self, urdf_path: str | None = None):
        if urdf_path is None:
            urdf_path = str(BASE_DIR / "urdf" / "urdf" / "F1_URDF_V04.urdf")
        self.urdf = urdf_path
        from ik_solver import F1Kinematics, hand_pose_to_robot_target
        self.F1Kinematics = F1Kinematics
        self.hand_pose_to_robot_target = hand_pose_to_robot_target
        self.ik = F1Kinematics(urdf_path)
        self.right_guess = None
        self.left_guess = None
        self.n_torso = self.ik.n_torso

    def solve(self, right_wrist, left_wrist):
        try:
            ra = np.zeros(self.ik.n_right)
            r_pos = np.array(right_wrist[:3])
            r_quat = np.array(right_wrist[3:7]) if len(right_wrist) >= 7 else np.array([0, 0, 0, 1])
            right_ok = False
            if np.linalg.norm(r_pos) > 0.001:
                tp, tq = self.hand_pose_to_robot_target(r_pos, r_quat)
                ra = self.ik.solve_ik(tp, tq, side="right", initial_guess=self.right_guess)
                self.right_guess = ra.copy()
                right_ok = True

            la = np.zeros(self.ik.n_left)
            l_pos = np.array(left_wrist[:3])
            l_quat = np.array(left_wrist[3:7]) if len(left_wrist) >= 7 else np.array([0, 0, 0, 1])
            left_ok = False
            if np.linalg.norm(l_pos) > 0.001:
                tp, tq = self.hand_pose_to_robot_target(l_pos, l_quat)
                la = self.ik.solve_ik(tp, tq, side="left", initial_guess=self.left_guess)
                self.left_guess = la.copy()
                left_ok = True

            torso = (ra[:self.n_torso] + la[:self.n_torso]) / 2.0
            joints = np.concatenate([torso, ra[self.n_torso:], la[self.n_torso:]])
            return {
                "joints_rad": joints,
                "joints_deg": np.degrees(joints),
                "right_wrist": r_pos.tolist(),
                "left_wrist": l_pos.tolist(),
                "right_ok": right_ok, "left_ok": left_ok,
            }
        except Exception as e:
            return {"error": str(e)}


# ---- 主录制逻辑 ----
def record_session(args):
    print("=" * 50)
    print("F1 VR 同步录制")
    print("=" * 50)

    # 初始化 IK
    ik = None
    if not args.no_ik:
        try:
            ik = IKSolver()
            print(f"[IK] 已加载 (URDF: {ik.urdf})")
        except Exception as e:
            print(f"[IK] 加载失败: {e}")
            ik = None

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)
    print(f"[UDP] 监听端口 {args.port}")

    # 录制状态
    frames = []
    right_wrist = [0.0] * 7
    left_wrist = [0.0] * 7
    start_time = time.time()
    frame_no = 0
    right_ok = 0
    left_ok = 0
    last_joints_deg = None

    print(f"\n[REC] 录制中... (帧率限制: {args.fps} fps)")
    print("[REC] 按 Ctrl+C 停止\n")

    def fmt_time(s):
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    min_interval = 1.0 / args.fps
    last_frame_time = 0
    _shutdown_flag = False

    def pad7(arr):
        a = np.array(arr, dtype=np.float32)
        if len(a) >= 7:
            return a[:7]
        p = np.zeros(7, dtype=np.float32)
        p[: len(a)] = a
        return p

    # 自动保存函数 —— 防止进程被强杀丢数据
    def _autosave(suffix="_autosave"):
        nonlocal frames
        if not frames:
            return
        n = len(frames)
        joints_rad = np.array([f["joints_rad"] for f in frames], dtype=np.float32)
        actions = np.zeros_like(joints_rad)
        actions[:-1] = joints_rad[1:]
        actions[-1] = joints_rad[-1]
        rw = np.array([pad7(f["right_wrist_raw"]) for f in frames], dtype=np.float32)
        lw = np.array([pad7(f["left_wrist_raw"]) for f in frames], dtype=np.float32)
        h5_tmp = RECORDINGS / f"session_{ts}{suffix}.h5"
        import h5py
        with h5py.File(str(h5_tmp), "w") as f:
            f.create_dataset("timestamp", data=np.array([fr["ts"] - start_time for fr in frames], dtype=np.float32))
            f.create_dataset("observation.state", data=joints_rad)
            f.create_dataset("action", data=actions)
            f.create_dataset("observation.right_hand.wrist_pose", data=rw)
            f.create_dataset("observation.left_hand.wrist_pose", data=lw)
            f.attrs["total_frames"] = n
            f.attrs["fps"] = n / max(time.time() - start_time, 0.01)
            f.attrs["recorded_at"] = datetime.now().isoformat()
            f.attrs["source"] = "record_session_autosave"
        return h5_tmp, n

    def _on_shutdown(sig=None, frame=None):
        nonlocal _shutdown_flag
        _shutdown_flag = True

    # Windows 下 SIGTERM 可能不可用，用 SIGBREAK 兜底
    for sig_name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_shutdown)
            except (ValueError, OSError):
                pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _last_autosave_frame = 0

    try:
        while not _shutdown_flag:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            text = data.decode("utf-8", errors="ignore").strip()
            for line in text.split("\n"):
                if "|" not in line or ":" not in line:
                    continue

                # 解析 "Left wrist|1858.7769:x,y,z,qx,qy,qz,qw"
                meta, vals_str = line.split(":", 1)
                parts = meta.split(" ")
                if len(parts) < 2:
                    continue
                side = parts[0]
                ptype = parts[1]

                if side not in ("Left", "Right") or ptype != "wrist":
                    continue

                vals = [float(x) for x in vals_str.strip().strip(",").split(",") if x.strip()]
                if len(vals) < 7:
                    continue

                if side == "Right":
                    right_wrist = vals[:7]
                else:
                    left_wrist = vals[:7]

            # 帧率限制
            now = time.time()
            if now - last_frame_time < min_interval:
                continue
            last_frame_time = now

            frame_no += 1

            # IK 求解
            ik_result = None
            if ik:
                ik_result = ik.solve(right_wrist, left_wrist)
                if ik_result and "error" not in ik_result:
                    last_joints_deg = ik_result["joints_deg"]
                    if ik_result["right_ok"]:
                        right_ok += 1
                    if ik_result["left_ok"]:
                        left_ok += 1

            # 存帧
            frame = {
                "ts": now,
                "right_wrist_raw": right_wrist[:],
                "left_wrist_raw": left_wrist[:],
            }
            if ik_result and "joints_rad" in ik_result:
                frame["joints_rad"] = ik_result["joints_rad"].tolist()
                frame["joints_deg"] = ik_result["joints_deg"].tolist()
                frame["right_ok"] = ik_result.get("right_ok", False)
                frame["left_ok"] = ik_result.get("left_ok", False)
            else:
                frame["joints_rad"] = [0.0] * 17
                frame["joints_deg"] = [0.0] * 17
                frame["right_ok"] = False
                frame["left_ok"] = False
                if ik_result and "error" in ik_result:
                    pass  # IK 偶尔失败是正常的，填 0

            frames.append(frame)

            # 每 100 帧自动保存一次
            if frame_no - _last_autosave_frame >= 100:
                h5_tmp, n_saved = _autosave()
                print(f"\n[AUTO] 已自动保存 {n_saved} 帧 → {h5_tmp.name}", flush=True)
                _last_autosave_frame = frame_no

            # 终端显示
            elapsed = now - start_time
            ik_rate = (right_ok + left_ok) / (2 * frame_no) * 100 if frame_no > 0 else 0
            joint_str = ""
            if last_joints_deg is not None:
                # 只显示几个关键关节
                key = [0, 3, 4, 5, 10, 11, 12]  # lift, J1-J3_R, J1-J3_L
                names = ["lift", "J1R", "J2R", "J3R", "J1L", "J2L", "J3L"]
                joint_str = " ".join(
                    f"{names[i]}:{last_joints_deg[k]:+6.1f}"
                    for i, k in enumerate(key) if k < len(last_joints_deg)
                )

            end_char = "\n" if frame_no % 10 == 0 else "\r"
            print(
                f"\r[{fmt_time(elapsed)}] 帧:{frame_no:5d}  "
                f"右IK:{right_ok}/{frame_no} 左IK:{left_ok}/{frame_no}  "
                f"| {joint_str}",
                end=end_char,
            )

    except (KeyboardInterrupt, SystemExit):
        print("\n\n[REC] 录制停止")
    finally:
        sock.close()

    elapsed = time.time() - start_time
    n = len(frames)
    if n == 0:
        print("[REC] 没收到任何数据，退出")
        return None

    fps_actual = n / max(elapsed, 0.01)
    print(f"[REC] 总计: {n} 帧, {elapsed:.1f} 秒, {fps_actual:.1f} fps")
    if ik:
        print(f"[REC] IK 成功率: 右 {right_ok}/{n} ({right_ok/n*100:.0f}%), 左 {left_ok}/{n} ({left_ok/n*100:.0f}%)")

    # ---- 存 HDF5（复用 autosave 逻辑）----
    h5_path, _ = _autosave(suffix="")

    # 验证 action shift
    joints_rad = np.array([f["joints_rad"] for f in frames], dtype=np.float32)
    actions = np.zeros_like(joints_rad)
    actions[:-1] = joints_rad[1:]
    actions[-1] = joints_rad[-1]
    shift_ok = np.allclose(actions[:-1], joints_rad[1:], atol=1e-5)
    shift_max = np.max(np.abs(actions[:-1] - joints_rad[1:]))
    print(f"[HDF5] 已保存: {h5_path}")
    print(f"[HDF5] action shift: {'OK' if shift_ok else 'FAIL'} (max_diff={shift_max:.2e})")

    # ---- 生成 LeRobot（纯动作）----
    dataset_name = args.dataset or f"f1_session_{ts}"
    _make_lerobot_no_vision(h5_path, dataset_name, joints_rad, actions, n)

    result = {
        "h5_path": str(h5_path),
        "dataset_name": dataset_name,
        "frames": n,
        "elapsed": elapsed,
    }

    # ---- 如果有视频，生成含视觉的 LeRobot ----
    if args.video:
        video_path = Path(args.video)
        if video_path.exists():
            vision_name = dataset_name + "_vision"
            try:
                _make_lerobot_with_vision(h5_path, video_path, vision_name, joints_rad, actions, n)
                print(f"\n[VISION] 含视觉数据集已生成: datasets/{vision_name}/")
                result["dataset_vision"] = vision_name
            except Exception as e:
                print(f"\n[VISION] 生成失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[WARN] 视频文件不存在: {args.video}")

    # 如果自动保存过，把临时文件删掉
    if _last_autosave_frame > 0:
        autosave_file = RECORDINGS / f"session_{ts}_autosave.h5"
        if autosave_file.exists():
            autosave_file.unlink()
            print(f"[CLEAN] 已删除自动保存临时文件: {autosave_file.name}")

    return result


def _make_lerobot_no_vision(h5_path, dataset_name, joints_rad, actions, n):
    """生成仅动作的 LeRobot 数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    N_JOINTS = joints_rad.shape[1]
    JOINT_NAMES = [
        "lift_joint", "waist1_joint", "waist2_joint",
        "J1_right_joint", "J2_right_joint", "J3_right_joint",
        "J4_right_joint", "J5_right_joint", "J6_right_joint", "J7_right_joint",
        "J1_left_joint", "J2_left_joint", "J3_left_joint",
        "J4_left_joint", "J5_left_joint", "J6_left_joint", "J7_left_joint",
    ][:N_JOINTS]

    features = {
        "observation.state": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES},
    }

    ds_path = DATASETS / dataset_name
    if ds_path.exists():
        import shutil
        shutil.rmtree(ds_path)

    ds = LeRobotDataset.create(
        repo_id=dataset_name, fps=30, features=features,
        root=str(ds_path), robot_type="F1", use_videos=False,
    )
    for i in range(n):
        ds.add_frame({
            "observation.state": joints_rad[i].astype(np.float32),
            "action": actions[i].astype(np.float32),
            "task": f"VR teleop session, {n} frames",
        })
    ds.save_episode()
    ds.finalize()
    print(f"[LeRobot] 已生成: datasets/{dataset_name}/  ({n} 帧)")


def _make_lerobot_with_vision(h5_path, video_path, dataset_name, joints_rad, actions, n):
    """生成含视觉的 LeRobot 数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import cv2

    N_JOINTS = joints_rad.shape[1]
    JOINT_NAMES = [
        "lift_joint", "waist1_joint", "waist2_joint",
        "J1_right_joint", "J2_right_joint", "J3_right_joint",
        "J4_right_joint", "J5_right_joint", "J6_right_joint", "J7_right_joint",
        "J1_left_joint", "J2_left_joint", "J3_left_joint",
        "J4_left_joint", "J5_left_joint", "J6_left_joint", "J7_left_joint",
    ][:N_JOINTS]

    cap = cv2.VideoCapture(str(video_path))
    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, f0 = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")
    h, w = f0.shape[:2]
    step = vid_frames / n

    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
        ret, fr = cap.read()
        if not ret:
            frames.append(frames[-1].copy())
        else:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"[VIDEO] 抽了 {len(frames)} 帧 ({w}x{h}), 视频共 {vid_frames} 帧")

    image_key = "observation.images.cam"
    features = {
        "observation.state": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (N_JOINTS,), "names": JOINT_NAMES},
        image_key: {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    }

    ds_path = DATASETS / dataset_name
    if ds_path.exists():
        import shutil
        shutil.rmtree(ds_path)

    ds = LeRobotDataset.create(
        repo_id=dataset_name, fps=30, features=features,
        root=str(ds_path), robot_type="F1", use_videos=False,
    )
    for i in range(n):
        ds.add_frame({
            "observation.state": joints_rad[i].astype(np.float32),
            "action": actions[i].astype(np.float32),
            image_key: frames[i],
            "task": f"VR teleop with vision, {n} frames",
        })
    ds.save_episode()
    ds.finalize()
    print(f"[LeRobot+Vision] 已生成: datasets/{dataset_name}/  ({n} 帧)")


# ---- 入口 ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 VR 同步录制")
    parser.add_argument("--port", type=int, default=9000, help="UDP 端口")
    parser.add_argument("--fps", type=int, default=30, help="录制帧率上限")
    parser.add_argument("--no-ik", action="store_true", help="跳过 IK")
    parser.add_argument("--video", type=str, default=None, help="mp4 视频路径（录制停止后生成含视觉数据集）")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称")
    args = parser.parse_args()

    result = record_session(args)
    if result:
        print(f"\n{'=' * 50}")
        print(f"录制完成!")
        print(f"  HDF5:     {result['h5_path']}")
        print(f"  LeRobot:  datasets/{result['dataset_name']}/")
        if "dataset_vision" in result:
            print(f"  +Vision:  datasets/{result['dataset_vision']}/")
        print(f"{'=' * 50}")
