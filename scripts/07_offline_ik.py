"""Offline IK: read recorded hand data from HDF5, solve IK, save joint angles.

Usage:
  python scripts/07_offline_ik.py --input recordings/episode_000000.h5 --dataset f1_vr_v1
"""

import numpy as np
import h5py
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ik_solver import F1Kinematics, hand_pose_to_robot_target

# 官方 LeRobot API（参照 openpi convert_aloha_data_to_lerobot.py）
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./recordings/episode_000000.h5")
parser.add_argument("--dataset", type=str, default="f1_vr_v1")
parser.add_argument("--task", type=str, default="pick up the cube")
args = parser.parse_args()

# Load URDF
urdf_path = Path(__file__).parent.parent / "urdf" / "urdf" / "F1_URDF_V04.urdf"
if not urdf_path.exists():
    urdf_path = Path("C:/Users/Administrator/Desktop/F1_URDF_V04/urdf/F1_URDF_V04.urdf")

print(f"[IK] Loading URDF: {urdf_path}")
f1 = F1Kinematics(str(urdf_path))

# Load HDF5
h5_path = Path(args.input)
print(f"\n[H5] Reading: {h5_path}")
with h5py.File(str(h5_path), "r") as f:
    n_frames = int(f.attrs["total_frames"])
    lw = f["observation.left_hand.wrist_pose"][:]
    rw = f["observation.right_hand.wrist_pose"][:]
    timestamps = f["timestamp"][:] if "timestamp" in f else np.arange(n_frames, dtype=np.float32)
    print(f"  Frames: {n_frames}")
    print(f"  Left wrist non-zero: {np.count_nonzero(lw)} / {lw.size}")
    print(f"  Right wrist non-zero: {np.count_nonzero(rw)} / {rw.size}")

all_joint_names = f1.get_all_joint_names()
n_joints = len(all_joint_names)
print(f"  Joints: {n_joints} -> {all_joint_names}")

# ---- IK Solve ----
print(f"\n[IK] Solving {n_frames} frames (offline, no rush)...")
joint_angles_all = np.zeros((n_frames, n_joints), dtype=np.float32)

right_guess = None
left_guess = None
right_ok = 0
left_ok = 0

for i in range(n_frames):
    # Right hand
    r_pos = rw[i][:3]
    r_quat = rw[i][3:7]
    if np.linalg.norm(r_pos) > 0.001:
        try:
            t_pos, t_quat = hand_pose_to_robot_target(r_pos, r_quat)
            ra = f1.solve_ik(t_pos, t_quat, side='right', initial_guess=right_guess)
            right_guess = ra.copy()
            right_ok += 1
        except:
            ra = np.zeros(f1.n_right)
    else:
        ra = np.zeros(f1.n_right)

    # Left hand
    l_pos = lw[i][:3]
    l_quat = lw[i][3:7]
    if np.linalg.norm(l_pos) > 0.001:
        try:
            t_pos, t_quat = hand_pose_to_robot_target(l_pos, l_quat)
            la = f1.solve_ik(t_pos, t_quat, side='left', initial_guess=left_guess)
            left_guess = la.copy()
            left_ok += 1
        except:
            la = np.zeros(f1.n_left)
    else:
        la = np.zeros(f1.n_left)

    torso = (ra[:f1.n_torso] + la[:f1.n_torso]) / 2.0
    right_arm = ra[f1.n_torso:]
    left_arm = la[f1.n_torso:]
    joint_angles_all[i] = np.concatenate([torso, right_arm, left_arm])

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{n_frames}... (right={right_ok}, left={left_ok})")

print(f"\n  Right IK solved: {right_ok}/{n_frames}")
print(f"  Left IK solved:  {left_ok}/{n_frames}")

# ---- Save enhanced HDF5 (raw backup, 弧度制) ----
h5_out = h5_path.with_suffix(".ik.h5")
with h5py.File(str(h5_out), "w") as f:
    with h5py.File(str(h5_path), "r") as src:
        for key in src.keys():
            src.copy(key, f)
        for key in src.attrs:
            f.attrs[key] = src.attrs[key]

    if "observation.state" in f:
        del f["observation.state"]
    if "action" in f:
        del f["action"]

    # action 比 state 超前一步（与 LeRobot 写入语义一致）
    joint_actions = np.zeros_like(joint_angles_all)
    joint_actions[:-1] = joint_angles_all[1:]          # action[i] = state[i+1]
    joint_actions[-1] = joint_angles_all[-1]            # 最后一帧指向自己

    f.create_dataset("observation.state", data=joint_angles_all)
    f.create_dataset("action", data=joint_actions)
    f.create_dataset("joint_names", data=np.array(all_joint_names, dtype=h5py.string_dtype()))
    f.attrs["ik_processed"] = True
    f.attrs["ik_right_solved"] = right_ok
    f.attrs["ik_left_solved"] = left_ok
    f.attrs["n_joints"] = n_joints

print(f"  HDF5+IK: {h5_out}")

# ---- LeRobot (official API, 参照 openpi convert_aloha_data_to_lerobot.py) ----
ds_root = Path("./datasets")
ds_repo_id = args.dataset
ds_full_path = ds_root / ds_repo_id

# 构建 features —— 关节名写进 names 字段，OpenPI DataConfig repack 会用到
joint_name_list = [str(j) for j in all_joint_names]
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (n_joints,),
        "names": [joint_name_list],
    },
    "action": {
        "dtype": "float32",
        "shape": (n_joints,),
        "names": [joint_name_list],
    },
}

# 创建或打开数据集（第一个 episode 用 create，后续追加用 open）
if ds_full_path.exists() and (ds_full_path / "meta" / "info.json").exists():
    print(f"[LeRobot] 打开已有数据集: {ds_full_path}")
    dataset = LeRobotDataset(ds_repo_id, root=str(ds_full_path))
else:
    print(f"[LeRobot] 创建新数据集: {ds_full_path}")
    dataset = LeRobotDataset.create(
        repo_id=ds_repo_id,
        fps=30,
        features=features,
        root=str(ds_full_path),
        robot_type="F1",
        use_videos=False,
    )

# 逐帧写入 —— action 比 state 超前一步（参照官方转换脚本的 action/state 语义）
for i in range(n_frames):
    next_i = min(i + 1, n_frames - 1)
    dataset.add_frame({
        "observation.state": joint_angles_all[i].astype(np.float32),
        "action": joint_angles_all[next_i].astype(np.float32),
        "task": args.task,
    })

dataset.save_episode()
dataset.finalize()

ep_idx = int(h5_path.stem.split("_")[-1])
print(f"  LeRobot: {ds_full_path}  (episode #{ep_idx} 已追加)")
print(f"\n=== DONE ===")
print(f"  Frames: {n_frames} -> {n_joints} DOF joint angles each")
print(f"  Right IK ok: {right_ok}/{n_frames}, Left IK ok: {left_ok}/{n_frames}")
print(f"  Sample (frame 0): {joint_angles_all[0][:3]}... (torso)")

# 验证 action shift
if n_frames >= 2:
    state_first = joint_angles_all[0]
    action_first = joint_angles_all[1]  # action[0] = state[1]
    if np.allclose(state_first, action_first):
        print("  [WARN] action[0] == state[0], shift 可能未生效")
    else:
        print(f"  [OK] action[0] != state[0], shift 生效")

# Quick check: are joint angles non-zero?
nonzero = np.count_nonzero(joint_angles_all)
print(f"  Non-zero joints: {nonzero} / {joint_angles_all.size}")
