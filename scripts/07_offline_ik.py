"""Offline IK: read recorded hand data from HDF5, solve IK, save joint angles.

Usage:
  python scripts/07_offline_ik.py --input recordings/episode_000000.h5 --dataset f1_vr_v1
"""

import numpy as np
import h5py
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ik_solver import F1Kinematics, hand_pose_to_robot_target

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

# ---- Save enhanced HDF5 ----
h5_out = h5_path.with_suffix(".ik.h5")
with h5py.File(str(h5_out), "w") as f:
    # Copy original datasets
    with h5py.File(str(h5_path), "r") as src:
        for key in src.keys():
            src.copy(key, f)
        for key in src.attrs:
            f.attrs[key] = src.attrs[key]

    # Add/overwrite joint angle data
    if "observation.state" in f:
        del f["observation.state"]
    if "action" in f:
        del f["action"]
    f.create_dataset("observation.state", data=joint_angles_all)
    f.create_dataset("action", data=joint_angles_all)
    f.create_dataset("joint_names", data=np.array(all_joint_names, dtype=h5py.string_dtype()))
    f.attrs["ik_processed"] = True
    f.attrs["ik_right_solved"] = right_ok
    f.attrs["ik_left_solved"] = left_ok
    f.attrs["n_joints"] = n_joints

print(f"  HDF5+IK: {h5_out}")

# ---- LeRobot ----
ds_dir = Path(f"./datasets/{args.dataset}")
(ds_dir / "meta").mkdir(parents=True, exist_ok=True)
(ds_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

# extract episode index from input filename
ep_idx = int(h5_path.stem.split("_")[-1])  # e.g. episode_000001.h5 -> 1

records = []
for i in range(n_frames):
    records.append({
        "observation.state": joint_angles_all[i].tolist(),
        "action": joint_angles_all[i].tolist(),
        "timestamp": float(timestamps[i]),
        "episode_index": int(ep_idx),
        "index": i,
        "task_index": 0,
        "next.done": (i == n_frames - 1),
        "next.reward": 1.0 if i == n_frames - 1 else 0.0,
    })

df = pd.DataFrame(records)
pq_path = ds_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
df.to_parquet(str(pq_path), engine="pyarrow")

features = {k: v for k, v in {
    "observation.state": {"dtype": "float32", "shape": [n_joints]},
    "action": {"dtype": "float32", "shape": [n_joints]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "next.done": {"dtype": "bool", "shape": [1]},
    "next.reward": {"dtype": "float32", "shape": [1]},
}.items()}

info = {
    "codebase_version": "v3.0",
    "robot_type": "F1",
    "fps": 30,
    "total_episodes": 1,
    "total_frames": int(n_frames),
    "features": features,
    "joint_names": [str(j) for j in all_joint_names],
    "n_joints": int(n_joints),
}

with open(ds_dir / "meta" / "info.json", "w") as f:
    json.dump(info, f, indent=2)
with open(ds_dir / "meta" / "episodes.jsonl", "w") as f:
    f.write(json.dumps({"episode_index": 0, "tasks": [args.task], "length": n_frames}) + "\n")
with open(ds_dir / "meta" / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

print(f"  LeRobot: {ds_dir}")
print(f"\n=== DONE ===")
print(f"  Frames: {n_frames} -> {n_joints} DOF joint angles each")
print(f"  Right IK ok: {right_ok}/{n_frames}, Left IK ok: {left_ok}/{n_frames}")
print(f"  Sample (frame 0): {joint_angles_all[0][:3]}... (torso)")

# Quick check: are joint angles non-zero?
nonzero = np.count_nonzero(joint_angles_all)
print(f"  Non-zero joints: {nonzero} / {joint_angles_all.size}")
